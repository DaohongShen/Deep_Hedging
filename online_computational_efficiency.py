"""Online computational efficiency extension for the Thalesians Heston project.

Place this script alongside evaluation.py, state_inputs.py, baselines.py,
    deep_learning_policy.py and heston_model_setup.py.

The PRETRAINED GRU and feature scaler must already exist under
    week2_task/evaluation_results/gru_policy.pt
    week2_task/evaluation_results/feature_scaler.npz
Run: python online_computational_efficiency.py
Quick: python online_computational_efficiency.py --paths 64 --batch-sizes 1,16,64 --repeats 15 --total-repeats 1

Measures end-to-end COMPUTATION only: from already-received observed stock
prices (and oracle variance for Heston) to hedge ratio, including feature
engineering/standardisation, tensor conversion, inference and output transfer.
No market-data fetching, broker execution, calibration or model loading is timed.
Full-sequence GRU recomputes all observed features and hidden states at every
hedge date; stateful GRU preserves the hidden state and computes ONLY the new
observation's features. Both use the exact same trained model and scaler.

Online total times include serial 63-date processing over a held-out set. They
are measurements, not extrapolations from single-request latencies. Model
batching groups independent simulation paths at each date; persistent GRU
hidden states remain separate for each batch. Heston is an ORACLE with latent
variance, which is not observable in actual markets.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import torch

from baselines import HestonParameters, black_scholes_delta, heston_delta
from deep_learning_policy import GRUPolicy
from heston_model_setup import K, T, r, theta, kappa, xi, rho, n_steps, S0, v0, simulate_heston


def simulate_set(n_paths: int, seed: int):
    _, stock, variance, _ = simulate_heston(S0=S0, v0=v0, r=r, kappa=kappa,
        theta=theta, xi=xi, rho=rho, T=T, n_steps=n_steps,
        n_paths=n_paths, seed=seed)
    return np.asarray(stock, dtype=np.float32), np.asarray(variance, dtype=np.float32)
from state_inputs import StateStandardScaler, build_state_inputs

ROOT = Path(__file__).resolve().parent
METHODS = ("BS Delta", "Heston Delta (Oracle)", "GRU Full-sequence", "GRU Stateful")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--evaluation-dir", type=Path, default=ROOT / "evaluation_results")
    p.add_argument("--out-dir", type=Path, default=ROOT / "online_computational_efficiency_results")
    p.add_argument("--paths", type=int, default=512, help="Held-out simulation paths; increase to 3000 for full evaluation size")
    p.add_argument("--batch-sizes", type=str, default="1,16,64,256")
    p.add_argument("--repeats", type=int, default=75, help="Measured repeats per single-date request")
    p.add_argument("--warmup", type=int, default=12)
    p.add_argument("--total-repeats", type=int, default=2, help="Full-set repetitions per batch size/method")
    p.add_argument("--seed", type=int, default=2030, help="2026+4 reproduces base evaluation test paths")
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--torch-threads", type=int, default=1)
    p.add_argument("--skip-heston", action="store_true")
    return p.parse_args()


def setup(args):
    if args.paths < 2 or args.repeats < 2 or args.warmup < 1 or args.total_repeats < 1:
        raise ValueError("paths>=2, repeats>=2, warmup>=1, total-repeats>=1 are required")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    if args.torch_threads < 1:
        raise ValueError("--torch-threads must be positive")
    torch.set_num_threads(args.torch_threads)
    batch_sizes = sorted({int(s.strip()) for s in args.batch_sizes.split(",") if s.strip()})
    if not batch_sizes or min(batch_sizes) < 1:
        raise ValueError("--batch-sizes must contain positive integers")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    return batch_sizes


def load_pretrained(eval_dir: Path, device: str):
    weights = eval_dir / "gru_policy.pt"
    scaler_file = eval_dir / "feature_scaler.npz"
    for filename in (weights, scaler_file):
        if not filename.is_file():
            raise FileNotFoundError(f"Missing {filename}. Run evaluation.py first; this script does NOT retrain.")
    checkpoint = torch.load(weights, map_location="cpu", weights_only=True)
    model = GRUPolicy(input_size=int(checkpoint["input_size"]),
                      hidden_size=int(checkpoint["hidden_size"]),
                      num_layers=int(checkpoint["num_layers"]))
    model.load_state_dict(checkpoint["state_dict"])
    model = model.to(device).eval()
    with np.load(scaler_file) as z:
        scaler = StateStandardScaler(mean_=z["mean"].astype(np.float32),
                                     scale_=z["scale"].astype(np.float32))
    if model.gru.input_size != 6 or scaler.mean_.shape != (6,):
        raise ValueError("Expected original six-feature GRU checkpoint and scaler")
    return model, scaler


def observed_features(prices: np.ndarray, dt: float, start_index: int = 0) -> np.ndarray:
    """Exact original six features for just the observed prefix.

    prices: (batch, t+1) with indices 0,...,t. For a new-only observation,
    callers can supply the last 21 prices and specify the absolute start_index;
    here we handle the new-only case separately to avoid any ambiguity.
    """
    if start_index != 0:
        raise ValueError("Prefix feature calculation must start at time zero")
    b, n = prices.shape
    log_r = np.zeros((b, n), dtype=np.float64)
    if n > 1:
        log_r[:, 1:] = np.log(prices[:, 1:] / prices[:, :-1])
    rv5, rv20 = np.zeros_like(log_r), np.zeros_like(log_r)
    for t in range(2, n):
        rv5[:, t] = log_r[:, max(1, t-4):t+1].std(axis=1, ddof=0) / math.sqrt(dt)
        rv20[:, t] = log_r[:, max(1, t-19):t+1].std(axis=1, ddof=0) / math.sqrt(dt)
    tau = np.maximum(T - np.arange(n) * dt, 0.)
    return np.stack((prices, log_r, rv5, rv20,
                     np.broadcast_to(tau, prices.shape), prices / K), axis=-1).astype(np.float32)


def newest_features(prices: np.ndarray, date: int, dt: float) -> np.ndarray:
    """Only the new date's feature vector; uses at most the most recent 21 prices."""
    b = prices.shape[0]
    last = np.asarray(prices[:, -1], dtype=np.float64)
    ret = np.zeros(b)
    rv5, rv20 = np.zeros(b), np.zeros(b)
    if date > 0:
        ret = np.log(last / prices[:, -2])
    if date >= 2:
        for window, dest in ((5, rv5), (20, rv20)):
            k = min(date, window)
            recent = np.asarray(prices[:, -(k+1):], dtype=np.float64)
            returns = np.log(recent[:, 1:] / recent[:, :-1])
            dest[:] = returns.std(axis=1, ddof=0) / math.sqrt(dt)
    return np.stack((last, ret, rv5, rv20,
                     np.full(b, max(T - date*dt, 0.)), last/K), axis=-1).astype(np.float32)


class OnlineHedger:
    def __init__(self, model, scaler, device, use_heston):
        self.model, self.scaler, self.device = model, scaler, device
        self.dt = T / n_steps
        self.heston_params = HestonParameters(r=r, kappa=kappa, theta=theta, xi=xi, rho=rho)
        self.use_heston = use_heston

    @torch.inference_mode()
    def step(self, method: str, price_prefix: np.ndarray, date: int,
             variance_now: np.ndarray, hidden=None):
        spot = price_prefix[:, -1]
        tau = float(T - date * self.dt)
        if method == "BS Delta":
            return black_scholes_delta(spot, K, tau, r, math.sqrt(theta)), None
        if method == "Heston Delta (Oracle)":
            return heston_delta(spot, np.maximum(variance_now, 0.), K, tau,
                                self.heston_params, n_quadrature=32,
                                integration_upper=150.), None
        if method == "GRU Full-sequence":
            feats = observed_features(price_prefix, self.dt)
            norm = self.scaler.transform(feats)
            states = torch.from_numpy(np.ascontiguousarray(norm)).to(self.device)
            positions = self.model(states)
            return positions[:, -1].detach().cpu().numpy(), None
        if method == "GRU Stateful":
            feats = newest_features(price_prefix, date, self.dt)[:, None, :]
            norm = self.scaler.transform(feats)
            states = torch.from_numpy(np.ascontiguousarray(norm)).to(self.device)
            output, h_next = self.model.gru(states, hidden)
            positions = self.model.output_layer(output[:, -1, :]).squeeze(-1)
            return positions.detach().cpu().numpy(), h_next
        raise ValueError(f"Unknown method {method}")


def sync(device):
    if device == "cuda":
        torch.cuda.synchronize()


def verify_equivalence(hedger, stocks, variances):
    """Test feature equality against ORIGINAL implementation and policy equality."""
    subset = stocks[:min(8, len(stocks))]
    expected = build_state_inputs(subset, strike=K, maturity=T, include_terminal=False)
    actual = observed_features(subset[:, :n_steps], T/n_steps)
    feature_err = float(np.max(np.abs(expected - actual)))
    if not np.allclose(expected, actual, rtol=1e-5, atol=2e-5):
        raise AssertionError(f"Feature builder differs from original: max abs {feature_err}")
    h = None
    errors = []
    for t in range(n_steps):
        hist = subset[:, :t+1]
        phi_full, _ = hedger.step("GRU Full-sequence", hist, t, variances[:len(subset), t])
        phi_state, h = hedger.step("GRU Stateful", hist, t, variances[:len(subset), t], h)
        errors.append(float(np.max(np.abs(phi_full - phi_state))))
    max_diff = max(errors)
    if max_diff > 1e-4:
        raise AssertionError(f"GRU full and stateful policies disagree (max diff {max_diff})")
    return {"max_feature_difference": feature_err, "max_gru_position_difference": max_diff}


def prefill_hidden(hedger, prices, variances, date):
    """Prepare genuine state from past observations. NOT included in today's latency."""
    h = None
    for t in range(date):
        _, h = hedger.step("GRU Stateful", prices[:, :t+1], t, variances[:, t], h)
    return h


def single_request_stats(hedger, method, prices, variances, date, batch, warmup, repeats):
    prefix = prices[:batch, :date+1]
    variance_now = variances[:batch, date]
    # For stateful requests, the past state is already available at the current date;
    # building it here is a one-off setup, NOT charged to each new quote.
    h0 = prefill_hidden(hedger, prices[:batch], variances[:batch], date) if method == "GRU Stateful" else None
    for _ in range(warmup):
        hedger.step(method, prefix, date, variance_now, h0)
    sync(hedger.device)
    samples = []
    for _ in range(repeats):
        sync(hedger.device)
        t0 = time.perf_counter_ns()
        hedger.step(method, prefix, date, variance_now, h0)
        sync(hedger.device)
        samples.append((time.perf_counter_ns()-t0)/1e6)
    samples = np.array(samples)
    return {"method": method, "batch_size": batch, "date": date,
            "mean_ms": float(samples.mean()), "median_ms": float(np.median(samples)),
            "p95_ms": float(np.percentile(samples, 95)),
            "p99_ms": float(np.percentile(samples, 99)),
            "std_ms": float(samples.std()),
            "min_ms": float(samples.min()),
            "max_ms": float(samples.max()),
            "single_request_decisions_per_sec": float(batch*1000/np.mean(samples))}, samples


def run_complete_test(hedger, method, stocks, variances, batch_size):
    """Time all dates and paths, including model inference AND feature work."""
    sync(hedger.device)
    t0 = time.perf_counter_ns()
    n = len(stocks)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        s, v = stocks[start:end], variances[start:end]
        h = None
        for t in range(n_steps):
            _, h = hedger.step(method, s[:, :t+1], t, v[:, t], h)
    sync(hedger.device)
    seconds = (time.perf_counter_ns()-t0)*1e-9
    decisions = n*n_steps
    return seconds, decisions/seconds


def chart_results(latency, totals, out):
    fig, ax = plt.subplots(figsize=(8, 4.7))
    for name, group in latency.groupby("method", sort=False):
        group = group.sort_values("batch_size")
        ax.plot(group.batch_size, group.median_ms, "o-", label=name)
    ax.set(xlabel="Batch size (independent hedge requests)",
           ylabel="Median request latency (ms)", title="End-to-end hedge-ratio latency")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.grid(alpha=.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "latency_vs_batch_size.png", dpi=170)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4.7))
    for name, group in totals.groupby("method", sort=False):
        group = group.sort_values("batch_size")
        ax.plot(group.batch_size, group.throughput_decisions_per_sec, "o-", label=name)
    ax.set(xlabel="Batch size (independent simulation paths)",
           ylabel="Hedge decisions per second", title="Full out-of-sample throughput")
    ax.set_xscale("log", base=2)
    ax.grid(alpha=.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "throughput_vs_batch_size.png", dpi=170)
    plt.close(fig)


def save_quality_comparison(eval_dir, latency, out):
    path = eval_dir / "comparison.csv"
    if not path.exists():
        print("Note: evaluation_results/comparison.csv not found; omitting measured quality plot.")
        return
    quality = pd.read_csv(path)
    if "scenario" in quality.columns:
        quality = quality[quality.scenario == "base"]
    name_map = {"BS Delta": "BS Delta", "Black-Scholes Delta": "BS Delta",
                "Heston Delta (Oracle)": "Heston Delta (Oracle)",
                "GRU Deep Hedging": "GRU Full-sequence"}
    quality = quality[quality.strategy.isin(name_map)].copy()
    quality["method"] = quality.strategy.map(name_map)
    stateful = quality[quality.method == "GRU Full-sequence"].copy()
    stateful["method"] = "GRU Stateful"
    quality = pd.concat([quality, stateful], ignore_index=True)
    reference = latency[latency.batch_size == 1][["method", "median_ms", "p95_ms"]]
    joined = quality.merge(reference, on="method", how="inner")
    joined.to_csv(out / "quality_vs_latency.csv", index=False)
    if "rmse" not in joined.columns or joined.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 4.7))
    for _, item in joined.iterrows():
        ax.scatter(item.median_ms, item.rmse, s=60)
        ax.annotate(item.method, (item.median_ms, item.rmse), xytext=(5, 5),
                    textcoords="offset points", fontsize=8)
    ax.set(xlabel="Median online single-request latency (ms)",
           ylabel="Out-of-sample hedging RMSE",
           title="Hedging quality vs online computation")
    ax.set_xscale("log")
    ax.grid(alpha=.25)
    fig.tight_layout()
    fig.savefig(out / "quality_vs_latency.png", dpi=170)
    plt.close(fig)


def main():
    args = parse_args()
    batch_sizes = setup(args)
    model, scaler = load_pretrained(args.evaluation_dir, args.device)
    hedger = OnlineHedger(model, scaler, args.device, not args.skip_heston)
    stocks, variances = simulate_set(args.paths, args.seed)
    checks = verify_equivalence(hedger, stocks, variances)
    methods = [m for m in METHODS if not (args.skip_heston and "Heston" in m)]
    hardware = {"platform": platform.platform(), "cpu": platform.processor(),
                "cpu_count": os.cpu_count(), "cuda_device":
                torch.cuda.get_device_name(0) if args.device == "cuda" else None,
                "device": args.device, "torch_threads": args.torch_threads,
                "torch": torch.__version__, "numpy": np.__version__, "scipy": scipy.__version__,
                "precision_gru": "float32", "precision_delta": "float64",
                "seed": args.seed, "paths": args.paths, "decisions_per_path": n_steps,
                "batch_sizes": batch_sizes, "warmup": args.warmup,
                "single_request_repeats": args.repeats,
                "full_set_repeats": args.total_repeats,
                "timed_boundary": "observed arrays -> feature engineering -> standardisation -> inference -> CPU output",
                "excluded": "simulation, model load, calibration, offline training, data transfer from market, broker execution",
                "single_request_date": n_steps//2,
                "stateful_request_setup": "hidden state from prior observations prefilled outside single-request timing",
                "checks": checks}
    (args.out_dir / "benchmark_metadata.json").write_text(json.dumps(hardware, indent=2, default=str))
    print("Hardware:", hardware["platform"], "device:", args.device)
    print("Validation:", checks)
    latency_rows, samples_rows, total_rows = [], [], []
    for batch in batch_sizes:
        effective = min(batch, args.paths)
        for method in methods:
            print(f"Benchmark {method} | batch={effective}", flush=True)
            stats, raw = single_request_stats(hedger, method, stocks, variances,
                                               n_steps//2, effective, args.warmup, args.repeats)
            latency_rows.append(stats)
            samples_rows.extend({"method": method, "batch_size": effective,
                                 "repeat": i+1, "latency_ms": float(ms)} for i, ms in enumerate(raw))
            for repetition in range(args.total_repeats):
                secs, throughput = run_complete_test(hedger, method, stocks, variances, effective)
                total_rows.append({"method": method, "batch_size": effective,
                                   "repeat": repetition+1, "total_seconds": secs,
                                   "throughput_decisions_per_sec": throughput,
                                   "decisions": args.paths*n_steps})
    latency = pd.DataFrame(latency_rows)
    totals_raw = pd.DataFrame(total_rows)
    totals = totals_raw.groupby(["method", "batch_size"], as_index=False).agg(
        total_seconds=("total_seconds", "median"),
        throughput_decisions_per_sec=("throughput_decisions_per_sec", "median"),
        time_std_seconds=("total_seconds", "std"))
    latency.to_csv(args.out_dir / "latency_summary.csv", index=False)
    pd.DataFrame(samples_rows).to_csv(args.out_dir / "latency_samples.csv", index=False)
    totals_raw.to_csv(args.out_dir / "full_set_repetitions.csv", index=False)
    totals.to_csv(args.out_dir / "throughput_summary.csv", index=False)
    chart_results(latency, totals, args.out_dir)
    save_quality_comparison(args.evaluation_dir, latency, args.out_dir)
    print("\nSingle-request median latency (ms):")
    print(latency.pivot(index="method", columns="batch_size", values="median_ms").round(4).to_string())
    print("\nFull-set throughput (decisions/second):")
    print(totals.pivot(index="method", columns="batch_size", values="throughput_decisions_per_sec").round(1).to_string())
    print("\nOutputs saved to:", args.out_dir.resolve())


if __name__ == "__main__":
    main()
