"""Part 6 — Train and evaluate a GRU-based Heston deep-hedging policy.

Place beside:
    heston_model_setup.py, hedging_objective.py, deep_learning_policy.py,
    state_inputs.py, baselines.py

Install: pip install numpy scipy pandas matplotlib torch
Run:     python evaluation.py
Fast:    python evaluation.py --epochs 3 --train-paths 1000 --test-paths 200 --skip-heston

Evaluation is out of sample. The training, validation, premium-pricing and test
simulations use independent random seeds; the feature scaler fits training only.
"""
from __future__ import annotations

import argparse
import copy
import json
import random
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from heston_model_setup import (
    S0, v0, r, kappa, theta, xi, rho, T, K, n_steps,
    simulate_heston,
)
from hedging_objective import (
    FULL_BID_ASK_SPREAD, COMMISSION_RATE,
    hedge_terminal_error, mean_variance_loss,
)
from deep_learning_policy import GRUPolicy
from state_inputs import build_state_inputs, StateStandardScaler
from baselines import (
    HestonParameters,
    no_hedging_policy,
    bs_delta_hedging_policy,
    heston_delta_hedging_policy,
)


@dataclass(frozen=True)
class Config:
    train_paths: int = 12000
    val_paths: int = 2000
    test_paths: int = 3000
    pricing_paths: int = 30000
    epochs: int = 35
    batch_size: int = 512
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    patience: int = 7
    hidden_size: int = 64
    num_layers: int = 1
    grad_clip: float = 1.0
    seed: int = 2026
    full_spread: float = FULL_BID_ASK_SPREAD
    commission_rate: float = COMMISSION_RATE
    cvar_alpha: float = 0.95
    heston_quadrature: int = 32
    heston_integration_upper: float = 150.0
    include_heston: bool = True
    run_stress: bool = True
    out_dir: str = "evaluation_results"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def simulate_set(n_paths: int, seed: int, *, rho_: float = rho,
                 xi_: float = xi, n_obs: int = n_steps + 1):
    """Request n_obs observation times (initial time included).

    The existing Heston simulator interprets n_steps as time intervals,
    returning n_steps + 1 observation points.
    """
    times, stock, variance, _raw_variance = simulate_heston(
        S0=S0, v0=v0, r=r, kappa=kappa, theta=theta, xi=xi_,
        rho=rho_, T=T, n_steps=n_obs - 1, n_paths=n_paths, seed=seed,
    )
    stock = np.asarray(stock, dtype=np.float32)
    variance = np.asarray(variance, dtype=np.float32)
    if stock.shape != (n_paths, n_obs) or variance.shape != stock.shape:
        raise ValueError(f"Unexpected simulator shapes: S={stock.shape}, v={variance.shape}; "
                         f"expected {(n_paths, n_obs)}")
    return stock, variance


def mc_option_premium(stock: np.ndarray) -> float:
    """Independent risk-neutral MC pricing, not fitted to test outcomes."""
    return float(np.exp(-r * T) * np.maximum(stock[:, -1] - K, 0).mean())


def make_features(stock: np.ndarray, scaler: StateStandardScaler,
                  *, fit: bool = False) -> np.ndarray:
    x = build_state_inputs(stock, strike=K, maturity=T, include_terminal=False)
    x = scaler.fit_transform(x) if fit else scaler.transform(x)
    if x.shape[:2] != (stock.shape[0], stock.shape[1] - 1):
        raise ValueError(f"Feature shape mismatch: {x.shape}")
    return np.asarray(x, dtype=np.float32)


def account(stock: torch.Tensor, positions: torch.Tensor, premium: float,
            cfg: Config):
    return hedge_terminal_error(
        mid_prices=stock,
        hedge_positions=positions,
        option_premium=premium,
        strike=K,
        maturity=T,
        risk_free_rate=r,
        full_spread=cfg.full_spread,
        commission_rate=cfg.commission_rate,
    )


def train_gru(cfg: Config, premium: float, scaler: StateStandardScaler,
              train_stock: np.ndarray, val_stock: np.ndarray):
    x_train = make_features(train_stock, scaler, fit=True)
    x_val = make_features(val_stock, scaler)
    dataset = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(train_stock))
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)
    model = GRUPolicy(input_size=x_train.shape[-1], hidden_size=cfg.hidden_size,
                      num_layers=cfg.num_layers).to(cfg.device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate,
                           weight_decay=cfg.weight_decay)
    x_val_t = torch.from_numpy(x_val).to(cfg.device)
    s_val_t = torch.from_numpy(val_stock).to(cfg.device)
    best_weights = copy.deepcopy(model.state_dict())
    best_val, stale = float("inf"), 0
    history = []
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        total, count = 0.0, 0
        for x_batch, stock_batch in loader:
            x_batch = x_batch.to(cfg.device)
            stock_batch = stock_batch.to(cfg.device)
            opt.zero_grad(set_to_none=True)
            positions = model(x_batch)
            result = account(stock_batch, positions, premium, cfg)
            loss = mean_variance_loss(result.hedging_error)
            if not torch.isfinite(loss):
                raise FloatingPointError("Training loss is NaN/Inf; check simulation and features")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            total += loss.item() * len(x_batch)
            count += len(x_batch)

        model.eval()
        with torch.no_grad():
            # Fixed held-out validation paths; no model updates or scaler refit.
            val_losses = []
            for start in range(0, len(val_stock), cfg.batch_size):
                end = start + cfg.batch_size
                p = model(x_val_t[start:end])
                result = account(s_val_t[start:end], p, premium, cfg)
                val_losses.append(result.hedging_error.square().sum().item())
            val_loss = sum(val_losses) / len(val_stock)
        train_loss = total / count
        history.append({"epoch": epoch, "train_mse": train_loss, "val_mse": val_loss})
        print(f"Epoch {epoch:03d} | train MSE={train_loss:.6f} | val MSE={val_loss:.6f}")
        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_weights = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= cfg.patience:
                print("Early stopping: validation MSE did not improve.")
                break
    model.load_state_dict(best_weights)
    return model, pd.DataFrame(history)


@torch.no_grad()
def predict_gru(model: GRUPolicy, stock: np.ndarray,
                scaler: StateStandardScaler, cfg: Config) -> np.ndarray:
    x = make_features(stock, scaler)
    model.eval()
    positions = []
    for offset in range(0, len(stock), cfg.batch_size):
        batch = torch.from_numpy(x[offset:offset + cfg.batch_size]).to(cfg.device)
        positions.append(model(batch).cpu().numpy())
    return np.concatenate(positions, axis=0)


def evaluate_positions(stock: np.ndarray, positions: np.ndarray,
                       premium: float, cfg: Config) -> dict:
    s = torch.as_tensor(stock, dtype=torch.float32, device=cfg.device)
    p = torch.as_tensor(positions, dtype=torch.float32, device=cfg.device)
    if p.shape != (s.shape[0], s.shape[1] - 1):
        raise ValueError(f"Position shape {tuple(p.shape)} not compatible with "
                         f"stock {tuple(s.shape)}")
    with torch.no_grad():
        result = account(s, p, premium, cfg)
    return {
        "error": result.hedging_error.detach().cpu().numpy(),
        "spread": result.total_spread_cost.detach().cpu().numpy(),
        "commission": result.total_commission.detach().cpu().numpy(),
        "turnover": result.total_turnover.detach().cpu().numpy(),
    }


def metrics(out: dict, alpha: float) -> dict:
    err = np.asarray(out["error"], dtype=np.float64).ravel()
    loss = -err
    var = float(np.quantile(loss, alpha))
    # Empirical CVaR: average of the worst (1-alpha) portion, including
    # a fractional boundary observation when required.
    worst_count = max((1.0 - alpha) * len(loss), 1.0)
    sorted_loss = np.sort(loss)[::-1]
    n_whole = int(np.floor(worst_count))
    tail_sum = sorted_loss[:n_whole].sum()
    if n_whole < len(loss):
        tail_sum += (worst_count - n_whole) * sorted_loss[n_whole]
    cvar = float(tail_sum / worst_count)
    centered = err - err.mean()
    sd = float(err.std())
    skew = float(np.mean(centered**3) / sd**3) if sd > 1e-12 else 0.0
    kurtosis = float(np.mean(centered**4) / sd**4) if sd > 1e-12 else 0.0
    return {
        "mean_error": float(err.mean()),
        "std_error": sd,
        "rmse": float(np.sqrt(np.mean(err**2))),
        "VaR_loss": var,
        "CVaR_loss": cvar,
        "skewness": skew,
        "kurtosis": kurtosis,
        "mean_spread_cost": float(np.mean(out["spread"])),
        "mean_commission": float(np.mean(out["commission"])),
        "mean_total_cost": float(np.mean(out["spread"] + out["commission"])),
        "mean_turnover": float(np.mean(out["turnover"])),
    }


def baseline_positions(stock: np.ndarray, variance: np.ndarray,
                       cfg: Config, *, rho_: float = rho, xi_: float = xi):
    result = {
        "No Hedging": no_hedging_policy(stock),
        "BS Delta": bs_delta_hedging_policy(
            stock, strike=K, maturity=T, risk_free_rate=r,
            volatility=float(np.sqrt(theta)),
        ),
    }
    if cfg.include_heston:
        params = HestonParameters(r=r, kappa=kappa, theta=theta, xi=xi_, rho=rho_)
        result["Heston Delta (Oracle)"] = heston_delta_hedging_policy(
            stock, variance, strike=K, maturity=T, params=params,
            n_quadrature=cfg.heston_quadrature,
            integration_upper=cfg.heston_integration_upper,
        )
    return result


def run_comparison(label: str, model: GRUPolicy, scaler: StateStandardScaler,
                   stock: np.ndarray, variance: np.ndarray,
                   premium: float, cfg: Config, *, rho_: float = rho,
                   xi_: float = xi):
    policies = baseline_positions(stock, variance, cfg, rho_=rho_, xi_=xi_)
    policies["GRU Deep Hedging"] = predict_gru(model, stock, scaler, cfg)
    results, rows = {}, []
    for name, positions in policies.items():
        results[name] = evaluate_positions(stock, np.asarray(positions), premium, cfg)
        rows.append({"scenario": label, "strategy": name,
                     **metrics(results[name], cfg.cvar_alpha)})
    return pd.DataFrame(rows), results


def save_plots(history: pd.DataFrame, comparison: pd.DataFrame,
               details: dict, outdir: Path):
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.plot(history["epoch"], history["train_mse"], label="Train MSE")
    ax.plot(history["epoch"], history["val_mse"], label="Validation MSE")
    ax.set(xlabel="Epoch", ylabel="Mean squared hedging error", title="GRU training")
    ax.legend(); fig.tight_layout(); fig.savefig(outdir / "training_curve.png", dpi=170)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.0, 4.4))
    for name, output in details.items():
        ax.hist(output["error"], bins=65, density=True, alpha=0.38, label=name)
    ax.set(xlabel="Terminal hedging error", ylabel="Density",
           title="Out-of-sample hedging error distributions")
    ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig(outdir / "hedging_error_histogram.png", dpi=170); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.0, 4.4))
    names = comparison["strategy"].tolist()
    xs = np.arange(len(names)); width = 0.37
    ax.bar(xs-width/2, comparison["rmse"], width, label="RMSE")
    ax.bar(xs+width/2, comparison["CVaR_loss"], width, label="CVaR of loss")
    ax.set_xticks(xs, names, rotation=15, ha="right")
    ax.set(title="Baseline comparison", ylabel="Amount (price currency)")
    ax.legend(); fig.tight_layout(); fig.savefig(outdir / "baseline_comparison.png", dpi=170)
    plt.close(fig)


def stress_tests(model: GRUPolicy, scaler: StateStandardScaler,
                 premium: float, cfg: Config) -> pd.DataFrame:
    """Test the SAME pretrained model under shocks; do not retrain it."""
    all_frames = []
    scenarios = [
        ("rho_-0.90", -0.90, xi, cfg.full_spread, cfg.commission_rate, n_steps + 1),
        ("rho_-0.30", -0.30, xi, cfg.full_spread, cfg.commission_rate, n_steps + 1),
        ("xi_0.50", rho, 0.50, cfg.full_spread, cfg.commission_rate, n_steps + 1),
        ("high_trading_costs", rho, xi, cfg.full_spread*3,
         cfg.commission_rate*3, n_steps + 1),
        ("weekly_rehedging", rho, xi, cfg.full_spread,
         cfg.commission_rate, max(2, round(n_steps/5)+1)),
    ]
    for i, (label, rho_, xi_, spread, commission, n_obs) in enumerate(scenarios):
        print(f"Stress test: {label}")
        s, v = simulate_set(cfg.test_paths, cfg.seed+600+i,
                            rho_=rho_, xi_=xi_, n_obs=n_obs)
        scenario_cfg = replace(cfg, full_spread=spread, commission_rate=commission)
        frame, _ = run_comparison(label, model, scaler, s, v, premium,
                                  scenario_cfg, rho_=rho_, xi_=xi_)
        all_frames.append(frame)
    return pd.concat(all_frames, ignore_index=True)


def main(cfg: Config):
    if min(cfg.train_paths, cfg.val_paths, cfg.test_paths, cfg.pricing_paths) < 2:
        raise ValueError("Every simulation size must be >=2")
    if not (0 < cfg.cvar_alpha < 1):
        raise ValueError("cvar_alpha must be strictly between 0 and 1")
    seed_all(cfg.seed)
    output = Path(cfg.out_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.json").write_text(json.dumps(asdict(cfg), indent=2))
    print(f"Device: {cfg.device}; training GRU before evaluation")

    # IMPORTANT: the option premium is estimated on an independent pricing set.
    pricing_stock, _ = simulate_set(cfg.pricing_paths, cfg.seed+1)
    premium = mc_option_premium(pricing_stock)
    del pricing_stock
    print(f"Independent MC option premium = {premium:.6f}")

    train_stock, _ = simulate_set(cfg.train_paths, cfg.seed+2)
    val_stock, _ = simulate_set(cfg.val_paths, cfg.seed+3)
    scaler = StateStandardScaler()
    model, history = train_gru(cfg, premium, scaler, train_stock, val_stock)
    history.to_csv(output / "training_history.csv", index=False)
    torch.save({"state_dict": model.state_dict(), "input_size": 6,
                "hidden_size": cfg.hidden_size, "num_layers": cfg.num_layers},
               output / "gru_policy.pt")
    np.savez(output / "feature_scaler.npz", mean=scaler.mean_, scale=scaler.scale_)
    del train_stock, val_stock

    test_stock, test_var = simulate_set(cfg.test_paths, cfg.seed+4)
    comparison, details = run_comparison("base", model, scaler,
                                         test_stock, test_var, premium, cfg)
    comparison.to_csv(output / "comparison.csv", index=False)
    np.savez_compressed(output / "test_hedging_errors.npz",
                        **{name.replace(" ", "_"): res["error"]
                           for name, res in details.items()})
    save_plots(history, comparison, details, output)
    print("\nOut-of-sample comparison:\n", comparison.to_string(index=False))

    if cfg.run_stress:
        stress = stress_tests(model, scaler, premium, cfg)
        stress.to_csv(output / "stress_tests.csv", index=False)
        fig, ax = plt.subplots(figsize=(10, 5))
        for name, sub in stress.groupby("strategy"):
            ax.plot(sub["scenario"], sub["rmse"], marker="o", label=name)
        ax.set(xlabel="Stress scenario", ylabel="RMSE", title="Out-of-distribution stress tests")
        ax.tick_params(axis="x", rotation=25)
        ax.legend(fontsize=8); fig.tight_layout()
        fig.savefig(output / "stress_test_rmse.png", dpi=170); plt.close(fig)
    print(f"\nSaved results to: {output.resolve()}")


def parse_args():
    p = argparse.ArgumentParser(description="Part 6: GRU hedging evaluation")
    p.add_argument("--epochs", type=int, default=35)
    p.add_argument("--train-paths", type=int, default=12000)
    p.add_argument("--val-paths", type=int, default=2000)
    p.add_argument("--test-paths", type=int, default=3000)
    p.add_argument("--pricing-paths", type=int, default=30000)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument(
    "--out-dir",
    default=str(Path(__file__).resolve().parent / "evaluation_results")
)
    p.add_argument("--skip-heston", action="store_true", help="Skip slower Heston oracle benchmark")
    p.add_argument("--skip-stress", action="store_true")
    a = p.parse_args()
    return Config(epochs=a.epochs, train_paths=a.train_paths,
                  val_paths=a.val_paths, test_paths=a.test_paths,
                  pricing_paths=a.pricing_paths, batch_size=a.batch_size,
                  seed=a.seed, device=a.device, out_dir=a.out_dir,
                  include_heston=not a.skip_heston,
                  run_stress=not a.skip_stress)


if __name__ == "__main__":
    main(parse_args())
