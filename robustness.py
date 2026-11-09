"""Optional Extension — Heston parameter-shift robustness for a frozen GRU.

Place this file beside evaluation.py and the other Part 1–6 modules.
Run: python robustness.py
Fast: python robustness.py --test-paths 200 --skip-heston

Uses the existing trained GRU and scaler from evaluation_results. Does not retrain.
All scenarios retain the original option premium, maturity, hedge frequency and
transaction costs, isolating the consequences of Heston parameter shifts.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from evaluation import (
    Config, GRUPolicy, StateStandardScaler, S0, v0, r, kappa, theta, xi,
    rho, T, K, n_steps, simulate_heston, mc_option_premium, predict_gru,
    evaluate_positions, metrics, no_hedging_policy, bs_delta_hedging_policy,
    HestonParameters, heston_delta_hedging_policy,
)


@dataclass(frozen=True)
class Scenario:
    label: str
    kappa: float
    xi: float
    rho: float


SCENARIOS = (
    Scenario("Original", kappa, xi, rho),
    Scenario("Lower mean reversion", 1.0, xi, rho),
    Scenario("Higher mean reversion", 4.0, xi, rho),
    Scenario("Lower vol-of-vol", kappa, 0.15, rho),
    Scenario("Higher vol-of-vol", kappa, 0.50, rho),
    Scenario("Weaker correlation", kappa, xi, -0.30),
    Scenario("Stronger correlation", kappa, xi, -0.90),
    Scenario("Combined stress", 1.0, 0.50, -0.30),
)


def simulate_scenario(n_paths: int, seed: int, scenario: Scenario):
    """Use actual shifted kappa, xi and rho (unlike evaluation.simulate_set)."""
    _, stock, variance, _ = simulate_heston(
        S0=S0, v0=v0, r=r, kappa=scenario.kappa, theta=theta,
        xi=scenario.xi, rho=scenario.rho, T=T, n_steps=n_steps,
        n_paths=n_paths, seed=seed,
    )
    stock = np.asarray(stock, dtype=np.float32)
    variance = np.asarray(variance, dtype=np.float32)
    expected = (n_paths, n_steps + 1)
    if stock.shape != expected or variance.shape != expected:
        raise ValueError(f"Expected {expected}, got S={stock.shape}, v={variance.shape}")
    return stock, variance


def load_frozen_model(results_dir: Path, cfg: Config):
    checkpoint_path = results_dir / "gru_policy.pt"
    scaler_path = results_dir / "feature_scaler.npz"
    if not checkpoint_path.is_file() or not scaler_path.is_file():
        raise FileNotFoundError(
            f"Expected {checkpoint_path.name} and {scaler_path.name} in {results_dir}. "
            "Run evaluation.py first to train and save the GRU."
        )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    with np.load(scaler_path, allow_pickle=False) as saved:
        mean = saved["mean"].copy()
        scale = saved["scale"].copy()
    input_size = int(checkpoint["input_size"])
    if mean.shape[-1] != input_size or scale.shape[-1] != input_size:
        raise ValueError("Saved scaler feature count differs from GRU input size")
    scaler = StateStandardScaler()
    scaler.mean_ = mean
    scaler.scale_ = scale
    model = GRUPolicy(input_size=input_size,
                      hidden_size=int(checkpoint["hidden_size"]),
                      num_layers=int(checkpoint["num_layers"])).to(cfg.device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, scaler


def original_premium(results_dir: Path, cfg: Config) -> float:
    """Reproduce evaluation.py's independent risk-neutral pricing simulation."""
    config_path = results_dir / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(
            "config.json is required to reproduce the original premium's seed "
            "and Monte Carlo sample size; run evaluation.py with this configuration."
        )
    original_cfg = json.loads(config_path.read_text(encoding="utf-8"))
    pricing_paths = int(original_cfg["pricing_paths"])
    pricing_seed = int(original_cfg["seed"]) + 1
    original = Scenario("Original", kappa, xi, rho)
    prices, _ = simulate_scenario(pricing_paths, pricing_seed, original)
    return mc_option_premium(prices)


def evaluate_scenario(model, scaler, scenario: Scenario, premium: float,
                      cfg: Config, common_seed: int) -> list[dict]:
    stock, variance = simulate_scenario(cfg.test_paths, common_seed, scenario)
    policies = {
        "No Hedging": no_hedging_policy(stock),
        "BS Delta": bs_delta_hedging_policy(
            stock, strike=K, maturity=T, risk_free_rate=r,
            volatility=float(np.sqrt(theta)),
        ),
    }
    if cfg.include_heston:
        # Oracle knows both the true instantaneous variance and shifted parameters.
        params = HestonParameters(r=r, kappa=scenario.kappa, theta=theta,
                                  xi=scenario.xi, rho=scenario.rho)
        policies["Heston Delta (Oracle)"] = heston_delta_hedging_policy(
            stock, variance, strike=K, maturity=T, params=params,
            n_quadrature=cfg.heston_quadrature,
            integration_upper=cfg.heston_integration_upper,
        )
    policies["GRU Deep Hedging"] = predict_gru(model, stock, scaler, cfg)
    rows = []
    for name, positions in policies.items():
        result = evaluate_positions(stock, np.asarray(positions), premium, cfg)
        rows.append({
            "scenario": scenario.label, "strategy": name,
            "kappa": scenario.kappa, "xi": scenario.xi, "rho": scenario.rho,
            **metrics(result, cfg.cvar_alpha),
        })
    return rows


def add_relative_changes(results: pd.DataFrame) -> pd.DataFrame:
    base = results.loc[results["scenario"] == "Original"].set_index("strategy")
    for metric in ("rmse", "CVaR_loss"):
        denominator = results["strategy"].map(base[metric])
        results[f"{metric}_change_pct"] = np.where(
            denominator.abs() > 1e-12,
            100.0 * (results[metric] - denominator) / denominator.abs(),
            np.nan,
        )
    bs = results.loc[results.strategy == "BS Delta"].set_index("scenario")
    bs_rmse = results["scenario"].map(bs["rmse"])
    results["rmse_vs_bs_pct"] = np.where(
        bs_rmse.abs() > 1e-12,
        100.0 * (results["rmse"] - bs_rmse) / bs_rmse, np.nan,
    )
    return results


def plot_results(results: pd.DataFrame, output_dir: Path) -> None:
    for column, ylabel, filename in (
        ("rmse", "Terminal hedging error RMSE", "robustness_rmse.png"),
        ("CVaR_loss", "CVaR of terminal hedging loss", "robustness_cvar.png"),
    ):
        fig, ax = plt.subplots(figsize=(11, 5))
        for strategy, group in results.groupby("strategy", sort=False):
            ax.plot(group["scenario"], group[column], marker="o", label=strategy)
        ax.set(title=f"Heston parameter-shift robustness: {ylabel}",
               xlabel="Scenario", ylabel=ylabel)
        ax.tick_params(axis="x", rotation=30)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(output_dir / filename, dpi=170)
        plt.close(fig)

    subset = results.loc[results.strategy == "GRU Deep Hedging"]
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar(subset["scenario"], subset["rmse_change_pct"])
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set(title="Frozen GRU: RMSE change from original regime",
           xlabel="Scenario", ylabel="RMSE change (%)")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(output_dir / "gru_rmse_change.png", dpi=170)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Optional: GRU robustness under shifted Heston parameters")
    base_dir = Path(__file__).resolve().parent
    parser.add_argument("--results-dir", type=Path, default=base_dir / "evaluation_results")
    parser.add_argument("--out-dir", type=Path, default=base_dir / "robustness_results")
    parser.add_argument("--test-paths", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=3026)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--skip-heston", action="store_true")
    args = parser.parse_args()
    if args.test_paths < 2 or args.batch_size < 1:
        parser.error("--test-paths must be >= 2 and --batch-size must be >= 1")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        parser.error("CUDA requested but not available")

    config_path = args.results_dir / "config.json"
    if not config_path.is_file():
        parser.error(f"Missing {config_path}; run evaluation.py first")
    saved_config = json.loads(config_path.read_text(encoding="utf-8"))
    cfg = Config(
        test_paths=args.test_paths, batch_size=args.batch_size, seed=args.seed,
        device=args.device, include_heston=not args.skip_heston,
        full_spread=float(saved_config["full_spread"]),
        commission_rate=float(saved_config["commission_rate"]),
        cvar_alpha=float(saved_config["cvar_alpha"]),
        heston_quadrature=int(saved_config["heston_quadrature"]),
        heston_integration_upper=float(saved_config["heston_integration_upper"]),
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    model, scaler = load_frozen_model(args.results_dir, cfg)
    premium = original_premium(args.results_dir, cfg)
    print(f"Fixed original-regime option premium: {premium:.6f}")
    print("Using frozen GRU and original feature scaler; no retraining or refitting.")

    rows = []
    for scenario in SCENARIOS:
        print(f"Evaluating {scenario.label} (kappa={scenario.kappa}, "
              f"xi={scenario.xi}, rho={scenario.rho})")
        # Same RNG seed across scenarios: common random numbers for comparison.
        rows.extend(evaluate_scenario(model, scaler, scenario, premium, cfg, args.seed))
    results = add_relative_changes(pd.DataFrame(rows))
    results.to_csv(args.out_dir / "robustness_comparison.csv", index=False)
    results.loc[results.strategy == "GRU Deep Hedging"].to_csv(
        args.out_dir / "gru_robustness_summary.csv", index=False,
    )
    plot_results(results, args.out_dir)
    metadata = {
        "original_premium": premium, "test_paths_per_scenario": args.test_paths,
        "common_random_seed": args.seed,
        "fixed_premium_across_scenarios": True,
        "frozen_model": True,
        "oracle_uses_shifted_parameters": cfg.include_heston,
        "scenarios": [vars(s) for s in SCENARIOS],
    }
    (args.out_dir / "robustness_config.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8",
    )
    columns = ["scenario", "strategy", "rmse", "rmse_change_pct", "CVaR_loss", "rmse_vs_bs_pct"]
    print("\nRobustness comparison:\n", results[columns].to_string(index=False))
    print(f"\nSaved robustness results to: {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
