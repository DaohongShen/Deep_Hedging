"""Optional Extension: causal autoencoder latent volatility state for Heston hedging.

Place alongside evaluation.py and all existing Part 1-6 modules.
Run: python autoencoder.py
Fast: python autoencoder.py --train-paths 500 --val-paths 150 --test-paths 200 --pricing-paths 1000 --ae-epochs 3 --epochs 3 --skip-heston

Trains an autoencoder on PAST returns and a past closing-price range proxy;
freezes its encoder; appends one latent factor to the existing 6 state inputs;
trains a fresh GRU hedger and compares it with a baseline on identical held-out
paths. No true instantaneous Heston variance is given to either model.
"""
from __future__ import annotations

import argparse
import copy
import json
from dataclasses import asdict, replace
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from evaluation import (
    Config, K, T, n_steps, seed_all, simulate_set, mc_option_premium,
    make_features, account, metrics, evaluate_positions, train_gru, predict_gru,
    baseline_positions,
)
from state_inputs import StateStandardScaler
from deep_learning_policy import GRUPolicy


# -----------------------------------------------------------------------------
# 1. Causal historical windows: 20 returns, 20 absolute returns, 1 close range.
# -----------------------------------------------------------------------------
def historical_windows(stock: np.ndarray, lookback: int = 20) -> np.ndarray:
    """Return (n_paths, hedge_dates, 2*lookback+2), no look-ahead.

    The range is max(log close)-min(log close) over OBSERVED closes only:
    it is not an intraday high-low range. Initial missing returns are zero,
    with a 'history fraction' feature to distinguish short early histories.
    Returns are rescaled by 1/sqrt(dt) to avoid tiny network gradients.
    """
    s = np.asarray(stock, dtype=np.float64)
    if s.ndim != 2 or s.shape[1] < 2 or not np.all(np.isfinite(s)) or np.any(s <= 0):
        raise ValueError("stock must be positive finite paths (n_paths, >=2 dates)")
    if lookback < 2:
        raise ValueError("lookback must be >= 2")
    n, n_obs = s.shape
    dt = T / (n_obs - 1)
    logs = np.log(s)
    ret = np.diff(logs, axis=1) / np.sqrt(dt)   # t=1,...,maturity
    n_hedges = n_obs - 1
    out = np.zeros((n, n_hedges, 2 * lookback + 2), dtype=np.float32)
    for t in range(n_hedges):
        k = min(t, lookback)  # at hedge t, returns 1..t are known
        if k:
            past = ret[:, t-k:t]  # DOES NOT contain future return t->t+1
            out[:, t, lookback-k:lookback] = past
            out[:, t, 2*lookback-k:2*lookback] = np.abs(past)
        close_start = max(0, t - lookback)
        observed_closes = logs[:, close_start:t+1]
        out[:, t, -2] = (
            observed_closes.max(axis=1) - observed_closes.min(axis=1)
        ) / np.sqrt(max(lookback, 1) * dt)
        out[:, t, -1] = k / lookback
    return out


class WindowScaler:
    def __init__(self):
        self.mean = None
        self.scale = None

    def fit(self, x: np.ndarray):
        self.mean = x.mean(axis=(0, 1), dtype=np.float64).astype(np.float32)
        std = x.std(axis=(0, 1), dtype=np.float64)
        self.scale = np.where(std > 1e-7, std, 1.0).astype(np.float32)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.mean is None:
            raise RuntimeError("Fit on training windows first")
        return ((x - self.mean) / self.scale).astype(np.float32)


# -----------------------------------------------------------------------------
# 2. Autoencoder with 1-dimensional latent state.
# -----------------------------------------------------------------------------
class Autoencoder(nn.Module):
    def __init__(self, input_size: int, latent_dim: int = 1, hidden: int = 32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden), nn.ReLU(),
            nn.Linear(hidden, 16), nn.ReLU(),
            nn.Linear(16, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 16), nn.ReLU(),
            nn.Linear(16, hidden), nn.ReLU(),
            nn.Linear(hidden, input_size),
        )

    def forward(self, x: torch.Tensor):
        z = self.encoder(x)
        return self.decoder(z), z


def fit_autoencoder(model: Autoencoder, train: np.ndarray, val: np.ndarray,
                    *, epochs: int, batch_size: int, lr: float,
                    patience: int, device: str) -> pd.DataFrame:
    # Flatten windows: AE learns individual past-only market snapshots.
    x_train = torch.from_numpy(train.reshape(-1, train.shape[-1]))
    x_val = torch.from_numpy(val.reshape(-1, val.shape[-1]))
    loader = DataLoader(TensorDataset(x_train), batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best = copy.deepcopy(model.state_dict())
    best_val, stale, history = float("inf"), 0, []
    for epoch in range(1, epochs + 1):
        model.train()
        total, count = 0.0, 0
        for (batch,) in loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            reconstructed, _ = model(batch)
            loss = nn.functional.mse_loss(reconstructed, batch)
            loss.backward()
            optimizer.step()
            total += float(loss.item()) * len(batch)
            count += len(batch)
        model.eval()
        with torch.no_grad():
            total_val = 0.0
            for start in range(0, len(x_val), batch_size):
                batch = x_val[start:start+batch_size].to(device)
                reconstructed, _ = model(batch)
                total_val += nn.functional.mse_loss(
                    reconstructed, batch, reduction="sum"
                ).item()
            val_mse = total_val / x_val.numel()
        train_mse = total / count
        history.append({"epoch": epoch, "train_mse": train_mse, "val_mse": val_mse})
        print(f"AE epoch {epoch:03d} | train={train_mse:.6f} | val={val_mse:.6f}")
        if val_mse < best_val - 1e-6:
            best_val, best, stale = val_mse, copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
            if stale >= patience:
                print("AE early stopping")
                break
    model.load_state_dict(best)
    model.eval()
    return pd.DataFrame(history)


@torch.no_grad()
def encode_paths(ae: Autoencoder, x: np.ndarray, *, device: str,
                 batch_size: int = 8192) -> np.ndarray:
    flat = x.reshape(-1, x.shape[-1])
    chunks = []
    ae.eval()
    for k in range(0, len(flat), batch_size):
        batch = torch.from_numpy(flat[k:k+batch_size]).to(device)
        chunks.append(ae.encoder(batch).cpu().numpy())
    return np.concatenate(chunks).reshape(*x.shape[:2], -1).astype(np.float32)


# -----------------------------------------------------------------------------
# 3. Frozen encoder -> append latent z to original 6 features -> new GRU.
# -----------------------------------------------------------------------------
def augmented_features(stock: np.ndarray, state_scaler: StateStandardScaler,
                       window_scaler: WindowScaler, ae: Autoencoder,
                       z_mean: np.ndarray, z_scale: np.ndarray,
                       *, lookback: int, device: str) -> np.ndarray:
    original = make_features(stock, state_scaler)
    past = window_scaler.transform(historical_windows(stock, lookback))
    z = encode_paths(ae, past, device=device)
    z = (z - z_mean) / z_scale
    return np.concatenate([original, z], axis=-1).astype(np.float32)


def train_augmented_gru(cfg: Config, premium: float, x_train: np.ndarray,
                        s_train: np.ndarray, x_val: np.ndarray,
                        s_val: np.ndarray):
    loader = DataLoader(TensorDataset(torch.from_numpy(x_train),
                                      torch.from_numpy(s_train)),
                        batch_size=cfg.batch_size, shuffle=True)
    model = GRUPolicy(input_size=x_train.shape[-1], hidden_size=cfg.hidden_size,
                      num_layers=cfg.num_layers).to(cfg.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate,
                                 weight_decay=cfg.weight_decay)
    x_val_tensor = torch.from_numpy(x_val).to(cfg.device)
    s_val_tensor = torch.from_numpy(s_val).to(cfg.device)
    best_weights = copy.deepcopy(model.state_dict())
    best_loss, stale, history = float("inf"), 0, []
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        total = 0.0
        for x, s in loader:
            x, s = x.to(cfg.device), s.to(cfg.device)
            optimizer.zero_grad(set_to_none=True)
            error = account(s, model(x), premium, cfg).hedging_error
            loss = error.square().mean()
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite hedging loss")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            total += loss.item() * len(x)
        model.eval()
        with torch.no_grad():
            val_sse = 0.0
            for start in range(0, len(x_val), cfg.batch_size):
                end = start + cfg.batch_size
                err = account(s_val_tensor[start:end],
                              model(x_val_tensor[start:end]), premium, cfg).hedging_error
                val_sse += err.square().sum().item()
        val_mse = val_sse / len(x_val)
        history.append({"epoch": epoch, "train_mse": total / len(x_train),
                        "val_mse": val_mse})
        print(f"GRU+AE epoch {epoch:03d} | train={total/len(x_train):.6f} | val={val_mse:.6f}")
        if val_mse < best_loss - 1e-6:
            best_loss, stale = val_mse, 0
            best_weights = copy.deepcopy(model.state_dict())
        else:
            stale += 1
            if stale >= cfg.patience:
                print("GRU+AE early stopping")
                break
    model.load_state_dict(best_weights)
    return model, pd.DataFrame(history)


@torch.no_grad()
def predict_augmented(model: GRUPolicy, x: np.ndarray, cfg: Config) -> np.ndarray:
    model.eval()
    return np.concatenate([
        model(torch.from_numpy(x[i:i+cfg.batch_size]).to(cfg.device)).cpu().numpy()
        for i in range(0, len(x), cfg.batch_size)
    ], axis=0)


def load_previous_gru(results_dir: Path, cfg: Config, state_scaler: StateStandardScaler):
    """Load only if checkpoint and saved configuration match the current experiment.

    Different train sizes/seed/training epochs make comparisons less controlled;
    in that case the caller must train an original-feature GRU anew.
    """
    path = results_dir / "gru_policy.pt"
    scale_path = results_dir / "feature_scaler.npz"
    config_path = results_dir / "config.json"
    if not all(p.is_file() for p in [path, scale_path, config_path]):
        return None
    old = json.loads(config_path.read_text())
    keys = ("seed", "train_paths", "val_paths", "epochs", "batch_size",
            "learning_rate", "weight_decay", "hidden_size", "num_layers",
            "full_spread", "commission_rate", "patience")
    if any(old.get(key) != getattr(cfg, key) for key in keys):
        return None
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if int(checkpoint["input_size"]) != 6:
        return None
    with np.load(scale_path, allow_pickle=False) as f:
        old_mean, old_scale = f["mean"], f["scale"]
    if not (np.allclose(old_mean, state_scaler.mean_) and
            np.allclose(old_scale, state_scaler.scale_)):
        return None
    model = GRUPolicy(input_size=6, hidden_size=int(checkpoint["hidden_size"]),
                      num_layers=int(checkpoint["num_layers"])).to(cfg.device)
    model.load_state_dict(checkpoint["state_dict"])
    return model


# -----------------------------------------------------------------------------
# 4. Independent OOS evaluation and latent-volatility diagnostics.
# -----------------------------------------------------------------------------
def latent_diagnostics(z: np.ndarray, stock: np.ndarray, variance: np.ndarray,
                       lookback: int) -> dict:
    """True simulated variance is used for offline diagnostics ONLY."""
    returns = np.diff(np.log(stock.astype(np.float64)), axis=1)
    rv = np.zeros_like(z[..., 0], dtype=np.float64)
    for t in range(1, rv.shape[1]):
        past = returns[:, max(0, t-lookback):t]
        if past.shape[1] >= 2:
            rv[:, t] = past.std(axis=1) / np.sqrt(T / (stock.shape[1] - 1))
    def safe_corr(a, b):
        a, b = a.ravel(), b.ravel()
        if a.std() < 1e-12 or b.std() < 1e-12:
            return float("nan")
        return float(np.corrcoef(a, b)[0, 1])
    return {
        "z_vs_realised_vol_correlation": safe_corr(z[..., 0], rv),
        "z_vs_heston_sqrt_v_correlation_diagnostic_only": safe_corr(
            z[..., 0], np.sqrt(np.maximum(variance[:, :-1], 0))
        ),
    }


def main(args):
    cfg = Config(train_paths=args.train_paths, val_paths=args.val_paths,
                 test_paths=args.test_paths, pricing_paths=args.pricing_paths,
                 epochs=args.epochs, batch_size=args.batch_size, seed=args.seed,
                 device=args.device, include_heston=not args.skip_heston,
                 run_stress=False)
    seed_all(cfg.seed)
    if min(cfg.train_paths, cfg.val_paths, cfg.test_paths, cfg.pricing_paths) < 2:
        raise ValueError("All path counts must be at least two")
    out = Path(__file__).resolve().parent / "autoencoder_results"
    out.mkdir(exist_ok=True, parents=True)
    (out / "config.json").write_text(json.dumps({**asdict(cfg),
        "lookback": args.lookback, "ae_epochs": args.ae_epochs}, indent=2))

    pricing_stock, _ = simulate_set(cfg.pricing_paths, cfg.seed+1)
    premium = mc_option_premium(pricing_stock)
    del pricing_stock
    train_stock, _ = simulate_set(cfg.train_paths, cfg.seed+2)
    val_stock, _ = simulate_set(cfg.val_paths, cfg.seed+3)
    test_stock, test_variance = simulate_set(cfg.test_paths, cfg.seed+4)
    print(f"Independent option premium: {premium:.6f}")

    # Original six state features, standardised using training paths only.
    state_scaler = StateStandardScaler()
    x_train_base = make_features(train_stock, state_scaler, fit=True)
    x_val_base = make_features(val_stock, state_scaler)
    np.savez(out / "state_scaler.npz", mean=state_scaler.mean_, scale=state_scaler.scale_)

    # AE uses no option premium, variance path, payoff or hedging labels.
    window_scaler = WindowScaler()
    w_train = window_scaler.fit(historical_windows(train_stock, args.lookback)).transform(
        historical_windows(train_stock, args.lookback))
    w_val = window_scaler.transform(historical_windows(val_stock, args.lookback))
    np.savez(out / "window_scaler.npz", mean=window_scaler.mean,
             scale=window_scaler.scale)
    ae = Autoencoder(w_train.shape[-1], latent_dim=1).to(cfg.device)
    ae_history = fit_autoencoder(ae, w_train, w_val, epochs=args.ae_epochs,
                                 batch_size=cfg.batch_size, lr=args.ae_lr,
                                 patience=args.ae_patience, device=cfg.device)
    ae_history.to_csv(out / "autoencoder_training.csv", index=False)
    torch.save({"state_dict": ae.state_dict(), "input_size": w_train.shape[-1],
                "latent_dim": 1, "lookback": args.lookback}, out / "autoencoder.pt")

    z_train = encode_paths(ae, w_train, device=cfg.device)
    z_mean = z_train.mean(axis=(0, 1), keepdims=True).astype(np.float32)
    z_std = z_train.std(axis=(0, 1), keepdims=True).astype(np.float32)
    z_std = np.maximum(z_std, 1e-6)
    np.savez(out / "latent_scaler.npz", mean=z_mean, scale=z_std)
    z_val = encode_paths(ae, w_val, device=cfg.device)
    x_train = np.concatenate([x_train_base, (z_train-z_mean)/z_std], axis=-1)
    x_val = np.concatenate([x_val_base, (z_val-z_mean)/z_std], axis=-1)

    # Reuse prior baseline only under matching experiment configuration.
    prior = Path(__file__).resolve().parent / "evaluation_results"
    baseline = load_previous_gru(prior, cfg, state_scaler)
    if baseline is None:
        print("Training original 6-feature GRU for fair comparison...")
        # Reset seeding for a reproducible baseline run.
        seed_all(cfg.seed)
        baseline, baseline_history = train_gru(
            cfg, premium, StateStandardScaler(), train_stock, val_stock)
        baseline_history.to_csv(out / "baseline_training.csv", index=False)
    else:
        print("Reusing matching pretrained GRU from evaluation_results")

    # Identical optimizer and objective, now on seven features.
    seed_all(cfg.seed)
    augmented, hedge_history = train_augmented_gru(
        cfg, premium, x_train, train_stock, x_val, val_stock)
    hedge_history.to_csv(out / "augmented_gru_training.csv", index=False)
    torch.save({"state_dict": augmented.state_dict(), "input_size": 7,
                "hidden_size": cfg.hidden_size, "num_layers": cfg.num_layers},
               out / "gru_autoencoder_policy.pt")

    x_test = augmented_features(test_stock, state_scaler, window_scaler, ae,
                                z_mean, z_std, lookback=args.lookback,
                                device=cfg.device)
    z_test = x_test[..., -1:]
    diagnostic = latent_diagnostics(z_test, test_stock, test_variance, args.lookback)
    (out / "latent_diagnostics.json").write_text(json.dumps(diagnostic, indent=2))

    policies = {
        "Original GRU": predict_gru(baseline, test_stock, state_scaler, cfg),
        "GRU + Autoencoder": predict_augmented(augmented, x_test, cfg),
    }
    if args.include_classical:
        policies.update(baseline_positions(test_stock, test_variance, cfg))
    details = {name: evaluate_positions(test_stock, p, premium, cfg)
               for name, p in policies.items()}
    comparison = pd.DataFrame([
        {"strategy": name, **metrics(result, cfg.cvar_alpha)}
        for name, result in details.items()
    ])
    comparison.to_csv(out / "comparison.csv", index=False)
    np.savez_compressed(out / "test_hedging_errors.npz",
                        **{name.replace(" ", "_").replace("+", "plus"): x["error"]
                           for name, x in details.items()})

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(ae_history.epoch, ae_history.train_mse, label="Train")
    ax.plot(ae_history.epoch, ae_history.val_mse, label="Validation")
    ax.set(xlabel="Epoch", ylabel="Reconstruction MSE", title="Autoencoder training")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "autoencoder_learning_curve.png", dpi=170)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for name in ["Original GRU", "GRU + Autoencoder"]:
        ax.hist(details[name]["error"], bins=55, density=True,
                alpha=0.45, label=name)
    ax.set(xlabel="Terminal hedging error", ylabel="Density",
           title="Out-of-sample hedging-error comparison")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "hedging_error_comparison.png", dpi=170)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    subset = comparison.set_index("strategy").loc[["Original GRU", "GRU + Autoencoder"]]
    subset[["rmse", "CVaR_loss"]].plot.bar(ax=ax, rot=0)
    ax.set(ylabel="Amount (price currency)", title="GRU vs GRU + Autoencoder")
    fig.tight_layout()
    fig.savefig(out / "risk_metric_comparison.png", dpi=170)
    plt.close(fig)

    print("\nOut-of-sample comparison:\n", comparison.to_string(index=False))
    print("Latent state diagnostics (correlation, not causation):", diagnostic)
    print("Results saved to:", out.resolve())


def parse_args():
    parser = argparse.ArgumentParser(description="Optional extension: Autoencoder latent state")
    parser.add_argument("--train-paths", type=int, default=12000)
    parser.add_argument("--val-paths", type=int, default=2000)
    parser.add_argument("--test-paths", type=int, default=3000)
    parser.add_argument("--pricing-paths", type=int, default=30000)
    parser.add_argument("--epochs", type=int, default=35, help="GRU epochs")
    parser.add_argument("--ae-epochs", type=int, default=25)
    parser.add_argument("--ae-patience", type=int, default=6)
    parser.add_argument("--ae-lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lookback", type=int, default=20)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--skip-heston", action="store_true")
    parser.add_argument("--include-classical", action="store_true",
                        help="Include no-hedge, BS delta, and optional Heston delta")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
