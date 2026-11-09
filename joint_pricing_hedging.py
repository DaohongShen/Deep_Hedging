"""Optional Extension 2: Joint pricing + hedging under Heston dynamics.

Put this file next to the existing six modules: heston_model_setup.py,
hedging_objective.py, deep_learning_policy.py, state_inputs.py, baselines.py,
evaluation.py. Run: python joint_pricing_hedging.py
Quick smoke test:
  python joint_pricing_hedging.py --epochs 2 --train-paths 300 --val-paths 100 \
    --test-paths 100 --pricing-paths 1000 --skip-heston --skip-baseline-train

This trains a SHARED GRU with two separate heads for call value C_t and hedge
holdings phi_t. Inputs are causal, observable stock-derived features only.

Pricing: discounted realised terminal payoff supplies unbiased noisy targets for
conditional risk-neutral values. No latent Heston variance is leaked into inputs.
Hedging: the original cost-aware, self-financing terminal wealth recursion.
Consistency: a *soft* discrete discounted-price/stock-gain residual, since
under stochastic volatility there is unhedgeable volatility risk and, with
frictions, exact replication and a delta=partial-derivative identity do not hold.
This is a learning objective, not an arbitrage-free pricing theorem.
"""
from __future__ import annotations

import argparse
import copy
import json
from dataclasses import asdict, replace
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from evaluation import (
    Config, K, T, r, seed_all, simulate_set, mc_option_premium,
    account, evaluate_positions, metrics, baseline_positions,
    train_gru, predict_gru,
)
from state_inputs import build_state_inputs, StateStandardScaler


class JointGRU(nn.Module):
    """Causal GRU: values at all 64 dates, positions at first 63 dates."""

    def __init__(self, input_size: int = 6, hidden_size: int = 64,
                 num_layers: int = 1):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers=num_layers,
                          batch_first=True)
        self.price_head = nn.Linear(hidden_size, 1)
        self.hedge_head = nn.Linear(hidden_size, 1)

    def forward(self, states: torch.Tensor):
        hidden, _ = self.gru(states)
        prices = F.softplus(self.price_head(hidden).squeeze(-1))
        positions = self.hedge_head(hidden[:, :-1]).squeeze(-1)
        return prices, positions


def features_full(stock: np.ndarray, scaler: StateStandardScaler,
                  *, fit: bool = False) -> np.ndarray:
    raw = build_state_inputs(stock, strike=K, maturity=T, include_terminal=True)
    if fit:
        # The scaler sees ONLY training-set information on decision dates.
        scaler.fit(raw[:, :-1])
    return np.asarray(scaler.transform(raw), dtype=np.float32)


def joint_loss(values: torch.Tensor, positions: torch.Tensor,
               stock: torch.Tensor, premium: float, cfg: Config,
               *, price_weight: float, consistency_weight: float,
               terminal_weight: float, anchor_weight: float):
    """Cost-aware terminal hedge loss + risk-neutral price learning.

    MSE of discounted payoff labels at date t is a valid (noisy) regression
    objective whose population minimiser is E_Q[exp(-r(T-t)) payoff|F_t].
    The residual consistency penalty is approximate, not an exact identity:
    stochastic-volatility risk is not spanned by stock and cash alone.
    """
    if values.shape != stock.shape or positions.shape != stock[:, :-1].shape:
        raise ValueError('Unexpected value/position/stock shapes')
    n_intervals = stock.shape[1] - 1
    dt = T / n_intervals
    payoff = (stock[:, -1] - K).clamp_min(0.)

    result = account(stock, positions, premium, cfg)
    hedge = result.hedging_error.square().mean()

    remaining = T - dt * torch.arange(n_intervals, device=stock.device,
                                       dtype=stock.dtype)
    discounted_payoffs = payoff[:, None] * torch.exp(-r * remaining)[None, :]
    # Scale the price regression across the 63 decision dates.
    price = (values[:, :-1] - discounted_payoffs).square().mean()
    terminal = (values[:, -1] - payoff).square().mean()
    anchor = (values[:, 0] - premium).square().mean()

    # One-period ex-dividend stock gain funded by the money market account.
    growth = torch.exp(stock.new_tensor(r * dt))
    stock_gain = stock[:, 1:] - growth * stock[:, :-1]
    residual = values[:, 1:] - growth * values[:, :-1] - positions * stock_gain
    consistency = residual.square().mean()

    total = (hedge + price_weight * price
             + consistency_weight * consistency
             + terminal_weight * terminal + anchor_weight * anchor)
    return total, {
        'hedge_mse': hedge, 'price_mse': price, 'consistency_mse': consistency,
        'terminal_mse': terminal, 'anchor_mse': anchor,
    }


def train_joint(cfg: Config, premium: float, scaler: StateStandardScaler,
                train_stock: np.ndarray, val_stock: np.ndarray,
                weights: dict):
    xt = features_full(train_stock, scaler, fit=True)
    xv = features_full(val_stock, scaler)
    data = TensorDataset(torch.from_numpy(xt), torch.from_numpy(train_stock))
    loader = DataLoader(data, batch_size=cfg.batch_size, shuffle=True)
    model = JointGRU(xt.shape[-1], cfg.hidden_size, cfg.num_layers).to(cfg.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate,
                                 weight_decay=cfg.weight_decay)
    best = copy.deepcopy(model.state_dict())
    best_val, stale = float('inf'), 0
    history = []
    val_x = torch.from_numpy(xv)
    val_s = torch.from_numpy(val_stock)

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        train_sum = 0.
        for x, s in loader:
            x, s = x.to(cfg.device), s.to(cfg.device)
            optimizer.zero_grad(set_to_none=True)
            values, positions = model(x)
            loss, _ = joint_loss(values, positions, s, premium, cfg, **weights)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError('Nonfinite joint loss')
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            train_sum += loss.item() * x.shape[0]

        model.eval()
        val_sums = {k: 0. for k in ['total', 'hedge_mse', 'price_mse',
                                        'consistency_mse', 'terminal_mse', 'anchor_mse']}
        with torch.no_grad():
            for i in range(0, len(val_stock), cfg.batch_size):
                x = val_x[i:i+cfg.batch_size].to(cfg.device)
                s = val_s[i:i+cfg.batch_size].to(cfg.device)
                values, positions = model(x)
                loss, parts = joint_loss(values, positions, s, premium, cfg, **weights)
                val_sums['total'] += loss.item() * len(x)
                for key in parts:
                    val_sums[key] += parts[key].item() * len(x)
        means = {k: v / len(val_stock) for k, v in val_sums.items()}
        history.append({'epoch': epoch, 'train_total': train_sum / len(train_stock),
                        **{'val_' + k: v for k, v in means.items()}})
        print(f"Epoch {epoch:03d} | train={history[-1]['train_total']:.5f} "
              f"| val={means['total']:.5f} "
              f"| hedge={means['hedge_mse']:.5f} "
              f"| price={means['price_mse']:.5f}")
        if means['total'] < best_val - 1e-6:
            best_val = means['total']
            best = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= cfg.patience:
                print('Early stopping on validation joint loss.')
                break
    model.load_state_dict(best)
    return model, pd.DataFrame(history)


@torch.no_grad()
def predict_joint(model: JointGRU, stock: np.ndarray,
                  scaler: StateStandardScaler, cfg: Config):
    x = features_full(stock, scaler)
    model.eval()
    prices, positions = [], []
    for i in range(0, len(x), cfg.batch_size):
        batch = torch.from_numpy(x[i:i+cfg.batch_size]).to(cfg.device)
        p, h = model(batch)
        prices.append(p.cpu().numpy())
        positions.append(h.cpu().numpy())
    return np.concatenate(prices), np.concatenate(positions)


def evaluate_pricing(values: np.ndarray, positions: np.ndarray,
                     stock: np.ndarray, premium: float):
    payoff = np.maximum(stock[:, -1] - K, 0.)
    n = stock.shape[1] - 1
    dt = T / n
    discount = np.exp(-r * (T - np.arange(n) * dt))
    targets = payoff[:, None] * discount[None, :]
    growth = np.exp(r * dt)
    residual = (values[:, 1:] - growth * values[:, :-1]
                - positions * (stock[:, 1:] - growth * stock[:, :-1]))
    return {
        'initial_predicted_price_mean': float(values[:, 0].mean()),
        'independent_mc_price': float(premium),
        'initial_price_bias': float(values[:, 0].mean() - premium),
        'price_payoff_regression_rmse': float(np.sqrt(np.mean((values[:, :-1] - targets) ** 2))),
        'terminal_payoff_rmse': float(np.sqrt(np.mean((values[:, -1] - payoff) ** 2))),
        'consistency_residual_rmse': float(np.sqrt(np.mean(residual ** 2))),
        'consistency_residual_mean': float(residual.mean()),
        'initial_price_prediction_std': float(values[:, 0].std()),
    }


def save_figures(history: pd.DataFrame, comparison: pd.DataFrame,
                 details: dict, values: np.ndarray, stock: np.ndarray,
                 out_dir: Path):
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(history.epoch, history.train_total, label='Train total')
    ax.plot(history.epoch, history.val_total, label='Validation total')
    ax.set(xlabel='Epoch', ylabel='Weighted joint loss', title='Joint model training')
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / 'joint_training_curve.png', dpi=170)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.7))
    for label, result in details.items():
        ax.hist(result['error'], bins=45, density=True, alpha=.36, label=label)
    ax.set(xlabel='Terminal hedging error', ylabel='Density',
           title='Held-out hedging-error distribution')
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / 'joint_hedging_errors.png', dpi=170)
    plt.close(fig)

    # A few sample paths, price value vs intrinsic payoff at maturity.
    fig, ax = plt.subplots(figsize=(8, 4.7))
    t = np.linspace(0, T, values.shape[1])
    for k in range(min(5, len(values))):
        ax.plot(t, values[k], alpha=.75, label=f'Path {k+1}')
    ax.set(xlabel='Time (years)', ylabel='Network call value',
           title='Predicted option values along held-out paths')
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / 'joint_predicted_prices.png', dpi=170)
    plt.close(fig)


def main(args):
    cfg = Config(train_paths=args.train_paths, val_paths=args.val_paths,
                 test_paths=args.test_paths, pricing_paths=args.pricing_paths,
                 epochs=args.epochs, batch_size=args.batch_size,
                 seed=args.seed, device=args.device,
                 include_heston=not args.skip_heston, run_stress=False,
                 out_dir=str(Path(__file__).resolve().parent / 'joint_pricing_hedging_results'))
    if min(cfg.train_paths, cfg.val_paths, cfg.test_paths, cfg.pricing_paths) < 2:
        raise ValueError('All path counts must be >= 2')
    if cfg.epochs < 1 or cfg.batch_size < 1:
        raise ValueError('epochs and batch-size must be positive')
    for name in ('price_weight', 'consistency_weight', 'terminal_weight', 'anchor_weight'):
        if getattr(args, name) < 0:
            raise ValueError(f'{name} must be nonnegative')
    seed_all(cfg.seed)
    output = Path(args.out_dir) if args.out_dir else Path(cfg.out_dir)
    output.mkdir(parents=True, exist_ok=True)
    weights = {name: getattr(args, name) for name in (
        'price_weight', 'consistency_weight', 'terminal_weight', 'anchor_weight')}
    (output / 'config.json').write_text(json.dumps({'model_config': asdict(cfg),
                                                      'loss_weights': weights,
                                                      'baseline_trained': not args.skip_baseline_train}, indent=2))

    # MC premium is estimated ONLY on independent pricing paths.
    pricing, _ = simulate_set(cfg.pricing_paths, cfg.seed + 1)
    premium = mc_option_premium(pricing)
    del pricing
    print(f'Independent risk-neutral MC premium: {premium:.6f}')
    train_stock, _ = simulate_set(cfg.train_paths, cfg.seed + 2)
    val_stock, _ = simulate_set(cfg.val_paths, cfg.seed + 3)
    scaler = StateStandardScaler()
    joint, hist = train_joint(cfg, premium, scaler, train_stock, val_stock, weights)
    hist.to_csv(output / 'joint_training_history.csv', index=False)
    torch.save({'state_dict': joint.state_dict(), 'input_size': 6,
                'hidden_size': cfg.hidden_size, 'num_layers': cfg.num_layers},
               output / 'joint_gru.pt')
    np.savez(output / 'joint_scaler.npz', mean=scaler.mean_, scale=scaler.scale_)

    # Fair comparison uses same independent train/val/test paths and same premium.
    baseline_gru = None
    baseline_scaler = None
    if not args.skip_baseline_train:
        print('\nTraining original hedge-only GRU for matched comparison...')
        baseline_scaler = StateStandardScaler()
        baseline_gru, base_hist = train_gru(cfg, premium, baseline_scaler,
                                            train_stock, val_stock)
        base_hist.to_csv(output / 'baseline_training_history.csv', index=False)
    del train_stock, val_stock

    test_stock, test_variance = simulate_set(cfg.test_paths, cfg.seed + 4)
    prices, positions = predict_joint(joint, test_stock, scaler, cfg)
    policies = baseline_positions(test_stock, test_variance, cfg)
    if baseline_gru is not None:
        policies['Original GRU Hedging'] = predict_gru(
            baseline_gru, test_stock, baseline_scaler, cfg)
    policies['Joint GRU Pricing + Hedging'] = positions
    details = {}
    rows = []
    for label, p in policies.items():
        result = evaluate_positions(test_stock, np.asarray(p), premium, cfg)
        details[label] = result
        rows.append({'strategy': label, **metrics(result, cfg.cvar_alpha)})
    comparison = pd.DataFrame(rows)
    comparison.to_csv(output / 'joint_hedging_comparison.csv', index=False)
    pricing_metrics = evaluate_pricing(prices, positions, test_stock, premium)
    (output / 'joint_pricing_metrics.json').write_text(json.dumps(pricing_metrics, indent=2))
    np.savez_compressed(output / 'joint_test_outputs.npz', stock=test_stock,
                        predicted_values=prices, predicted_positions=positions,
                        **{'error_' + str(i): v['error'] for i, v in enumerate(details.values())})
    (output / 'error_array_labels.json').write_text(json.dumps(list(details), indent=2))
    save_figures(hist, comparison, details, prices, test_stock, output)
    print('\nOut-of-sample hedging metrics:\n', comparison.to_string(index=False))
    print('\nOut-of-sample pricing diagnostics:\n', json.dumps(pricing_metrics, indent=2))
    print(f'\nResults saved to {output.resolve()}')


def parse_args():
    p = argparse.ArgumentParser(description='Joint Heston option pricing and deep hedging')
    p.add_argument('--epochs', type=int, default=35)
    p.add_argument('--train-paths', type=int, default=12000)
    p.add_argument('--val-paths', type=int, default=2000)
    p.add_argument('--test-paths', type=int, default=3000)
    p.add_argument('--pricing-paths', type=int, default=30000)
    p.add_argument('--batch-size', type=int, default=512)
    p.add_argument('--seed', type=int, default=2026)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--out-dir', default=None)
    p.add_argument('--skip-heston', action='store_true')
    p.add_argument('--skip-baseline-train', action='store_true')
    p.add_argument('--price-weight', type=float, default=0.05)
    p.add_argument('--consistency-weight', type=float, default=0.02)
    p.add_argument('--terminal-weight', type=float, default=1.0)
    p.add_argument('--anchor-weight', type=float, default=1.0)
    return p.parse_args()


if __name__ == '__main__':
    main(parse_args())
