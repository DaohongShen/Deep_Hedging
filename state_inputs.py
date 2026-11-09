"""State inputs for the Heston deep-hedging project.

The policy observes prices and price-derived information, NOT the Heston
instantaneous variance v_t (which is latent in a realistic trading setting).

Features, in order:
    [stock_price, log_return, RV5, RV20, time_to_maturity, moneyness]

All rolling statistics at time t use only returns observed at or before t.
The maturity grid point is omitted by default, giving one state per hedge
rebalancing decision (e.g. 64 stock-price times -> 63 hedge decisions).
"""

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np
import torch

FEATURE_NAMES: Tuple[str, ...] = (
    "stock_price",
    "log_return",
    "RV5",
    "RV20",
    "time_to_maturity",
    "moneyness",
)


def rolling_realised_volatility(      #This function calculates rolling realised volatility (RV5 and RV20) from historical log returns, providing observable estimates of market volatility without relying on the unobservable Heston variance v_t
    log_returns: np.ndarray,
    window: int,
    dt: float,
) -> np.ndarray:
    """Annualised rolling historical volatility, using only available history.

    Parameters
    ----------
    log_returns : array of shape (n_paths, n_times)
        The first column should be zero (no return is observed at t=0).
    window : int
        Number of *past returns* in the rolling window (5 or 20).
    dt : float
        Years per simulation step; annualisation is 1/sqrt(dt).

    A rolling standard deviation requires at least two observed returns.
    The value is zero if there is insufficient history, and early windows
    use all returns currently available (not zero-padded future returns).
    """
    returns = np.asarray(log_returns, dtype=np.float64)
    if returns.ndim != 2:
        raise ValueError("log_returns must have shape (n_paths, n_times).")
    if not isinstance(window, int) or window < 2:
        raise ValueError("window must be an integer >= 2.")
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be a positive finite number.")

    result = np.zeros_like(returns)
    n_times = returns.shape[1]
    for t in range(2, n_times):
        # Returns at t=1,...,t are known when the hedge is chosen at t.
        start = max(1, t - window + 1)
        historical = returns[:, start : t + 1]
        result[:, t] = historical.std(axis=1, ddof=0) / np.sqrt(dt)
    return result


def build_state_inputs(
    stock_paths: np.ndarray,
    strike: float,
    maturity: float,
    *,
    include_terminal: bool = False,
) -> np.ndarray:
    """Construct observable states from simulated stock prices.

    Parameters
    ----------
    stock_paths : array, shape (n_paths, n_times)
        Heston simulated stock prices, including initial and terminal times.
        Do NOT pass variance paths: the policy never observes true v_t.
    strike : float
        European call strike K.
    maturity : float
        Option maturity T measured in years.
    include_terminal : bool, default False
        False: return only hedging-decision times (exclude maturity).
        True: include the maturity state, useful for diagnostics only.

    Returns
    -------
    states : float32 array, shape (n_paths, n_decisions, 6)
        Feature order is FEATURE_NAMES.
    """
    prices = np.asarray(stock_paths, dtype=np.float64)
    if prices.ndim != 2 or prices.shape[1] < 2:
        raise ValueError("stock_paths must have shape (n_paths, n_times>=2).")
    if prices.shape[0] < 1 or not np.all(np.isfinite(prices)) or np.any(prices <= 0):
        raise ValueError("stock_paths must contain positive finite prices.")
    if not np.isfinite(strike) or strike <= 0:
        raise ValueError("strike must be positive and finite.")
    if not np.isfinite(maturity) or maturity <= 0:
        raise ValueError("maturity must be positive and finite.")

    n_times = prices.shape[1]
    dt = maturity / (n_times - 1)
    log_returns = np.zeros_like(prices)
    log_returns[:, 1:] = np.log(prices[:, 1:] / prices[:, :-1])

    rv5 = rolling_realised_volatility(log_returns, window=5, dt=dt)
    rv20 = rolling_realised_volatility(log_returns, window=20, dt=dt)
    tau = maturity - np.arange(n_times, dtype=np.float64) * dt
    tau = np.maximum(tau, 0.0)
    tau = np.broadcast_to(tau, prices.shape)
    moneyness = prices / strike

    states = np.stack((prices, log_returns, rv5, rv20, tau, moneyness), axis=-1)
    if not include_terminal:
        states = states[:, :-1, :]
    return states.astype(np.float32)


@dataclass
class StateStandardScaler:
    """Standardise features with training-set statistics only.

    Fit on training paths; reuse without refitting on validation/test paths.
    Very small/constant-variance columns use scale=1.
    """

    mean_: Optional[np.ndarray] = None
    scale_: Optional[np.ndarray] = None

    def fit(self, train_states: np.ndarray) -> "StateStandardScaler":
        x = np.asarray(train_states, dtype=np.float64)
        if x.ndim != 3 or x.shape[-1] != len(FEATURE_NAMES):
            raise ValueError("Expected training states of shape (N, T, 6).")
        if not np.all(np.isfinite(x)):
            raise ValueError("Training states must be finite.")
        self.mean_ = x.mean(axis=(0, 1)).astype(np.float32)
        scale = x.std(axis=(0, 1))
        self.scale_ = np.where(scale > 1e-8, scale, 1.0).astype(np.float32)
        return self

    def transform(self, states: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("Call scaler.fit(training_states) before transform().")
        x = np.asarray(states, dtype=np.float32)
        if x.ndim != 3 or x.shape[-1] != len(FEATURE_NAMES):
            raise ValueError("Expected states of shape (N, T, 6).")
        return ((x - self.mean_) / self.scale_).astype(np.float32)

    def fit_transform(self, train_states: np.ndarray) -> np.ndarray:
        return self.fit(train_states).transform(train_states)


def to_gru_tensor(states: np.ndarray, device: str = "cpu") -> torch.Tensor:
    """Return a float32 tensor with shape (batch, sequence, features)."""
    x = np.asarray(states, dtype=np.float32)
    if x.ndim != 3 or x.shape[-1] != len(FEATURE_NAMES):
        raise ValueError("Expected states of shape (N, T, 6).")
    return torch.as_tensor(x, dtype=torch.float32, device=device)


if __name__ == "__main__":
    # Small independent demo. Replace these prices with the S paths returned
    # from heston_model_setup.py when integrating the complete project.
    rng = np.random.default_rng(42)
    n_paths, n_steps, S0, T, K = 100, 63, 100.0, 0.25, 100.0
    dt = T / n_steps
    shocks = rng.standard_normal((n_paths, n_steps))
    demo_paths = np.concatenate(
        [np.full((n_paths, 1), S0),
         S0 * np.exp(np.cumsum(-0.5 * 0.2**2 * dt + 0.2 * np.sqrt(dt) * shocks,
                              axis=1))],
        axis=1,
    )

    # Split by complete paths BEFORE fitting any scaling statistics.
    train_paths, test_paths = demo_paths[:80], demo_paths[80:]
    train_states = build_state_inputs(train_paths, strike=K, maturity=T)
    test_states = build_state_inputs(test_paths, strike=K, maturity=T)

    scaler = StateStandardScaler()
    train_tensor = to_gru_tensor(scaler.fit_transform(train_states))
    test_tensor = to_gru_tensor(scaler.transform(test_states))

    print("Features:", FEATURE_NAMES)
    print("Train GRU tensor:", tuple(train_tensor.shape))
    print("Test GRU tensor:", tuple(test_tensor.shape))
    print("True variance v_t included?", "No")
