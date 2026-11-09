"""Baseline strategies for the Thalesians Heston deep-hedging project.

All policies return the *target number of shares* held at each rebalancing
instant, with shape (n_paths, n_steps). The terminal observation is excluded.

The three benchmarks are:
    1. No hedging.
    2. Black--Scholes delta hedging with a fixed volatility.
    3. Oracle Heston delta hedging using the simulated latent variance.

The Oracle benchmark has more information than the realistic GRU policy,
which does not observe the true instantaneous Heston variance.

Required dependencies: numpy, scipy. PyTorch is optional and is used only
when inputs are torch tensors (to return matching torch tensors).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.special import ndtr


@dataclass(frozen=True)
class HestonParameters:
    """Risk-neutral Heston parameters; ``variance0`` is not needed for delta."""

    r: float = 0.03
    kappa: float = 2.0
    theta: float = 0.04
    xi: float = 0.30
    rho: float = -0.70

    def __post_init__(self) -> None:
        if self.kappa <= 0 or self.theta < 0 or self.xi <= 0:
            raise ValueError("Require kappa > 0, theta >= 0 and xi > 0.")
        if not -1 <= self.rho <= 1:
            raise ValueError("rho must lie between -1 and 1.")


def _to_numpy(values: object, name: str) -> np.ndarray:
    """Convert array or torch tensor to a float64 NumPy array."""
    if hasattr(values, "detach") and hasattr(values, "cpu"):
        values = values.detach().cpu().numpy()
    result = np.asarray(values, dtype=np.float64)
    if result.ndim != 2:
        raise ValueError(f"{name} must have shape (n_paths, n_steps + 1).")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} contains non-finite values.")
    return result


def _as_original_type(values: np.ndarray, reference: object):
    """Return a torch tensor on the original device when input was torch."""
    try:
        import torch
    except ImportError:
        return values
    if isinstance(reference, torch.Tensor):
        dtype = reference.dtype if reference.is_floating_point() else torch.float32
        return torch.as_tensor(values, dtype=dtype, device=reference.device)
    return values


def _check_stock_paths(stock_paths: object) -> np.ndarray:
    stocks = _to_numpy(stock_paths, "stock_paths")
    if stocks.shape[1] < 2 or np.any(stocks <= 0):
        raise ValueError("stock_paths needs at least 2 dates and positive prices.")
    return stocks


def _time_grid(maturity: float, n_steps: int) -> np.ndarray:
    if not np.isfinite(maturity) or maturity <= 0:
        raise ValueError("maturity must be positive and finite.")
    return maturity - np.arange(n_steps) * maturity / n_steps


def no_hedging_policy(stock_paths: object):
    """Hold zero shares at all decision times; shape (n_paths, n_steps)."""
    stocks = _check_stock_paths(stock_paths)
    positions = np.zeros((stocks.shape[0], stocks.shape[1] - 1))
    return _as_original_type(positions, stock_paths)


def black_scholes_delta(
    stock_price: object,
    strike: float,
    time_to_maturity: object,
    risk_free_rate: float,
    volatility: float,
) -> np.ndarray:
    """European call delta N(d1) using a fixed Black--Scholes volatility."""
    if strike <= 0 or volatility <= 0:
        raise ValueError("strike and volatility must be positive.")
    spot = np.asarray(stock_price, dtype=float)
    tau = np.asarray(time_to_maturity, dtype=float)
    if np.any(spot <= 0) or np.any(tau < 0):
        raise ValueError("stock_price must be positive; time cannot be negative.")
    safe_tau = np.maximum(tau, np.finfo(float).tiny)
    d1 = (
        np.log(spot / strike)
        + (risk_free_rate + 0.5 * volatility**2) * safe_tau
    ) / (volatility * np.sqrt(safe_tau))
    delta = ndtr(d1)
    # The terminal limit is not used by the policy, but makes the helper robust.
    terminal_delta = np.where(spot > strike, 1.0, np.where(spot < strike, 0.0, 0.5))
    return np.where(tau == 0, terminal_delta, delta)


def bs_delta_hedging_policy(
    stock_paths: object,
    strike: float,
    maturity: float,
    risk_free_rate: float = 0.03,
    volatility: float = 0.20,
):
    """Daily-rebalanced Black--Scholes call delta on a uniform time grid.

    ``volatility`` is fixed across dates and must not be set using future
    information. sqrt(theta) = 0.20 is a reasonable model-based reference.
    """
    stocks = _check_stock_paths(stock_paths)
    remaining = _time_grid(maturity, stocks.shape[1] - 1)
    positions = black_scholes_delta(
        stocks[:, :-1], strike, remaining[None, :], risk_free_rate, volatility
    )
    return _as_original_type(positions, stock_paths)


def _heston_cd(
    complex_u: np.ndarray,
    tau: float,
    params: HestonParameters,
) -> tuple[np.ndarray, np.ndarray]:
    """Little-Heston-trap characteristic-function coefficients C and D."""
    i = 1j
    kappa, theta, xi, rho, r = (
        params.kappa,
        params.theta,
        params.xi,
        params.rho,
        params.r,
    )
    b = kappa - rho * xi * i * complex_u
    d = np.sqrt(b * b + xi * xi * (complex_u * complex_u + i * complex_u))
    # Choose the square-root branch with nonnegative real part.
    d = np.where(np.real(d) < 0, -d, d)
    g = (b - d) / (b + d)
    decay = np.exp(-d * tau)
    log_term = np.log((1 - g * decay) / (1 - g))
    C = i * complex_u * r * tau + (kappa * theta / xi**2) * (
        (b - d) * tau - 2 * log_term
    )
    D = (b - d) / xi**2 * (1 - decay) / (1 - g * decay)
    return C, D


def heston_delta(
    stock_price: object,
    variance: object,
    strike: float,
    time_to_maturity: float,
    params: HestonParameters,
    n_quadrature: int = 96,
    integration_upper: float = 160.0,
    chunk_size: int = 4096,
) -> np.ndarray:
    """Heston European-call delta via risk-neutral stock-numeraire P1 integral.

    Delta = P1 = 1/2 + (1/pi) * integral_0^inf Re[e^(-iu ln K)
                 phi(u-i) / (phi(-i) * iu)] du.

    Uses Gauss--Legendre quadrature truncated at ``integration_upper``.
    Numerical accuracy can be checked by increasing both quadrature settings.
    For a given expiry this uses the *true* variance, so it is an oracle.
    """
    if strike <= 0 or not np.isfinite(strike):
        raise ValueError("strike must be positive and finite.")
    if time_to_maturity < 0 or not np.isfinite(time_to_maturity):
        raise ValueError("time_to_maturity must be nonnegative and finite.")
    if n_quadrature < 8 or integration_upper <= 0 or chunk_size < 1:
        raise ValueError("Invalid quadrature or chunk_size arguments.")
    spot, var = np.broadcast_arrays(
        np.asarray(stock_price, dtype=float), np.asarray(variance, dtype=float)
    )
    if np.any(~np.isfinite(spot)) or np.any(spot <= 0):
        raise ValueError("stock_price must be finite and positive.")
    if np.any(~np.isfinite(var)) or np.any(var < 0):
        raise ValueError("variance must be finite and nonnegative.")
    if time_to_maturity == 0:
        return np.where(spot > strike, 1.0, np.where(spot < strike, 0.0, 0.5))

    nodes, weights = leggauss(n_quadrature)
    u = 0.5 * integration_upper * (nodes + 1.0)
    integration_weights = 0.5 * integration_upper * weights
    # Under the share measure, phi(u-i) / phi(-i); phi(-i) = S exp(r*tau).
    C, D = _heston_cd(u - 1j, time_to_maturity, params)
    factor = integration_weights * np.exp(C - params.r * time_to_maturity) / (1j * u)

    flat_spot, flat_var = spot.reshape(-1), var.reshape(-1)
    result = np.empty_like(flat_spot)
    for start in range(0, flat_spot.size, chunk_size):
        stop = min(start + chunk_size, flat_spot.size)
        log_moneyness = np.log(flat_spot[start:stop] / strike)
        exponent = (1j * u[None, :] * log_moneyness[:, None]
                    + flat_var[start:stop, None] * D[None, :])
        integral = np.real(np.exp(exponent) @ factor)
        result[start:stop] = np.clip(0.5 + integral / np.pi, 0.0, 1.0)
    return result.reshape(spot.shape)


def heston_delta_hedging_policy(
    stock_paths: object,
    variance_paths: object,
    strike: float,
    maturity: float,
    params: HestonParameters | None = None,
    n_quadrature: int = 96,
    integration_upper: float = 160.0,
):
    """Oracle Heston delta hedging on the simulator's uniform time grid.

    The variance grid is the *actual simulated* v_t, not an estimated state.
    This is an information-advantaged benchmark against GRU with hidden v_t.
    """
    stocks = _check_stock_paths(stock_paths)
    variances = _to_numpy(variance_paths, "variance_paths")
    if stocks.shape != variances.shape:
        raise ValueError("stock_paths and variance_paths must have equal shapes.")
    if params is None:
        params = HestonParameters()
    remaining = _time_grid(maturity, stocks.shape[1] - 1)
    positions = np.empty((stocks.shape[0], stocks.shape[1] - 1), dtype=float)
    for t, tau in enumerate(remaining):
        # Full truncation in the simulator may temporarily produce raw negative
        # variance grid values. Only nonnegative instantaneous variance is valid.
        variance_t = np.maximum(variances[:, t], 0.0)
        positions[:, t] = heston_delta(
            stocks[:, t], variance_t, strike, float(tau), params,
            n_quadrature=n_quadrature,
            integration_upper=integration_upper,
        )
    return _as_original_type(positions, stock_paths)


if __name__ == "__main__":
    # Smoke test: replace these toy paths with your Part 1 simulated paths.
    rng = np.random.default_rng(42)
    n_paths, n_steps, maturity = 8, 63, 0.25
    stocks = np.full((n_paths, n_steps + 1), 100.0)
    shocks = rng.normal(size=(n_paths, n_steps))
    stocks[:, 1:] = 100 * np.exp(np.cumsum(0.2 * np.sqrt(maturity / n_steps)
                                             * shocks, axis=1))
    variances = np.full_like(stocks, 0.04)
    params = HestonParameters()

    zero = no_hedging_policy(stocks)
    bs = bs_delta_hedging_policy(stocks, strike=100, maturity=maturity)
    heston = heston_delta_hedging_policy(
        stocks, variances, strike=100, maturity=maturity, params=params
    )
    for name, positions in (("No hedging", zero), ("BS delta", bs),
                            ("Oracle Heston delta", heston)):
        print(f"{name}: shape={positions.shape}, "
              f"initial_position={positions[0, 0]:.4f}")
