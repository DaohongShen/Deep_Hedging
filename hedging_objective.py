"""Task 2: Hedging objective with bid-ask spread, commission and MSE loss.

Keep this file beside heston_model_setup.py (Task 1). Requires numpy and torch.
The simulated Heston S is the mid price. Sell 1 European call; hedge with stock.
At t=0,...,t_(n-1), trade to phi_i; close positions at maturity t_n.
The policy's decisions must be based only on information available at t_i.
"""
from dataclasses import dataclass
import numpy as np
import torch
import math

from heston_model_setup import (
    S0, v0, r, kappa, theta, xi, rho, T, K, n_steps, n_paths, seed,
    simulate_heston,
)

# Illustrative assumptions, not market-observed parameters.
# We set these two parameters as we assumed the trading cost is proportional.
FULL_BID_ASK_SPREAD = 0.001  # 0.10% full spread (half spread on each side)
COMMISSION_RATE = 0.0002   # 0.02% additional commission per side


@dataclass
class HedgeResult:
    terminal_wealth: torch.Tensor
    option_payoff: torch.Tensor
    hedging_error: torch.Tensor
    total_spread_cost: torch.Tensor
    total_commission: torch.Tensor
    total_turnover: torch.Tensor

#Three optimisation criteria, choose one for your deep hedge training. All are differentiable and can be used with PyTorch autograd.

#Mean-Variance of terminal hedging error. Minimises overall hedging error by penalising both bias and variability, providing a simple and stable training objective.
def mean_variance_loss(hedging_error: torch.Tensor) -> torch.Tensor:
    """MSE = E[E_T^2] = Var(E_T) + E[E_T]^2; differentiable."""
    return hedging_error.square().mean()


#CVaR (Expected Shortfall) of terminal hedging error. Focuses on reducing extreme hedging losses, making it suitable for managing tail risk under adverse market conditions.
def cvar_loss(hedging_error: torch.Tensor, alpha: float = 0.95) -> torch.Tensor:        #Alpha is the confidence level for CVaR, e.g., 0.95 for 95% CVaR
    """Empirical CVaR of terminal hedging losses."""
    if not 0.0 <= alpha < 1.0:
        raise ValueError("alpha must be in [0, 1)")
    if hedging_error.ndim != 1 or hedging_error.numel() == 0:
        raise ValueError("hedging_error must be a nonempty 1D tensor")
  
    losses = -hedging_error
    n_tail = max(1, math.ceil((1.0 - alpha) * losses.numel()))

    return torch.topk(losses, k=n_tail, largest=True).values.mean()

#Exponential utility of terminal PnL. Balances expected hedging PnL and risk according to the investor's risk aversion, with stronger penalties for large losses.
def exponential_utility_loss(
    hedging_error: torch.Tensor, gamma: float = 0.1
) -> torch.Tensor:
    """Negative certainty equivalent under exponential utility."""
    if gamma <= 0:
        raise ValueError("gamma must be positive")
    if hedging_error.ndim != 1 or hedging_error.numel() == 0:
        raise ValueError("hedging_error must be a nonempty 1D tensor")

    return (
        torch.logsumexp(-gamma * hedging_error, dim=0)
        - math.log(hedging_error.numel())
    ) / gamma



def hedge_terminal_error(
    mid_prices: torch.Tensor,
    hedge_positions: torch.Tensor,
    option_premium: float | torch.Tensor,
    strike: float = K,
    maturity: float = T,
    risk_free_rate: float = r,
    full_spread: float = FULL_BID_ASK_SPREAD,
    commission_rate: float = COMMISSION_RATE,
) -> HedgeResult:
    """Account for self-financing trades, bid/ask and commission (short 1 call).

    mid_prices: [batch, n_steps+1] Heston *mid* prices.
    hedge_positions: [batch, n_steps] shares held during each interval.
    At each hedge time i, buy at ask = S_i*(1+spread/2) and sell at
    bid = S_i*(1-spread/2). Commission is additional to spread.
    Cash starts with option premium; it can be negative (borrowing at r).
    At maturity, liquidate shares at bid/ask and pay the call payoff.
    Returned terminal_wealth is cash AFTER liquidation, BEFORE option payoff.
    """
    if mid_prices.ndim != 2 or hedge_positions.ndim != 2:
        raise ValueError('mid_prices and hedge_positions must be 2D tensors')
    batch, n_times = mid_prices.shape
    if n_times < 2 or hedge_positions.shape != (batch, n_times - 1):
        raise ValueError('Expected positions shape (batch, n_times-1)')
    if maturity <= 0 or strike < 0 or not 0 <= full_spread < 2 or commission_rate < 0:
        raise ValueError('Invalid maturity, strike, spread, or commission')
    if not bool(torch.isfinite(mid_prices).all()) or not bool(torch.isfinite(hedge_positions).all()):
        raise ValueError('All prices and positions must be finite')
    if bool((mid_prices <= 0).any()):
        raise ValueError('Mid-prices must be strictly positive')

    # Include initial trade (0 -> phi_0) and final liquidation (phi_{n-1} -> 0).
    zeros = torch.zeros_like(hedge_positions[:, :1])
    old_positions = torch.cat((zeros, hedge_positions), dim=1)
    new_positions = torch.cat((hedge_positions, zeros), dim=1)
    trades = new_positions - old_positions
    mid_notionals = trades.abs() * mid_prices   #Your theoretical trade price that does not include the spread or commission
    spread_costs = 0.5 * full_spread * mid_notionals
    commissions = commission_rate * mid_notionals

    premium = torch.as_tensor(option_premium, dtype=mid_prices.dtype, device=mid_prices.device)
    cash = torch.zeros_like(mid_prices[:, 0]) + premium
    growth = torch.exp(torch.as_tensor(
        risk_free_rate * maturity / (n_times - 1),
        dtype=mid_prices.dtype, device=mid_prices.device,
    ))
    for i in range(n_times):
        if i > 0:
            cash = cash * growth
        # Equivalent to buying at ask or selling at bid, PLUS commission.
        cash = cash - trades[:, i] * mid_prices[:, i] - spread_costs[:, i] - commissions[:, i]

    call_payoff = torch.clamp(mid_prices[:, -1] - strike, min=0.0)
    return HedgeResult(
        terminal_wealth=cash,
        option_payoff=call_payoff,
        hedging_error=cash - call_payoff,
        total_spread_cost=spread_costs.sum(dim=1),
        total_commission=commissions.sum(dim=1),
        total_turnover=mid_notionals.sum(dim=1),
    )


def run_checks() -> None:
    """Hand-calculated accounting and differentiation tests."""
    prices = torch.tensor([[100.0, 110.0]], dtype=torch.float64)
    holdings = torch.tensor([[1.0]], dtype=torch.float64, requires_grad=True)
    result = hedge_terminal_error(prices, holdings, option_premium=5.0,
                                  strike=100.0, maturity=1.0,
                                  risk_free_rate=0.0, full_spread=0.02,
                                  commission_rate=0.001)
    # Buy 1 at 101+0.1, sell 1 at 108.9-0.11. Cash = 12.69; payoff=10.
    assert torch.allclose(result.terminal_wealth, torch.tensor([12.69], dtype=torch.float64))
    assert torch.allclose(result.hedging_error, torch.tensor([2.69], dtype=torch.float64))
    assert torch.allclose(result.total_spread_cost, torch.tensor([2.10], dtype=torch.float64))
    assert torch.allclose(result.total_commission, torch.tensor([0.21], dtype=torch.float64))
    mean_variance_loss(result.hedging_error).backward()
    assert holdings.grad is not None and torch.isfinite(holdings.grad).all()

    # Zero trading costs and zero price changes: hedge cannot create stock PnL.
    flat = torch.tensor([[100.0, 100.0, 100.0]], dtype=torch.float64)
    pos = torch.tensor([[0.4, 0.7]], dtype=torch.float64)
    flat_result = hedge_terminal_error(flat, pos, option_premium=3.0,
                                       strike=100.0, maturity=1.0,
                                       risk_free_rate=0.0, full_spread=0.0,
                                       commission_rate=0.0)
    assert torch.allclose(flat_result.terminal_wealth, torch.tensor([3.0], dtype=torch.float64))
    print('Accounting and PyTorch gradient checks passed.')


def main() -> None:
    run_checks()
    _, stock_paths, _, _ = simulate_heston(
        S0, v0, r, kappa, theta, xi, rho, T, n_steps, n_paths, seed
    )
    # Demo pricing: same paths as hedging. Later use a separate independent
    # pricing sample, then train and evaluate on separate samples as well.
    option_premium = float(np.exp(-r*T) * np.maximum(stock_paths[:, -1] - K, 0).mean())
    stock = torch.as_tensor(stock_paths, dtype=torch.float64)

    # Example ONLY, not an optimised hedge; replace with LSTM/GRU in Task 3.
    positions = torch.full((n_paths, n_steps), 0.5, dtype=torch.float64)
    result = hedge_terminal_error(stock, positions, option_premium)
    loss = mean_variance_loss(result.hedging_error)
    print(f'Illustrative short-call premium: {option_premium:.4f}')
    print(f'Mean hedging error: {result.hedging_error.mean().item():.4f}')
    print(f'Mean squared hedging error: {loss.item():.4f}')
    print(f'Hedging RMSE: {loss.sqrt().item():.4f}')
    print(f'Mean spread cost: {result.total_spread_cost.mean().item():.4f}')
    print(f'Mean commission: {result.total_commission.mean().item():.4f}')
    print('This is a fixed 0.5-share accounting demo, NOT a trained deep hedge.')


if __name__ == '__main__':
    main()
