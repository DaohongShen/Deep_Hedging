"""Task 1: Heston model setup and path simulation (risk-neutral measure)."""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# Example parameters; chosen for an illustrative 3-month European call.
S0 = 100.0                  # initial stock price
v0 = 0.04                   # initial variance
r = 0.03                    # risk-free rate
kappa = 2.0                 # mean reversion rate
theta = 0.04                # long-term variance
xi = 0.30                   # volatility of variance
rho = -0.70                 # correlation between stock and variance   

T = 0.25                    # time to maturity (in years)
K = 100.0                   # strike price

n_steps = 63                 # daily hedging, approx. 252 trading days/year
n_paths = 10_000             # number of Monte Carlo paths
seed = 42                    # random seed for reproducibility


def simulate_heston(S0, v0, r, kappa, theta, xi, rho, T, n_steps, n_paths, seed=42):
    """Simulate Heston S and v on a risk-neutral grid.

    Full-truncation Euler for raw variance, and log-Euler for the stock.
    The raw variance can be negative numerically. Only max(v, 0) is used
    in the drift/diffusion coefficients and as the reported usable variance.
    """

    """First check whether the parameters are valid."""
    if not (-1 <= rho <= 1):
        raise ValueError('rho must be between -1 and 1')
    if min(S0, v0, kappa, theta, xi, T) < 0 or n_steps < 1 or n_paths < 1:
        raise ValueError('Invalid parameters')

    """Set up the simulation grid and arrays."""
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    times = np.linspace(0.0, T, n_steps + 1)
    S = np.empty((n_paths, n_steps + 1), dtype=float)
    v_raw = np.empty_like(S)
    S[:, 0] = S0
    v_raw[:, 0] = v0

    for i in range(n_steps):
        #Construct two brownian increments with correlation rho
        z1 = rng.standard_normal(n_paths)   #Standard normal for the first Brownian increment
        z_ind = rng.standard_normal(n_paths)    #Independent standard normal for the second Brownian increment
        z2 = rho * z1 + np.sqrt(1 - rho**2) * z_ind     #So that the correlation between z1 and z2 is rho and the variance of z2 is 1
        v_pos = np.maximum(v_raw[:, i], 0.0)
        #Update the stock price using log-Euler and the raw variance using full-truncation Euler
        S[:, i + 1] = S[:, i] * np.exp(
            (r - 0.5 * v_pos) * dt + np.sqrt(v_pos * dt) * z1
        )               #Derived by Itos Lemma for log(S) 
        v_raw[:, i + 1] = (
            v_raw[:, i] + kappa * (theta - v_pos) * dt
            + xi * np.sqrt(v_pos * dt) * z2
        )

    return times, S, np.maximum(v_raw, 0.0), v_raw


def main():
    times, S, v, v_raw = simulate_heston(
        S0, v0, r, kappa, theta, xi, rho, T, n_steps, n_paths, seed
    )
    payoff = np.maximum(S[:, -1] - K, 0.0)  #Formula for payoff of a European call option
    disc_mean = np.exp(-r*T) * S[:, -1].mean()  #Discounted mean of the terminal stock price
    print(f'S paths shape: {S.shape}')
    print(f'Variance paths shape: {v.shape}')
    print(f'Mean discounted terminal stock price: {disc_mean:.4f} (target {S0:.4f})')
    print(f'Fraction of negative raw variance grid values: {(v_raw < 0).mean():.5%}')
    print(f'Monte Carlo discounted call payoff (illustrative): {np.exp(-r*T)*payoff.mean():.4f}')
    print(f'Feller check 2*kappa*theta >= xi**2: {2*kappa*theta >= xi**2}')

    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    for j in range(12):     #Actually we have 10,000 paths but we only plot 12 for clarity
        axes[0].plot(times, S[j], lw=1)
        axes[1].plot(times, np.sqrt(v[j]), lw=1)
    axes[0].set(ylabel='Stock price', title='Simulated Heston paths (12 examples)')
    axes[1].set(xlabel='Time (years)', ylabel='Instantaneous volatility')
    fig.tight_layout()


    output_path = Path(__file__).resolve().parent / "heston_simulated_paths.png"

    fig.savefig(output_path, dpi=160)
    plt.show()


if __name__ == '__main__':
    main()


