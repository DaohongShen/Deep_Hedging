"""Neural-network policies for discrete-time Heston deep hedging.

All policies receive observable state features with shape
    (batch_size, n_hedging_dates, input_size)
and return target stock holdings with shape
    (batch_size, n_hedging_dates).

For a stock path with 64 price observations, there are normally 63 hedge
positions: one at each trading date, excluding the terminal maturity date.

These policies do not construct state features or train themselves. They can be
trained end-to-end with hedge_terminal_error and the chosen loss function from
hedging_objective.py. GRU is the default model.
"""

import math

import torch
from torch import nn


# -----------------------------------------------------------------------------
# Module 1: LSTM policy. Chosen for its ability to capture long-term temporal dependencies in market dynamics and hedging decisions.
# -----------------------------------------------------------------------------
class LSTMPolicy(nn.Module):
    """Uses a recurrent hidden state and cell state to process past observations."""

    def __init__(self, input_size: int, hidden_size: int = 64,
                 num_layers: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.output_layer = nn.Linear(hidden_size, 1)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        # states: (batch, dates, features)
        hidden_states, _ = self.lstm(states)
        # Predict one target stock position per hedging date.
        return self.output_layer(hidden_states).squeeze(-1)


# -----------------------------------------------------------------------------
# Module 2: GRU policy (We will select this). Chosen for its ability to capture temporal dependencies with lower computational complexity, making it efficient for dynamic hedging.
# -----------------------------------------------------------------------------
class GRUPolicy(nn.Module):
    """Uses a lightweight recurrent hidden state for sequential hedge decisions."""

    def __init__(self, input_size: int, hidden_size: int = 64,
                 num_layers: int = 1):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.output_layer = nn.Linear(hidden_size, 1)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        # A GRU at date t has only seen states up to and including date t.
        hidden_states, _ = self.gru(states)
        return self.output_layer(hidden_states).squeeze(-1)


# -----------------------------------------------------------------------------
# Module 3: Transformer Encoder policy. Chosen for its self-attention mechanism, which captures complex relationships across historical market observations.
# -----------------------------------------------------------------------------
class TransformerEncoderPolicy(nn.Module):
    """Causally masked encoder: a hedge at date t cannot access future dates."""

    def __init__(self, input_size: int, hidden_size: int = 64,
                 num_layers: int = 1, num_heads: int = 4,
                 dropout: float = 0.0, max_seq_len: int = 256):
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        self.max_seq_len = max_seq_len
        self.input_layer = nn.Linear(input_size, hidden_size)
        self.position_embedding = nn.Embedding(max_seq_len, hidden_size)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=4 * hidden_size,
            dropout=dropout,
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_layer = nn.Linear(hidden_size, 1)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        # states: (batch, dates, features)
        n_dates = states.size(1)
        if n_dates > self.max_seq_len:
            raise ValueError(f"Sequence length {n_dates} exceeds {self.max_seq_len}")

        positions = torch.arange(n_dates, device=states.device)
        x = self.input_layer(states) * math.sqrt(self.input_layer.out_features)
        x = x + self.position_embedding(positions).unsqueeze(0)

        # True entries above the diagonal BLOCK attention to future dates.
        future_mask = torch.triu(
            torch.ones(n_dates, n_dates, device=states.device, dtype=torch.bool),
            diagonal=1,
        )
        hidden_states = self.encoder(x, mask=future_mask)
        return self.output_layer(hidden_states).squeeze(-1)


# -----------------------------------------------------------------------------
# Model selection: GRU
# -----------------------------------------------------------------------------
def build_policy(input_size: int, model_type: str = "gru", **kwargs) -> nn.Module:
    model_classes = {
        "lstm": LSTMPolicy,
        "gru": GRUPolicy,
        "transformer": TransformerEncoderPolicy,
    }
    model_type = model_type.lower()
    if model_type not in model_classes:
        raise ValueError("model_type must be 'lstm', 'gru', or 'transformer'")
    return model_classes[model_type](input_size=input_size, **kwargs)


if __name__ == "__main__":
    # Shape test only: replace these dummy states with actual state features
    # (e.g., price/moneyness, past returns, realised volatility, remaining time).
    example_states = torch.randn(4, 63, 6)

    for name in ("lstm", "gru", "transformer"):
        model = build_policy(input_size=6, model_type=name)
        hedge_positions = model(example_states)
        assert hedge_positions.shape == (4, 63)
        print(f"{name.upper()} hedge positions shape: {tuple(hedge_positions.shape)}")

    policy = build_policy(input_size=6)  # GRU is selected by default.
    print(f"Selected model: {policy.__class__.__name__}")
