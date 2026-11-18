"""Binary MLP; the final output is a logit, not a probability."""

from torch import nn


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_sizes):
        super().__init__()
        if input_dim <= 0 or not hidden_sizes or any(w <= 0 for w in hidden_sizes):
            raise ValueError("Input and hidden dimensions must be positive")
        layers = []
        previous = input_dim
        for width in hidden_sizes:
            layers.extend([nn.Linear(previous, width), nn.ReLU()])
            previous = width
        layers.append(nn.Linear(previous, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x).squeeze(-1)
