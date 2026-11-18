"""Low-rank adaptation (LoRA) of the linear layers in KryptoNet-40.

For a frozen layer y = W x + b with W of shape (d_out, d_in), LoRA trains

    y = W x + b + (alpha / r) * B (A x),   A: (r, d_in),  B: (d_out, r),

so the weight update Delta W = (alpha / r) * B A has rank at most r.
A starts random and B starts at zero, so Delta W = 0 and the adapted model
initially computes exactly the same function as the pretrained one.

PyTorch stores a batch as rows, X of shape (N, d_in). The forward pass therefore
evaluates (X A^T) B^T, which is the row-major form of B (A X): the input is first
projected to r numbers per example and never multiplied by a full d_out x d_in matrix.
"""

import copy
import math

import torch
import torch.nn as nn


# Positions of the nn.Linear modules inside the nn.Sequential built by kryptonet.py.
HIDDEN_LINEAR = (0, 2, 4, 6)
HEAD = 8


class LoRALinear(nn.Module):
    """Wrap a frozen nn.Linear with a trainable rank-r update."""

    def __init__(self, base, rank, alpha):
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError("LoRALinear wraps an nn.Linear layer")
        if int(rank) != rank or rank < 1:
            raise ValueError("rank must be a positive integer")
        self.base = base
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        self.A = nn.Parameter(torch.empty(self.rank, base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, self.rank))
        # Same default as nn.Linear and the peft library: Kaiming-uniform for A.
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))

    def forward(self, x):
        return self.base(x) + self.scaling * ((x @ self.A.T) @ self.B.T)

    def delta_weight(self):
        """Return the weight update (alpha / r) B A of shape (d_out, d_in)."""
        return self.scaling * (self.B @ self.A)

    def merged(self):
        """Return an ordinary nn.Linear computing the same function, for deployment."""
        layer = nn.Linear(self.base.in_features, self.base.out_features)
        with torch.no_grad():
            layer.weight.copy_(self.base.weight + self.delta_weight())
            layer.bias.copy_(self.base.bias)
        return layer

    def extra_repr(self):
        return f"rank={self.rank}, alpha={self.alpha}"


def apply_lora(model, rank, alpha, layers=HIDDEN_LINEAR, train_head=True):
    """Return a copy of KryptoNet with LoRA on `layers`; the original model is left unchanged.

    Every pretrained parameter is frozen except the output layer when train_head is True.
    The output layer keeps its pretrained weights as the starting point.
    """
    adapted = copy.deepcopy(model)
    for parameter in adapted.parameters():
        parameter.requires_grad_(False)
    for index in layers:
        if index == HEAD:
            raise ValueError("Adapt the output layer by setting train_head, not through LoRA")
        adapted[index] = LoRALinear(adapted[index], rank, alpha)
    for parameter in adapted[HEAD].parameters():
        parameter.requires_grad_(train_head)
    return adapted


def merge_lora(model):
    """Return a plain KryptoNet whose LoRA layers are folded into their weights."""
    merged = copy.deepcopy(model)
    for index, module in enumerate(merged):
        if isinstance(module, LoRALinear):
            merged[index] = module.merged()
    return merged


def trainable_parameters(model):
    return [parameter for parameter in model.parameters() if parameter.requires_grad]


def count_trainable(model):
    return sum(parameter.numel() for parameter in trainable_parameters(model))


def expected_lora_count(model, rank, layers=HIDDEN_LINEAR, train_head=True):
    """Closed-form count used in the report: r (d_in + d_out) per adapted layer, plus the head."""
    total = sum(rank * (model[i].in_features + model[i].out_features) for i in layers)
    if train_head:
        total += model[HEAD].weight.numel() + model[HEAD].bias.numel()
    return total
