"""KryptoNet-40: the pretrained model released with Kryptonite-3.0.

Usage:
    from kryptonet import load_kryptonet
    model = load_kryptonet("kryptonet40.pt")
    logits = model(torch.from_numpy(X).float())    # X: (N, 40) array from the Datasets directory
    predictions = (logits.squeeze(1) > 0).long()
"""
import torch
import torch.nn as nn

N_INPUTS = 40
HIDDEN = (512, 512, 512, 512)


def kryptonet(n_inputs=N_INPUTS, hidden=HIDDEN):
    """A plain ReLU MLP that outputs one logit per input row."""
    layers, d = [], n_inputs
    for h in hidden:
        layers += [nn.Linear(d, h), nn.ReLU()]
        d = h
    return nn.Sequential(*layers, nn.Linear(d, 1))


def load_kryptonet(path="kryptonet40.pt"):
    model = kryptonet()
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    return model.eval()
