"""Regenerate the four submitted hidden-label arrays from recorded checkpoints.

Run from the repository root with the course-supplied Datasets/ and kryptonet40.pt:
    python paper/export_hidden_labels.py --output /path/to/new-hiddenlabels

The three main datasets use sign-CatBoost seed 42. K40-Adapt uses the recorded
LoRA rank-4 seed-42 checkpoint (alpha=8). The output directory must be new.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from catboost.pipeline import predict as cat_predict
from common import load_hidden
from lora.kryptonet import load_kryptonet
from lora.model import apply_lora

MODELS = ROOT / "results/strict_87ec2ce_20261118/models"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def predictions():
    """Return predictions in the supplied hidden arrays' original row order."""
    labels = {}
    sources = {}
    for dataset in ("20", "30", "40"):
        model = MODELS / f"{dataset}_sign-CatBoost_s42"
        pred, _ = cat_predict(model, load_hidden(dataset))
        labels[f"y_predicted_{dataset}.npy"] = pred
        sources[dataset] = {"method": "sign-CatBoost", "seed": 42,
                            "checkpoint": str(model.relative_to(ROOT)),
                            "model_sha256": digest(model / "model.cbm")}

    torch.set_num_threads(1)
    torch.manual_seed(42)
    base = load_kryptonet(ROOT / "kryptonet40.pt")
    model = apply_lora(base, rank=4, alpha=8.0, train_head=True)
    checkpoint = MODELS / "40-adapt_lora-r4_s42.pt"
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    model.eval()
    x = torch.from_numpy(np.ascontiguousarray(load_hidden("40-adapt"), dtype=np.float32))
    with torch.no_grad():
        scores = model(x).squeeze(1).numpy()
    labels["y_predicted_40_adapt.npy"] = (scores > 0).astype(np.int64)
    sources["40-adapt"] = {"method": "LoRA", "rank": 4, "alpha": 8.0,
                           "seed": 42, "checkpoint": str(checkpoint.relative_to(ROOT)),
                           "model_sha256": digest(checkpoint),
                           "pretrained_sha256": digest(ROOT / "kryptonet40.pt")}

    for name, vector in labels.items():
        if vector.shape != (10000,) or not np.isin(vector, [0, 1]).all():
            raise ValueError(f"Unexpected hidden-label format: {name}")
    return labels, sources


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument("--output", type=Path, help="Write into a new directory")
    choice.add_argument("--verify", action="store_true", help="Compare with the existing hiddenlabels directory")
    args = parser.parse_args()
    labels, sources = predictions()
    if args.verify:
        for name, vector in labels.items():
            saved = np.load(ROOT / "hiddenlabels" / name, allow_pickle=False)
            if not np.array_equal(vector, saved):
                raise ValueError(f"Existing hidden labels do not match checkpoint: {name}")
            print("Matched", name)
    else:
        output = args.output.resolve()
        output.mkdir(parents=True, exist_ok=False)
        for name, vector in labels.items():
            np.save(output / name, vector, allow_pickle=False)
        (output / "provenance.json").write_text(json.dumps(sources, indent=2) + "\n")
        print("Wrote", output)


if __name__ == "__main__":
    main()
