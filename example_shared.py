"""Check the shared data and metric interfaces without training a model."""

import numpy as np

from common import evaluate, load_split


def main():
    for name in ("20", "30", "40", "40-adapt"):
        x_train, y_train, x_val, y_val, x_test, y_test = load_split(name)
        print(f"{name}: train={x_train.shape}, validation={x_val.shape}, test={x_test.shape}")
        # Illustrative constant predictions check the interface, not a trained model.
        print("  validation example:", evaluate(y_val, np.zeros_like(y_val)))


if __name__ == "__main__":
    main()
