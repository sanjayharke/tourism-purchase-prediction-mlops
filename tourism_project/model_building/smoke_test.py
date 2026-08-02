"""Fail fast if the packaged model cannot make a valid prediction."""

from pathlib import Path

import joblib
import pandas as pd

from project_config import DEPLOYMENT_DIR, SPLIT_DIR


def main() -> None:
    model_path = DEPLOYMENT_DIR / "model.joblib"
    test_path = SPLIT_DIR / "Xtest.csv"
    if not model_path.exists():
        raise FileNotFoundError(model_path)
    if not test_path.exists():
        raise FileNotFoundError(test_path)

    bundle = joblib.load(model_path)
    sample = pd.read_csv(test_path).head(1)
    sample = sample[bundle["feature_names"]]
    probability = float(bundle["model"].predict_proba(sample)[0, 1])
    if not 0.0 <= probability <= 1.0:
        raise AssertionError(f"Invalid probability: {probability}")
    print("MODEL SMOKE TEST: PASSED")
    print(f"Input columns: {sample.shape[1]}")
    print(f"Purchase probability: {probability:.4f}")
    print(f"Operating threshold: {bundle['threshold']:.2f}")


if __name__ == "__main__":
    main()

