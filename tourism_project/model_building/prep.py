"""Clean the registered tourism data and create leakage-safe train/test splits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from project_config import (
    CATEGORICAL_FEATURES,
    DATA_PATH,
    EXPECTED_COLUMNS,
    IDENTIFIER_COLUMNS,
    MODEL_FEATURES,
    NUMERIC_FEATURES,
    RANDOM_STATE,
    SPLIT_DIR,
    TARGET,
    TEST_SIZE,
)


CATEGORY_REPLACEMENTS = {
    "Gender": {"Fe Male": "Female"},
    "Occupation": {"Free Lancer": "Freelancer"},
    "MaritalStatus": {"Unmarried": "Single"},
}


def clean_dataset(data: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Apply reproducible cleaning without learning anything from the test set."""
    data = data.copy()
    data.columns = data.columns.str.strip()

    missing_columns = sorted(set(EXPECTED_COLUMNS) - set(data.columns))
    if missing_columns:
        raise ValueError(f"Missing expected columns: {missing_columns}")

    raw_rows = len(data)
    missing_target_rows = int(data[TARGET].isna().sum())
    data = data.dropna(subset=[TARGET]).copy()

    for column in CATEGORICAL_FEATURES:
        data[column] = data[column].astype("string").str.strip()
        if column in CATEGORY_REPLACEMENTS:
            data[column] = data[column].replace(CATEGORY_REPLACEMENTS[column])

    for column in NUMERIC_FEATURES + [TARGET]:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    # These two fields identify a person/row but carry no reusable behavior signal.
    data = data.drop(columns=IDENTIFIER_COLUMNS)

    # After identifiers are removed, repeated records can otherwise leak across splits.
    duplicate_model_rows = int(data.duplicated().sum())
    data = data.drop_duplicates().reset_index(drop=True)
    data[TARGET] = data[TARGET].astype(int)

    invalid_target_values = sorted(set(data[TARGET].unique()) - {0, 1})
    if invalid_target_values:
        raise ValueError(f"Invalid target values after cleaning: {invalid_target_values}")

    ordered_columns = MODEL_FEATURES + [TARGET]
    data = data[ordered_columns]

    summary = {
        "raw_rows": raw_rows,
        "rows_with_missing_target_removed": missing_target_rows,
        "duplicate_model_rows_removed": duplicate_model_rows,
        "clean_rows": int(len(data)),
        "identifier_columns_removed": IDENTIFIER_COLUMNS,
        "category_replacements": CATEGORY_REPLACEMENTS,
        "remaining_missing_values": {
            column: int(count)
            for column, count in data.isna().sum().items()
            if count > 0
        },
    }
    return data, summary


def save_splits(data_path: Path, output_dir: Path) -> dict:
    """Clean, stratify, split, and save four workflow-ready CSV files."""
    raw = pd.read_csv(data_path)
    cleaned, summary = clean_dataset(raw)

    X = cleaned.drop(columns=TARGET)
    y = cleaned[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    X_train.to_csv(output_dir / "Xtrain.csv", index=False)
    X_test.to_csv(output_dir / "Xtest.csv", index=False)
    y_train.to_frame(TARGET).to_csv(output_dir / "ytrain.csv", index=False)
    y_test.to_frame(TARGET).to_csv(output_dir / "ytest.csv", index=False)

    summary.update(
        {
            "random_state": RANDOM_STATE,
            "test_size": TEST_SIZE,
            "train_shape": [int(X_train.shape[0]), int(X_train.shape[1])],
            "test_shape": [int(X_test.shape[0]), int(X_test.shape[1])],
            "train_positive_rate": round(float(y_train.mean()), 6),
            "test_positive_rate": round(float(y_test.mean()), 6),
            "output_files": [
                "Xtrain.csv",
                "Xtest.csv",
                "ytrain.csv",
                "ytest.csv",
            ],
        }
    )
    (output_dir / "preparation_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    parser.add_argument("--output-dir", type=Path, default=SPLIT_DIR)
    args = parser.parse_args()

    summary = save_splits(args.data, args.output_dir)
    print("DATA PREPARATION: PASSED")
    print(f"Raw rows: {summary['raw_rows']:,}")
    print(
        "Removed: "
        f"{summary['rows_with_missing_target_removed']} missing-target rows and "
        f"{summary['duplicate_model_rows_removed']} duplicate model rows"
    )
    print(f"Removed identifier columns: {summary['identifier_columns_removed']}")
    print(f"Training split: {tuple(summary['train_shape'])}")
    print(f"Testing split: {tuple(summary['test_shape'])}")
    print(
        "Positive rates: "
        f"train={summary['train_positive_rate']:.2%}, "
        f"test={summary['test_positive_rate']:.2%}"
    )
    print(f"Saved workflow artifacts to: {args.output_dir}")


if __name__ == "__main__":
    main()

