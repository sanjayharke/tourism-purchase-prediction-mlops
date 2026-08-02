"""Validate and register the tourism dataset by creating an auditable manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from project_config import DATA_MANIFEST_PATH, DATA_PATH, EXPECTED_COLUMNS, TARGET


def sha256sum(path: Path) -> str:
    """Return a SHA-256 checksum for reproducible dataset identification."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_dataset(data_path: Path) -> tuple[pd.DataFrame, dict]:
    """Validate the dataset schema and business-critical constraints."""
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    data = pd.read_csv(data_path)
    if data.empty:
        raise ValueError("The dataset is empty.")
    if data.columns.duplicated().any():
        duplicates = data.columns[data.columns.duplicated()].tolist()
        raise ValueError(f"Duplicate column names found: {duplicates}")

    missing_columns = sorted(set(EXPECTED_COLUMNS) - set(data.columns))
    unexpected_columns = sorted(set(data.columns) - set(EXPECTED_COLUMNS))
    if missing_columns:
        raise ValueError(f"Missing expected columns: {missing_columns}")

    target_values = set(data[TARGET].dropna().unique())
    if not target_values.issubset({0, 1}):
        raise ValueError(
            f"{TARGET} must contain only 0 and 1; found {sorted(target_values)}"
        )
    if data["CustomerID"].duplicated().any():
        raise ValueError("CustomerID must be unique.")

    class_counts = {
        str(int(label)): int(count)
        for label, count in data[TARGET].value_counts().sort_index().items()
    }
    manifest = {
        "registered_at_utc": datetime.now(timezone.utc).isoformat(),
        "file_name": data_path.name,
        "sha256": sha256sum(data_path),
        "rows": int(data.shape[0]),
        "columns": int(data.shape[1]),
        "column_names": data.columns.tolist(),
        "unexpected_columns": unexpected_columns,
        "total_missing_values": int(data.isna().sum().sum()),
        "duplicate_full_rows": int(data.duplicated().sum()),
        "duplicate_customer_ids": int(data["CustomerID"].duplicated().sum()),
        "target_distribution": class_counts,
        "target_positive_rate": round(float(data[TARGET].mean()), 6),
    }
    return data, manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    parser.add_argument("--manifest", type=Path, default=DATA_MANIFEST_PATH)
    args = parser.parse_args()

    data, manifest = validate_dataset(args.data)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("DATA REGISTRATION: PASSED")
    print(f"File: {args.data}")
    print(f"Shape: {data.shape[0]:,} rows x {data.shape[1]} columns")
    print(f"Schema: all {len(EXPECTED_COLUMNS)} expected columns are present")
    print(f"Missing values: {manifest['total_missing_values']:,}")
    print(f"Target counts: {manifest['target_distribution']}")
    print(f"Positive class rate: {manifest['target_positive_rate']:.2%}")
    print(f"SHA-256: {manifest['sha256']}")
    print(f"Manifest saved: {args.manifest}")


if __name__ == "__main__":
    main()

