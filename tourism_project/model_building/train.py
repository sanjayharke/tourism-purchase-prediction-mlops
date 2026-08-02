"""Tune, track, evaluate, and package a tourism purchase-prediction model."""

from __future__ import annotations

import argparse
import json
import platform
import warnings
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_predict,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from project_config import (
    CATEGORICAL_FEATURES,
    DEPLOYMENT_DIR,
    NUMERIC_FEATURES,
    RANDOM_STATE,
    REPORT_DIR,
    SPLIT_DIR,
    TARGET,
)


try:
    import mlflow
    import mlflow.sklearn

    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False


PARAMETER_GRID = {
    "model__n_estimators": [200, 400],
    "model__max_depth": [None, 8, 14],
    "model__min_samples_leaf": [1, 3],
    "model__class_weight": [None, "balanced_subsample"],
}


def load_splits(split_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    required = ["Xtrain.csv", "Xtest.csv", "ytrain.csv", "ytest.csv"]
    missing = [name for name in required if not (split_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing split artifacts: {missing}")

    X_train = pd.read_csv(split_dir / "Xtrain.csv")
    X_test = pd.read_csv(split_dir / "Xtest.csv")
    y_train = pd.read_csv(split_dir / "ytrain.csv")[TARGET].astype(int)
    y_test = pd.read_csv(split_dir / "ytest.csv")[TARGET].astype(int)

    if list(X_train.columns) != list(X_test.columns):
        raise ValueError("Train and test feature columns do not match.")
    if len(X_train) != len(y_train) or len(X_test) != len(y_test):
        raise ValueError("Feature/target row counts do not align.")
    return X_train, X_test, y_train, y_test


def build_pipeline() -> Pipeline:
    numeric_pipeline = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median"))]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    classifier = RandomForestClassifier(
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", classifier),
        ]
    )


def choose_business_threshold(
    model: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv: StratifiedKFold,
) -> tuple[float, pd.DataFrame]:
    """Choose a train-only threshold maximizing F2 (recall weighted twice)."""
    out_of_fold_probability = cross_val_predict(
        model,
        X_train,
        y_train,
        cv=cv,
        method="predict_proba",
        n_jobs=-1,
    )[:, 1]
    thresholds = np.round(np.arange(0.20, 0.71, 0.01), 2)
    rows = []
    for threshold in thresholds:
        prediction = (out_of_fold_probability >= threshold).astype(int)
        rows.append(
            {
                "threshold": threshold,
                "precision": precision_score(y_train, prediction, zero_division=0),
                "recall": recall_score(y_train, prediction, zero_division=0),
                "f1": f1_score(y_train, prediction, zero_division=0),
                "f2": fbeta_score(y_train, prediction, beta=2, zero_division=0),
            }
        )
    threshold_results = pd.DataFrame(rows)
    best_row = threshold_results.sort_values(
        ["f2", "recall", "precision"], ascending=False
    ).iloc[0]
    return float(best_row["threshold"]), threshold_results


def metric_set(
    y_true: pd.Series, probability: np.ndarray, threshold: float
) -> dict[str, float]:
    prediction = (probability >= threshold).astype(int)
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, prediction)),
        "precision": float(precision_score(y_true, prediction, zero_division=0)),
        "recall": float(recall_score(y_true, prediction, zero_division=0)),
        "f1": float(f1_score(y_true, prediction, zero_division=0)),
        "f2": float(fbeta_score(y_true, prediction, beta=2, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probability)),
        "average_precision": float(average_precision_score(y_true, probability)),
    }


def save_diagnostics(
    best_model: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    probability: np.ndarray,
    threshold: float,
    report_dir: Path,
) -> dict[str, float]:
    report_dir.mkdir(parents=True, exist_ok=True)
    prediction = (probability >= threshold).astype(int)

    selected_metrics = metric_set(y_test, probability, threshold)
    default_metrics = metric_set(y_test, probability, 0.50)
    all_metrics = {
        "business_threshold": selected_metrics,
        "default_threshold_0_50": default_metrics,
    }
    (report_dir / "metrics.json").write_text(
        json.dumps(all_metrics, indent=2), encoding="utf-8"
    )

    report = pd.DataFrame(
        classification_report(
            y_test,
            prediction,
            target_names=["Not Purchased", "Purchased"],
            output_dict=True,
            zero_division=0,
        )
    ).transpose()
    report.to_csv(report_dir / "classification_report.csv")

    matrix = confusion_matrix(y_test, prediction)
    pd.DataFrame(
        matrix,
        index=["Actual_No", "Actual_Yes"],
        columns=["Predicted_No", "Predicted_Yes"],
    ).to_csv(report_dir / "confusion_matrix.csv")

    pd.DataFrame(
        {
            "actual": y_test.to_numpy(),
            "purchase_probability": probability,
            "prediction": prediction,
        }
    ).to_csv(report_dir / "test_predictions.csv", index=False)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(6.4, 5.0))
    ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=["Not Purchased", "Purchased"],
    ).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"Confusion Matrix at Business Threshold = {threshold:.2f}")
    fig.tight_layout()
    fig.savefig(report_dir / "confusion_matrix.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8))
    RocCurveDisplay.from_predictions(
        y_test,
        probability,
        ax=axes[0],
        curve_kwargs={"color": "#0B6E75"},
    )
    axes[0].plot([0, 1], [0, 1], "--", color="gray", linewidth=1)
    axes[0].set_title("ROC Curve")
    PrecisionRecallDisplay.from_predictions(
        y_test,
        probability,
        ax=axes[1],
        curve_kwargs={"color": "#D97706"},
    )
    axes[1].set_title("Precision-Recall Curve")
    fig.tight_layout()
    fig.savefig(report_dir / "roc_pr_curves.png", dpi=180)
    plt.close(fig)

    transformed_names = best_model.named_steps[
        "preprocessor"
    ].get_feature_names_out()
    importance = best_model.named_steps["model"].feature_importances_
    importance_table = (
        pd.DataFrame({"feature": transformed_names, "importance": importance})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    importance_table.to_csv(report_dir / "feature_importance.csv", index=False)

    top = importance_table.head(15).sort_values("importance")
    fig, ax = plt.subplots(figsize=(8.0, 6.0))
    ax.barh(top["feature"], top["importance"], color="#2878B5")
    ax.set_xlabel("Random Forest importance")
    ax.set_title("Top 15 Predictive Features")
    fig.tight_layout()
    fig.savefig(report_dir / "feature_importance.png", dpi=180)
    plt.close(fig)
    return selected_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-dir", type=Path, default=SPLIT_DIR)
    parser.add_argument("--report-dir", type=Path, default=REPORT_DIR)
    parser.add_argument("--deployment-dir", type=Path, default=DEPLOYMENT_DIR)
    parser.add_argument(
        "--mlruns-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "mlruns",
    )
    args = parser.parse_args()

    X_train, X_test, y_train, y_test = load_splits(args.split_dir)
    args.report_dir.mkdir(parents=True, exist_ok=True)
    args.deployment_dir.mkdir(parents=True, exist_ok=True)

    pipeline = build_pipeline()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=PARAMETER_GRID,
        scoring="roc_auc",
        cv=cv,
        n_jobs=-1,
        refit=True,
        return_train_score=True,
        verbose=1,
    )

    run_context = nullcontext()
    run_id = None
    if MLFLOW_AVAILABLE:
        args.mlruns_dir.mkdir(parents=True, exist_ok=True)
        mlflow.set_tracking_uri(args.mlruns_dir.resolve().as_uri())
        mlflow.set_experiment("tourism-purchase-prediction")
        mlflow.sklearn.autolog(
            log_models=False,
            max_tuning_runs=50,
            silent=True,
        )
        run_context = mlflow.start_run(run_name="random_forest_grid_search")

    with run_context as active_run:
        if active_run is not None:
            run_id = active_run.info.run_id

        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_

        cv_results = pd.DataFrame(grid_search.cv_results_).sort_values(
            "rank_test_score"
        )
        cv_results.to_csv(args.report_dir / "cv_results_all_trials.csv", index=False)

        threshold, threshold_results = choose_business_threshold(
            best_model, X_train, y_train, cv
        )
        threshold_results.to_csv(
            args.report_dir / "threshold_analysis.csv", index=False
        )

        test_probability = best_model.predict_proba(X_test)[:, 1]
        selected_metrics = save_diagnostics(
            best_model,
            X_test,
            y_test,
            test_probability,
            threshold,
            args.report_dir,
        )

        bundle = {
            "model": best_model,
            "threshold": threshold,
            "target": TARGET,
            "feature_names": X_train.columns.tolist(),
            "numeric_features": NUMERIC_FEATURES,
            "categorical_features": CATEGORICAL_FEATURES,
            "best_parameters": grid_search.best_params_,
            "cross_validated_roc_auc": float(grid_search.best_score_),
            "test_metrics": selected_metrics,
            "trained_at_utc": datetime.now(timezone.utc).isoformat(),
            "random_state": RANDOM_STATE,
            "python_version": platform.python_version(),
            "sklearn_version": sklearn.__version__,
        }
        model_path = args.deployment_dir / "model.joblib"
        joblib.dump(bundle, model_path)

        metadata = {key: value for key, value in bundle.items() if key != "model"}
        (args.deployment_dir / "model_metadata.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )

        if MLFLOW_AVAILABLE:
            for key, value in grid_search.best_params_.items():
                mlflow.log_param(f"best_{key}", value)
            mlflow.log_param("selection_metric", "roc_auc")
            mlflow.log_param("threshold_objective", "F2 on out-of-fold training predictions")
            mlflow.log_metric("best_cv_roc_auc", float(grid_search.best_score_))
            mlflow.log_metrics(
                {f"test_{key}": value for key, value in selected_metrics.items()}
            )
            mlflow.log_artifacts(str(args.report_dir), artifact_path="evaluation")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                mlflow.sklearn.log_model(best_model, artifact_path="best_model")

    print("MODEL TRAINING AND TRACKING: PASSED")
    print(f"Training rows: {len(X_train):,}; testing rows: {len(X_test):,}")
    print(
        f"Grid search: {len(cv_results):,} parameter combinations x "
        f"{cv.get_n_splits()} folds"
    )
    print(f"Best CV ROC-AUC: {grid_search.best_score_:.4f}")
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Business threshold selected from training folds: {threshold:.2f}")
    print(
        "Test metrics: "
        f"accuracy={selected_metrics['accuracy']:.4f}, "
        f"precision={selected_metrics['precision']:.4f}, "
        f"recall={selected_metrics['recall']:.4f}, "
        f"F1={selected_metrics['f1']:.4f}, "
        f"F2={selected_metrics['f2']:.4f}, "
        f"ROC-AUC={selected_metrics['roc_auc']:.4f}"
    )
    print(f"All tuning trials logged to: {args.report_dir / 'cv_results_all_trials.csv'}")
    if MLFLOW_AVAILABLE:
        print(f"MLflow tracking: enabled (run_id={run_id})")
    else:
        print(
            "MLflow tracking: package unavailable in this runtime; "
            "CSV/JSON experiment logs were saved. Installing requirements.txt "
            "enables MLflow automatically."
        )
    print(f"Packaged model: {model_path}")
    print(f"Evaluation reports: {args.report_dir}")


if __name__ == "__main__":
    main()
