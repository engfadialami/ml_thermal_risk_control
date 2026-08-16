"""Stage 5 - Train and save the final Random Forest prediction models.

This script preserves the final-model configuration approved in p4.py. It
trains one complete preprocessing-and-classification pipeline for each
prediction horizon using the training and validation sequences only. The test
sequences are not read or used.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# ---------------------------------------------------------
# Configuration copied from the approved Stage 4 script
# ---------------------------------------------------------

INPUT_FILE = Path("data/processed/incubator_stage3_targets.csv")
MODEL_DIR = Path("models/stage5")

TARGET_COLUMNS = [
    "temperature_risk_5m",
    "temperature_risk_10m",
    "temperature_risk_15m",
]

MODEL_FILENAMES = {
    "temperature_risk_5m": "random_forest_5m.joblib",
    "temperature_risk_10m": "random_forest_10m.joblib",
    "temperature_risk_15m": "random_forest_15m.joblib",
}

CLASS_ORDER = ["LOW", "NORMAL", "HIGH"]
TRAIN_SEQUENCES = [1, 2, 3, 5]
VALIDATION_SEQUENCES = [7]

DROP_FROM_FEATURES = [
    "record_id",
    "sequence_id",
    "estimated_timestamp",
    "target_segment_id",
    # Future information - never use as ML input
    "target_avg_t_5m",
    "target_avg_t_10m",
    "target_avg_t_15m",
    # Classification outputs
    "temperature_risk_5m",
    "temperature_risk_10m",
    "temperature_risk_15m",
]

RF_CONFIG = {
    "temperature_risk_5m": {
        "max_depth": 6,
        "min_samples_leaf": 20,
    },
    "temperature_risk_10m": {
        "max_depth": 6,
        "min_samples_leaf": 20,
    },
    "temperature_risk_15m": {
        "max_depth": 6,
        "min_samples_leaf": 10,
    },
}


def make_json_safe(value: Any) -> Any:
    """Convert pandas/numpy values into values supported by JSON."""

    if pd.isna(value):
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def load_final_training_data(input_file: Path) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    list[str],
]:
    """Load Stage 3 data and return the final Train + Validation dataset."""

    if not input_file.exists():
        raise FileNotFoundError(
            f"Stage 3 input file was not found: {input_file}"
        )

    df = pd.read_csv(input_file)

    required_columns = {"record_id", "sequence_id", *TARGET_COLUMNS}
    missing_required = sorted(required_columns.difference(df.columns))
    if missing_required:
        raise ValueError(
            "Stage 3 input is missing required columns: "
            + ", ".join(missing_required)
        )

    df = (
        df.dropna(subset=TARGET_COLUMNS)
        .sort_values("record_id")
        .reset_index(drop=True)
    )

    feature_columns = [
        column
        for column in df.columns
        if column not in DROP_FROM_FEATURES
    ]

    final_sequences = TRAIN_SEQUENCES + VALIDATION_SEQUENCES
    final_train_df = df[df["sequence_id"].isin(final_sequences)].copy()

    if final_train_df.empty:
        raise ValueError(
            "No rows were found for the final training sequences: "
            f"{final_sequences}"
        )

    X_final_train = final_train_df[feature_columns].copy()
    Y_final_train = final_train_df[TARGET_COLUMNS].copy()

    return X_final_train, Y_final_train, feature_columns


def build_tree_preprocessor(
    numeric_features: list[str],
    categorical_features: list[str],
) -> ColumnTransformer:
    """Build the same tree preprocessing used in Stage 4."""

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    return ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features),
    ])


def train_and_save_models(
    input_file: Path = INPUT_FILE,
    model_dir: Path = MODEL_DIR,
) -> None:
    """Train the three final models and save their deployment artifacts."""

    X_final_train, Y_final_train, feature_columns = (
        load_final_training_data(input_file)
    )

    numeric_features = (
        X_final_train.select_dtypes(include="number").columns.tolist()
    )
    categorical_features = (
        X_final_train.select_dtypes(exclude="number").columns.tolist()
    )

    tree_preprocessor = build_tree_preprocessor(
        numeric_features,
        categorical_features,
    )

    model_dir.mkdir(parents=True, exist_ok=True)

    for target in TARGET_COLUMNS:
        config = RF_CONFIG[target]

        model = Pipeline([
            ("preprocessor", clone(tree_preprocessor)),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=200,
                    max_depth=config["max_depth"],
                    min_samples_leaf=config["min_samples_leaf"],
                    max_features="sqrt",
                    class_weight="balanced",
                    n_jobs=-1,
                    random_state=42,
                ),
            ),
        ])

        model.fit(X_final_train, Y_final_train[target])
        joblib.dump(model, model_dir / MODEL_FILENAMES[target])

    # Use a real prepared row as the Swagger request example.
    example_row = X_final_train.iloc[-1]
    example_features = {
        feature: make_json_safe(example_row[feature])
        for feature in feature_columns
    }

    metadata = {
        "model_type": "RandomForestClassifier",
        "model_version": "1.0",
        "training_sequences": TRAIN_SEQUENCES,
        "validation_sequences": VALIDATION_SEQUENCES,
        "test_sequences_used_for_training": [],
        "feature_count": len(feature_columns),
        "feature_columns": feature_columns,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "target_columns": TARGET_COLUMNS,
        "class_order": CLASS_ORDER,
        "model_files": MODEL_FILENAMES,
        "random_forest_config": RF_CONFIG,
        "example_features": example_features,
        "training_rows": len(X_final_train),
    }

    with (model_dir / "model_metadata.json").open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(metadata, file, indent=2, ensure_ascii=False)

    print("\n=== Stage 5 Model Export Complete ===")
    print(f"Training rows: {len(X_final_train):,}")
    print(f"Features: {len(feature_columns)}")
    print(f"Saved models: {len(TARGET_COLUMNS)}")
    print(f"Output directory: {model_dir}")
    print("Test sequences used for training: none")


if __name__ == "__main__":
    train_and_save_models()