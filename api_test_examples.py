"""Select five real held-out rows for testing the prediction API.

The examples come from the normalized Stage 4 test file. They are never added
to model training. LOW/HIGH outcomes and large positive/negative temperature
slopes are prioritized so the API is tested with more than a stable NORMAL row.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


TEST_FILE = Path("data/processed/incubator_stage4_test_normalized.csv")
METADATA_FILE = Path("models/stage5/model_metadata.json")
OUTPUT_FILE = Path("models/stage5/api_test_examples.json")

TARGET_COLUMNS = [
    "temperature_risk_5m",
    "temperature_risk_10m",
    "temperature_risk_15m",
]


def make_json_safe(value: Any) -> Any:
    """Convert pandas/numpy scalar values into JSON-compatible values."""

    if pd.isna(value):
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def select_row(
    frame: pd.DataFrame,
    candidate_mask: pd.Series,
    score_column: str,
    ascending: bool,
    selected_indices: set[int],
) -> int | None:
    """Return the best candidate that has not already been selected."""

    candidates = frame[
        candidate_mask & ~frame.index.isin(selected_indices)
    ]

    if candidates.empty:
        return None

    return int(
        candidates.sort_values(
            score_column,
            ascending=ascending,
        ).index[0]
    )


def build_examples(
    test_file: Path = TEST_FILE,
    metadata_file: Path = METADATA_FILE,
    output_file: Path = OUTPUT_FILE,
) -> list[dict[str, Any]]:
    """Select and save five API examples from the held-out test data."""

    if not metadata_file.exists():
        raise FileNotFoundError(
            f"Model metadata was not found: {metadata_file}. Run p5.py first."
        )

    if not test_file.exists():
        raise FileNotFoundError(
            f"Normalized test file was not found: {test_file}"
        )

    with metadata_file.open("r", encoding="utf-8") as file:
        metadata = json.load(file)

    feature_columns = metadata["feature_columns"]
    test_df = pd.read_csv(test_file).dropna(subset=TARGET_COLUMNS).copy()

    required_columns = set(feature_columns + TARGET_COLUMNS)
    missing_columns = sorted(required_columns.difference(test_df.columns))
    if missing_columns:
        raise ValueError(
            "The normalized test file is missing required columns: "
            + ", ".join(missing_columns)
        )

    # Prefer actual temperature-slope features and exclude unrelated slopes.
    slope_columns = [
        column
        for column in feature_columns
        if "_t_slope_" in column.lower()
    ]
    if not slope_columns:
        slope_columns = [
            column
            for column in feature_columns
            if "slope" in column.lower()
        ]
    if not slope_columns:
        raise ValueError("No slope features were found in the model metadata.")

    slope_values = test_df[slope_columns].apply(
        pd.to_numeric,
        errors="coerce",
    )

    # Mean direction identifies rising/falling trends. Maximum absolute slope
    # identifies the strongest temperature change across the available windows.
    test_df["_directional_slope"] = slope_values.mean(axis=1, skipna=True)
    test_df["_maximum_absolute_slope"] = slope_values.abs().max(
        axis=1,
        skipna=True,
    )

    all_low = test_df[TARGET_COLUMNS].eq("LOW").all(axis=1)
    any_low = test_df[TARGET_COLUMNS].eq("LOW").any(axis=1)
    all_high = test_df[TARGET_COLUMNS].eq("HIGH").all(axis=1)
    any_high = test_df[TARGET_COLUMNS].eq("HIGH").any(axis=1)
    every_row = pd.Series(True, index=test_df.index)

    selection_plan = [
        (
            "LOW at all horizons with the strongest downward temperature slope",
            all_low,
            "_directional_slope",
            True,
        ),
        (
            "LOW in at least one horizon with a strong downward temperature slope",
            any_low,
            "_directional_slope",
            True,
        ),
        (
            "HIGH at all horizons with the strongest upward temperature slope",
            all_high,
            "_directional_slope",
            False,
        ),
        (
            "HIGH in at least one horizon with a strong upward temperature slope",
            any_high,
            "_directional_slope",
            False,
        ),
        (
            "Largest remaining absolute temperature slope",
            every_row,
            "_maximum_absolute_slope",
            False,
        ),
    ]

    selected: list[tuple[str, int]] = []
    selected_indices: set[int] = set()

    for description, mask, score_column, ascending in selection_plan:
        row_index = select_row(
            test_df,
            mask,
            score_column,
            ascending,
            selected_indices,
        )

        # A same-class pattern may not exist. Fall back to the strongest unused
        # slope while keeping all five examples real and unique.
        if row_index is None:
            row_index = select_row(
                test_df,
                every_row,
                "_maximum_absolute_slope",
                False,
                selected_indices,
            )
            description += " (strongest available fallback row)"

        if row_index is None:
            raise ValueError("The test file contains fewer than five usable rows.")

        selected.append((description, row_index))
        selected_indices.add(row_index)

    examples = []

    for example_id, (description, row_index) in enumerate(selected, start=1):
        row = test_df.loc[row_index]

        features = {
            feature: make_json_safe(row[feature])
            for feature in feature_columns
        }
        actual_targets = {
            target: str(row[target])
            for target in TARGET_COLUMNS
        }
        temperature_slopes = {
            column: make_json_safe(row[column])
            for column in slope_columns
        }

        examples.append({
            "example_id": example_id,
            "description": description,
            "record_id": make_json_safe(row.get("record_id")),
            "sequence_id": make_json_safe(row.get("sequence_id")),
            "temperature_slopes": temperature_slopes,
            "actual_targets": actual_targets,
            "request_body": {"features": features},
        })

    output = {
        "source": str(test_file),
        "note": (
            "These are real normalized held-out test rows. They are used only "
            "for API demonstration, never for model training."
        ),
        "examples": examples,
    }

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as file:
        json.dump(output, file, indent=2, ensure_ascii=False)

    return examples


def main() -> None:
    examples = build_examples()

    print("\n=== Five API Test Examples Created ===")
    for example in examples:
        print(f"\nExample {example['example_id']}: {example['description']}")
        print(f"Actual targets: {example['actual_targets']}")
        print(f"Temperature slopes: {example['temperature_slopes']}")

    print(f"\nSaved to: {OUTPUT_FILE}")
    print("Copy one example's request_body into POST /predict in Swagger.")


if __name__ == "__main__":
    main()
