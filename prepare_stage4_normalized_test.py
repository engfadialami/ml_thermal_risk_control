"""Create the normalized Stage 4 test dataset for sequences 8 and 9.

The original Stage 3 file is read only. A constant +0.2 degC coordinate
translation is applied only to absolute-temperature columns. Temperature-risk
labels, slopes, operating mode, controller variables, humidity, and identifiers
remain unchanged.
"""

from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

INPUT_FILE = Path("data/processed/incubator_stage3_targets.csv")
OUTPUT_FILE = Path("data/processed/incubator_stage4_test_normalized.csv")

TEST_SEQUENCES = [8, 9]
TEMPERATURE_OFFSET_C = 0.2

TARGET_COLUMNS = [
    "temperature_risk_5m",
    "temperature_risk_10m",
    "temperature_risk_15m",
]

ABSOLUTE_TEMPERATURE_COLUMNS = [
    "s1_t",
    "s2_t",
    "s3_t",
    "s4_t",
    "avg_t",
    "s2_t_mean_last_5m",
    "s3_t_mean_last_5m",
    "s2_t_mean_last_10m",
    "s3_t_mean_last_10m",
    "s2_t_mean_last_15m",
    "s3_t_mean_last_15m",
    "target_avg_t_5m",
    "target_avg_t_10m",
    "target_avg_t_15m",
]

UNCHANGED_TEMPERATURE_COLUMNS = [
    "sensor_spread_c",
    "avg_t_slope_last_5m",
    "avg_t_slope_last_10m",
    "avg_t_slope_last_15m",
]


# ---------------------------------------------------------
# Build and verify the corrected test dataset
# ---------------------------------------------------------

def build_normalized_test_dataset():
    if INPUT_FILE.resolve() == OUTPUT_FILE.resolve():
        raise ValueError("OUTPUT_FILE must not overwrite the original Stage 3 file.")

    df = pd.read_csv(INPUT_FILE)

    required_columns = (
        ["record_id", "sequence_id", "operating_mode"]
        + TARGET_COLUMNS
        + ABSOLUTE_TEMPERATURE_COLUMNS
        + UNCHANGED_TEMPERATURE_COLUMNS
    )

    missing_columns = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    test_df = (
        df[df["sequence_id"].isin(TEST_SEQUENCES)]
        .sort_values("record_id")
        .copy()
    )

    found_sequences = sorted(test_df["sequence_id"].unique().tolist())
    if found_sequences != TEST_SEQUENCES:
        raise ValueError(
            f"Expected test sequences {TEST_SEQUENCES}, found {found_sequences}."
        )

    found_modes = sorted(test_df["operating_mode"].dropna().unique().tolist())
    if found_modes != ["HTCH"]:
        raise ValueError(
            f"Sequences 8 and 9 must contain only HTCH rows; found {found_modes}."
        )

    labels_before = test_df[TARGET_COLUMNS].copy(deep=True)
    unchanged_before = test_df[
        [
            column for column in test_df.columns
            if column not in ABSOLUTE_TEMPERATURE_COLUMNS
            and column not in TARGET_COLUMNS
        ]
    ].copy(deep=True)
    temperatures_before = test_df[ABSOLUTE_TEMPERATURE_COLUMNS].copy(deep=True)

    test_df[ABSOLUTE_TEMPERATURE_COLUMNS] = (
        test_df[ABSOLUTE_TEMPERATURE_COLUMNS]
        + TEMPERATURE_OFFSET_C
    )

    pd.testing.assert_frame_equal(
        test_df[TARGET_COLUMNS],
        labels_before,
        check_dtype=True,
    )

    pd.testing.assert_frame_equal(
        test_df[unchanged_before.columns],
        unchanged_before,
        check_dtype=True,
    )

    for column in ABSOLUTE_TEMPERATURE_COLUMNS:
        valid = temperatures_before[column].notna()
        actual_change = (
            test_df.loc[valid, column]
            - temperatures_before.loc[valid, column]
        )

        if not np.allclose(
            actual_change,
            TEMPERATURE_OFFSET_C,
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError(f"Incorrect offset detected in {column}.")

        if not test_df.loc[~valid, column].isna().all():
            raise ValueError(f"Missing values changed in {column}.")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    test_df.to_csv(OUTPUT_FILE, index=False)

    saved_df = pd.read_csv(OUTPUT_FILE)

    if saved_df.columns.tolist() != df.columns.tolist():
        raise ValueError("The saved test file does not preserve the Stage 3 schema.")

    if len(saved_df) != len(test_df):
        raise ValueError("The saved test file has an unexpected row count.")

    pd.testing.assert_frame_equal(
        saved_df[TARGET_COLUMNS],
        labels_before.reset_index(drop=True),
        check_dtype=False,
    )

    usable_rows = len(saved_df.dropna(subset=TARGET_COLUMNS))

    print("\n=== Stage 4 normalized test dataset ===")
    print(f"Input file:       {INPUT_FILE}")
    print(f"Output file:      {OUTPUT_FILE}")
    print(f"Sequences:        {TEST_SEQUENCES}")
    print(f"Operating mode:   {found_modes}")
    print(f"Temperature shift: +{TEMPERATURE_OFFSET_C:.1f} degC")
    print(f"Rows written:     {len(saved_df):,}")
    print(f"Usable test rows: {usable_rows:,}")
    print("Risk labels:      unchanged")
    print("Slopes/spread:    unchanged")

    return saved_df


if __name__ == "__main__":
    build_normalized_test_dataset()
