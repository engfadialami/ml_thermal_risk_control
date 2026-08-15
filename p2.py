"""Stage 2: time reconstruction, feature engineering, and future targets.

Input
-----
data/processed/incubator_prepared.csv
    The 53,061-row output produced by p1.py.

Outputs
-------
data/processed/incubator_stage2_master.csv
    The original Stage 1 records plus useful Stage 2 columns.

data/processed/incubator_stage2_ml.csv
    A compact Stage 3 table containing only identifiers, the approved
    Version 1 features, one quality filter, and the future targets.

reports/stage2/*.csv
    Sequence, quality, feature, and target reports. These reports are not
    model features.
"""

# Imports and configuration
from pathlib import Path

import numpy as np
import pandas as pd


INPUT_FILE = Path("data/processed/incubator_prepared.csv")
MASTER_OUTPUT_FILE = Path("data/processed/incubator_stage2_master.csv")
ML_OUTPUT_FILE = Path("data/processed/incubator_stage2_ml.csv")
REPORTS_OUTPUT_DIR = Path("reports/stage2")

EXPECTED_STAGE1_ROWS = 53_061
MAX_CONTINUOUS_GAP_MINUTES = 3
MIN_EXPECTED_SAMPLE_SECONDS = 10
MAX_EXPECTED_SAMPLE_SECONDS = 30
MAX_TARGET_MATCH_ERROR_SECONDS = 30

SENSORS = ("s1", "s2", "s3", "s4")
VALID_SENSOR_FLAGS = {"A", "P", "ERR"}

TEMP_PLAUSIBLE_MIN_C = 0
TEMP_PLAUSIBLE_MAX_C = 60
HUMIDITY_PLAUSIBLE_MIN_PCT = 0
HUMIDITY_PLAUSIBLE_MAX_PCT = 100

HISTORY_WINDOWS_MINUTES = (5, 10, 15)
TARGET_HORIZONS_MINUTES = (5, 10, 15)

EXPECTED_STAGE1_COLUMNS = [
    "record_id",
    "timestamp",
    "s1_t",
    "s1_h",
    "s1_flag",
    "s2_t",
    "s2_h",
    "s2_flag",
    "s3_t",
    "s3_h",
    "s3_flag",
    "s4_t",
    "s4_h",
    "s4_flag",
    "avg_t",
    "avg_h",
    "duty_pct",
    "duty_correction_pct",
    "heater_on",
    "humidity_fan_on",
    "servo_position",
    "operating_mode",
    "sensor_spread_c",
    "active_sensors_count",
    "rotation_minutes_remaining",
    "low_edge_counter",
    "high_edge_counter",
]

# The current active-sensor temperatures and heater duty are primary.
PRIMARY_CURRENT_FEATURES = [
    "s2_t",
    "s3_t",
    "duty_pct",
]

# s2_t and s3_t are not repeated here because they are already primary.
# Together, the primary and supporting lists contain all four temperatures.
SUPPORTING_CURRENT_FEATURES = [
    "s1_t",
    "s4_t",
    "s1_h",
    "s2_h",
    "s3_h",
    "s4_h",
    "duty_correction_pct",
    "sensor_spread_c",
    "avg_t",
    "avg_h",
    "humidity_fan_on",
    "operating_mode",
]

EXCLUDED_MODEL_COLUMNS = [
    "heater_on",
    "low_edge_counter",
    "high_edge_counter",
    "rotation_minutes_remaining",
    "active_sensors_count",
    "servo_position",
]


def history_feature_name(source: str, minutes: int) -> str:
    """Return the saved name of one historical feature."""
    if source == "humidity_fan_on":
        return f"humidity_fan_on_fraction_last_{minutes}m"
    return f"{source}_mean_last_{minutes}m"


HISTORY_SOURCE_FEATURES = [
    "s2_t",
    "s3_t",
    "duty_pct",
    "humidity_fan_on",
]

HISTORY_FEATURES = [
    history_feature_name(source, minutes)
    for minutes in HISTORY_WINDOWS_MINUTES
    for source in HISTORY_SOURCE_FEATURES
]

VERSION1_FEATURES = (
    PRIMARY_CURRENT_FEATURES
    + SUPPORTING_CURRENT_FEATURES
    + HISTORY_FEATURES
)

TARGET_COLUMNS = [
    f"target_avg_t_{minutes}m"
    for minutes in TARGET_HORIZONS_MINUTES
]

ML_METADATA_COLUMNS = [
    "record_id",
    "sequence_id",
    "estimated_timestamp",
    "sensor_quality_issue_count",
]

ML_TABLE_COLUMNS = (
    ML_METADATA_COLUMNS
    + VERSION1_FEATURES
    + TARGET_COLUMNS
)


# Stage 1 input
def load_stage1_data(input_file: Path = INPUT_FILE) -> pd.DataFrame:
    """Load and validate the fixed Stage 1 dataset."""
    data = pd.read_csv(input_file, parse_dates=["timestamp"])

    if data.shape != (EXPECTED_STAGE1_ROWS, len(EXPECTED_STAGE1_COLUMNS)):
        raise ValueError(
            "Expected Stage 1 shape "
            f"({EXPECTED_STAGE1_ROWS}, {len(EXPECTED_STAGE1_COLUMNS)}), "
            f"but loaded {data.shape}."
        )

    if data.columns.tolist() != EXPECTED_STAGE1_COLUMNS:
        raise ValueError("The Stage 1 columns or their order have changed.")

    if data["timestamp"].isna().any():
        raise ValueError("Stage 1 contains a missing timestamp.")

    if not data["record_id"].is_monotonic_increasing:
        raise ValueError("record_id is not in acquisition order.")

    if data["record_id"].duplicated().any():
        raise ValueError("Stage 1 contains duplicate record_id values.")

    return data


# Sequences and estimated sub-minute timestamps
def add_time_structure(
    data: pd.DataFrame,
    max_gap_minutes: int = MAX_CONTINUOUS_GAP_MINUTES,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create sequences and distribute repeated readings within each minute."""
    result = data.copy().reset_index(drop=True)

    gap_minutes = (
        result["timestamp"].diff().dt.total_seconds().div(60)
    )

    sequence_start = (
        gap_minutes.isna()
        | gap_minutes.lt(0)
        | gap_minutes.gt(max_gap_minutes)
    )

    result["sequence_id"] = sequence_start.cumsum().astype("int32")

    internal_gap = gap_minutes.mask(sequence_start)
    if not internal_gap.dropna().between(0, max_gap_minutes).all():
        raise ValueError("A sequence contains an invalid internal time gap.")

    estimated_ns = np.empty(len(result), dtype=np.int64)
    sequence_rows = []

    for sequence_id, group in result.groupby("sequence_id", sort=False):
        positions = group.index.to_numpy()
        row_count = len(group)
        start_time = group["timestamp"].iloc[0]
        end_time = group["timestamp"].iloc[-1]
        duration_seconds = (end_time - start_time).total_seconds()

        if row_count > 1 and duration_seconds > 0:
            sample_seconds = duration_seconds / (row_count - 1)
            sample_status = (
                "confirmed"
                if MIN_EXPECTED_SAMPLE_SECONDS
                <= sample_seconds
                <= MAX_EXPECTED_SAMPLE_SECONDS
                else "review"
            )
        else:
            sample_seconds = np.nan
            sample_status = "not_estimable"

        # The source timestamp has one-minute resolution. Readings sharing a
        # minute are placed at equally spaced interval centres inside that
        # minute. This preserves every recorded minute and avoids inventing a
        # sequence-wide clock drift.
        for minute, minute_group in group.groupby(
            "timestamp", sort=False
        ):
            minute_positions = minute_group.index.to_numpy()
            readings_in_minute = len(minute_group)
            offsets_ns = (
                (2 * np.arange(readings_in_minute, dtype=np.int64) + 1)
                * 60_000_000_000
                // (2 * readings_in_minute)
            )
            estimated_ns[minute_positions] = minute.value + offsets_ns

        first_position = positions[0]
        gap_before = gap_minutes.iloc[first_position]

        if first_position == 0:
            start_reason = "dataset_start"
        elif gap_before < 0:
            start_reason = "backward_timestamp"
        else:
            start_reason = f"gap_over_{max_gap_minutes}m"

        sequence_rows.append(
            {
                "sequence_id": int(sequence_id),
                "start_reason": start_reason,
                "gap_before_minutes": gap_before,
                "start_record_id": int(group["record_id"].iloc[0]),
                "end_record_id": int(group["record_id"].iloc[-1]),
                "start_time": start_time,
                "end_time": end_time,
                "row_count": row_count,
                "duration_minutes": duration_seconds / 60,
                "estimated_sample_interval_seconds": sample_seconds,
                "sampling_interval_status": sample_status,
            }
        )

    result["estimated_timestamp"] = pd.to_datetime(estimated_ns)
    sequence_summary = pd.DataFrame(sequence_rows)

    inside_recorded_minute = (
        result["estimated_timestamp"].ge(result["timestamp"])
        & result["estimated_timestamp"].lt(
            result["timestamp"] + pd.Timedelta(minutes=1)
        )
    )
    if not inside_recorded_minute.all():
        raise ValueError(
            "An estimated timestamp falls outside its recorded minute."
        )

    review_sequences = sequence_summary[
        "sampling_interval_status"
    ].eq("review")

    if review_sequences.any():
        ids = sequence_summary.loc[review_sequences, "sequence_id"].tolist()
        raise ValueError(
            "Unexpected sampling interval in sequence(s): "
            f"{ids}. Inspect sequence_summary."
        )

    for _, group in result.groupby("sequence_id", sort=False):
        if len(group) > 1 and not group[
            "estimated_timestamp"
        ].is_monotonic_increasing:
            raise ValueError(
                "Estimated timestamps are not increasing within a sequence."
            )

    return result, sequence_summary


# Sensor-quality reports and compact master-table indicators
def add_sensor_quality(
    data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Add three row-level indicators and return detailed sensor reports."""
    result = data.copy()

    missing_count = pd.Series(0, index=result.index, dtype="int8")
    err_count = pd.Series(0, index=result.index, dtype="int8")
    issue_count = pd.Series(0, index=result.index, dtype="int8")

    summary_rows = []
    sensor_issue_masks = {}

    for sensor in SENSORS:
        temperature = result[f"{sensor}_t"]
        humidity = result[f"{sensor}_h"]
        flag = result[f"{sensor}_flag"]

        has_missing = temperature.isna() | humidity.isna()
        is_err = flag.eq("ERR")
        unexpected_missing = has_missing & ~is_err
        invalid_flag = ~flag.isin(VALID_SENSOR_FLAGS)

        implausible_value = (
            temperature.notna()
            & ~temperature.between(
                TEMP_PLAUSIBLE_MIN_C,
                TEMP_PLAUSIBLE_MAX_C,
            )
        ) | (
            humidity.notna()
            & ~humidity.between(
                HUMIDITY_PLAUSIBLE_MIN_PCT,
                HUMIDITY_PLAUSIBLE_MAX_PCT,
            )
        )

        err_with_values = (
            is_err
            & result[[f"{sensor}_t", f"{sensor}_h"]]
            .notna()
            .any(axis=1)
        )

        if err_with_values.any():
            raise ValueError(
                f"{sensor}: an ERR state contains a sensor value."
            )

        sensor_issue = (
            unexpected_missing | invalid_flag | implausible_value
        )
        sensor_issue_masks[sensor] = sensor_issue

        missing_count += has_missing.astype("int8")
        err_count += is_err.astype("int8")
        issue_count += sensor_issue.astype("int8")

        summary_rows.append(
            {
                "sensor": sensor,
                "missing_readings": int(has_missing.sum()),
                "explicit_ERR": int(is_err.sum()),
                "unexpected_missing": int(unexpected_missing.sum()),
                "invalid_flags": int(invalid_flag.sum()),
                "implausible_values": int(implausible_value.sum()),
            }
        )

    result["missing_sensor_count"] = missing_count
    result["sensor_err_count"] = err_count
    result["sensor_quality_issue_count"] = issue_count

    sensor_quality_summary = (
        pd.DataFrame(summary_rows).set_index("sensor")
    )

    issue_row_mask = result["sensor_quality_issue_count"].gt(0)
    sensor_issue_rows = result.loc[
        issue_row_mask,
        EXPECTED_STAGE1_COLUMNS
        + [
            "sequence_id",
            "estimated_timestamp",
            "missing_sensor_count",
            "sensor_err_count",
            "sensor_quality_issue_count",
        ],
    ].copy()

    if not sensor_issue_rows.empty:
        sensor_issue_rows["affected_sensors"] = [
            ",".join(
                sensor
                for sensor, mask in sensor_issue_masks.items()
                if bool(mask.loc[index])
            )
            for index in sensor_issue_rows.index
        ]

    return result, sensor_quality_summary, sensor_issue_rows


# Controller-quality reports (no extra master-table columns)
def build_controller_quality_reports(
    data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Validate controller fields without using the checks as predictors."""
    rotation_issue = (
        (
            data["operating_mode"].eq("NRML")
            & (
                data["rotation_minutes_remaining"].isna()
                | ~data["rotation_minutes_remaining"].between(0, 299)
            )
        )
        | (
            data["operating_mode"].eq("HTCH")
            & data["rotation_minutes_remaining"].notna()
        )
    )

    checks = pd.DataFrame(
        {
            "avg_t": (
                data["avg_t"].isna()
                | ~data["avg_t"].between(0, 60)
            ),
            "avg_h": (
                data["avg_h"].isna()
                | ~data["avg_h"].between(0, 100)
            ),
            "duty_pct": (
                data["duty_pct"].isna()
                | ~data["duty_pct"].between(0, 100)
            ),
            "duty_correction_pct": (
                data["duty_correction_pct"].isna()
                | ~data["duty_correction_pct"].between(-25, 25)
            ),
            "heater_on": ~data["heater_on"].isin([0, 1]),
            "humidity_fan_on": ~data["humidity_fan_on"].isin([0, 1]),
            "servo_position": ~data["servo_position"].isin(
                ["MIN", "MAX", "STOPPED"]
            ),
            "operating_mode": ~data["operating_mode"].isin(
                ["NRML", "HTCH"]
            ),
            "sensor_spread_c": (
                data["sensor_spread_c"].isna()
                | ~data["sensor_spread_c"].between(0, 60)
            ),
            "active_sensors_count": (
                data["active_sensors_count"].isna()
                | ~data["active_sensors_count"].between(1, 4)
            ),
            "rotation_state": rotation_issue,
            "low_edge_counter": (
                data["low_edge_counter"].isna()
                | ~data["low_edge_counter"].between(0, 900)
            ),
            "high_edge_counter": (
                data["high_edge_counter"].isna()
                | ~data["high_edge_counter"].between(0, 900)
            ),
        },
        index=data.index,
    )

    controller_quality_summary = pd.DataFrame(
        {"issue_count": checks.sum().astype("int64")}
    )
    controller_quality_summary.index.name = "check"

    issue_mask = checks.any(axis=1)
    checked_value_columns = [
        "record_id",
        "timestamp",
        "avg_t",
        "avg_h",
        "duty_pct",
        "duty_correction_pct",
        "heater_on",
        "humidity_fan_on",
        "servo_position",
        "operating_mode",
        "sensor_spread_c",
        "active_sensors_count",
        "rotation_minutes_remaining",
        "low_edge_counter",
        "high_edge_counter",
    ]
    controller_issue_rows = data.loc[
        issue_mask, checked_value_columns
    ].copy()

    if not controller_issue_rows.empty:
        controller_issue_rows["failed_checks"] = checks.loc[
            issue_mask
        ].apply(
            lambda row: ",".join(row.index[row].tolist()),
            axis=1,
        )

    return controller_quality_summary, controller_issue_rows


# Historical features using every reading in each past interval
def add_historical_features(data: pd.DataFrame) -> pd.DataFrame:
    """Add full-window means for the primary signals and fan state."""
    result = data.copy()

    for column in HISTORY_FEATURES:
        result[column] = np.nan

    for _, group in result.groupby("sequence_id", sort=False):
        positions = group.index
        history_source = group[
            ["estimated_timestamp"] + HISTORY_SOURCE_FEATURES
        ].copy()
        history_source = history_source.set_index("estimated_timestamp")

        for sensor in ("s2", "s3"):
            sensor_valid = (
                group[f"{sensor}_t"].between(
                    TEMP_PLAUSIBLE_MIN_C,
                    TEMP_PLAUSIBLE_MAX_C,
                )
                & group[f"{sensor}_h"].between(
                    HUMIDITY_PLAUSIBLE_MIN_PCT,
                    HUMIDITY_PLAUSIBLE_MAX_PCT,
                )
                & group[f"{sensor}_flag"].eq("A")
            ).to_numpy()
            history_source.loc[
                ~sensor_valid, f"{sensor}_t"
            ] = np.nan

        history_source.loc[
            ~history_source["duty_pct"].between(0, 100),
            "duty_pct",
        ] = np.nan
        history_source.loc[
            ~history_source["humidity_fan_on"].isin([0, 1]),
            "humidity_fan_on",
        ] = np.nan

        elapsed_minutes = (
            history_source.index - history_source.index[0]
        ).total_seconds() / 60

        for minutes in HISTORY_WINDOWS_MINUTES:
            rolling_mean = history_source.rolling(
                f"{minutes}min",
                closed="both",
                min_periods=1,
            ).mean()

            full_window = elapsed_minutes >= minutes

            for source in HISTORY_SOURCE_FEATURES:
                output_column = history_feature_name(source, minutes)
                values = np.where(
                    full_window,
                    rolling_mean[source].to_numpy(),
                    np.nan,
                )
                result.loc[positions, output_column] = values

    return result


# Exact-horizon future temperature targets
def add_future_targets(
    data: pd.DataFrame,
    max_match_error_seconds: int = MAX_TARGET_MATCH_ERROR_SECONDS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Match each row to the nearest valid future state in its sequence."""
    result = data.copy()
    target_arrays = {
        minutes: np.full(len(result), np.nan, dtype="float64")
        for minutes in TARGET_HORIZONS_MINUTES
    }
    error_arrays = {
        minutes: np.full(len(result), np.nan, dtype="float64")
        for minutes in TARGET_HORIZONS_MINUTES
    }

    tolerance_ns = int(max_match_error_seconds * 1_000_000_000)

    for _, group in result.groupby("sequence_id", sort=False):
        positions = group.index.to_numpy()
        current_times_ns = (
            group["estimated_timestamp"].astype("int64").to_numpy()
        )

        valid_target_source = (
            group["avg_t"].between(
                TEMP_PLAUSIBLE_MIN_C,
                TEMP_PLAUSIBLE_MAX_C,
            )
            & group["sensor_quality_issue_count"].eq(0)
        ).to_numpy()

        candidate_times_ns = current_times_ns[valid_target_source]
        candidate_values = group.loc[
            valid_target_source, "avg_t"
        ].to_numpy(dtype="float64")

        if candidate_times_ns.size == 0:
            continue

        for minutes in TARGET_HORIZONS_MINUTES:
            desired_times_ns = (
                current_times_ns
                + int(minutes * 60 * 1_000_000_000)
            )

            insertion = np.searchsorted(
                candidate_times_ns,
                desired_times_ns,
                side="left",
            )
            right = np.clip(
                insertion,
                0,
                candidate_times_ns.size - 1,
            )
            left = np.clip(
                insertion - 1,
                0,
                candidate_times_ns.size - 1,
            )

            left_error = np.abs(
                candidate_times_ns[left] - desired_times_ns
            )
            right_error = np.abs(
                candidate_times_ns[right] - desired_times_ns
            )
            use_right = right_error < left_error
            nearest = np.where(use_right, right, left)
            nearest_error = np.minimum(left_error, right_error)
            accepted = nearest_error <= tolerance_ns

            target_arrays[minutes][positions[accepted]] = (
                candidate_values[nearest[accepted]]
            )
            error_arrays[minutes][positions[accepted]] = (
                nearest_error[accepted] / 1_000_000_000
            )

    target_summary_rows = []

    for minutes in TARGET_HORIZONS_MINUTES:
        target_column = f"target_avg_t_{minutes}m"
        result[target_column] = target_arrays[minutes]
        available = np.isfinite(target_arrays[minutes])
        accepted_errors = error_arrays[minutes][available]

        target_summary_rows.append(
            {
                "target": target_column,
                "horizon_minutes": minutes,
                "available_rows": int(available.sum()),
                "unavailable_rows": int((~available).sum()),
                "availability_pct": float(available.mean() * 100),
                "median_alignment_error_seconds": (
                    float(np.median(accepted_errors))
                    if accepted_errors.size
                    else np.nan
                ),
                "max_alignment_error_seconds": (
                    float(np.max(accepted_errors))
                    if accepted_errors.size
                    else np.nan
                ),
            }
        )

    target_summary = pd.DataFrame(target_summary_rows).set_index(
        "target"
    )

    return result, target_summary


# Feature-selection report
def build_feature_manifest() -> pd.DataFrame:
    """Describe which columns belong to Version 1 and which are excluded."""
    rows = []

    for feature in PRIMARY_CURRENT_FEATURES:
        rows.append(
            {
                "column": feature,
                "role": "primary_current",
                "used_in_version1": True,
                "reason": "active temperature or commanded heater duty",
            }
        )

    for feature in SUPPORTING_CURRENT_FEATURES:
        rows.append(
            {
                "column": feature,
                "role": "supporting_current",
                "used_in_version1": True,
                "reason": "additional current system state",
            }
        )

    for feature in HISTORY_FEATURES:
        rows.append(
            {
                "column": feature,
                "role": "historical",
                "used_in_version1": True,
                "reason": "all valid readings in the stated past window",
            }
        )

    excluded_reasons = {
        "heater_on": "logging is too slow for short PWM pulses",
        "low_edge_counter": "controller implementation detail",
        "high_edge_counter": "controller implementation detail",
        "rotation_minutes_remaining": "not a thermal predictor for Version 1",
        "active_sensors_count": "not selected for Version 1",
        "servo_position": "not selected for Version 1",
    }

    for column in EXCLUDED_MODEL_COLUMNS:
        rows.append(
            {
                "column": column,
                "role": "excluded",
                "used_in_version1": False,
                "reason": excluded_reasons[column],
            }
        )

    return pd.DataFrame(rows)


# Complete Stage 2 construction
def build_stage2_master(
    input_file: Path = INPUT_FILE,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Build the master table and return all callable reports."""
    master_df = load_stage1_data(input_file)
    rows_before = len(master_df)

    master_df, sequence_summary = add_time_structure(master_df)
    (
        master_df,
        sensor_quality_summary,
        sensor_issue_rows,
    ) = add_sensor_quality(master_df)

    (
        controller_quality_summary,
        controller_issue_rows,
    ) = build_controller_quality_reports(master_df)

    master_df = add_historical_features(master_df)
    master_df, target_summary = add_future_targets(master_df)
    feature_manifest = build_feature_manifest()

    reports = {
        "sequence_summary": sequence_summary,
        "sensor_quality_summary": sensor_quality_summary,
        "sensor_issue_rows": sensor_issue_rows,
        "controller_quality_summary": controller_quality_summary,
        "controller_issue_rows": controller_issue_rows,
        "feature_manifest": feature_manifest,
        "target_summary": target_summary,
    }

    validate_stage2(master_df, reports, rows_before)
    return master_df, reports


def build_ml_table(master_df: pd.DataFrame) -> pd.DataFrame:
    """Select only the columns needed for Stage 3 and ML experiments."""
    missing_columns = set(ML_TABLE_COLUMNS).difference(master_df.columns)
    if missing_columns:
        raise ValueError(
            "Cannot build the ML table; missing columns: "
            f"{sorted(missing_columns)}"
        )

    ml_df = master_df[ML_TABLE_COLUMNS].copy()

    if len(ml_df) != len(master_df):
        raise ValueError("The compact ML table changed the row count.")

    if ml_df.columns.duplicated().any():
        raise ValueError("The compact ML table contains duplicate columns.")

    return ml_df


def validate_stage2(
    master_df: pd.DataFrame,
    reports: dict[str, pd.DataFrame],
    rows_before: int,
) -> None:
    """Check the invariants required before saving Stage 2."""
    if len(master_df) != rows_before:
        raise ValueError("Stage 2 changed the number of records.")

    if not master_df["record_id"].is_monotonic_increasing:
        raise ValueError("Stage 2 changed acquisition order.")

    if master_df["record_id"].duplicated().any():
        raise ValueError("Stage 2 created duplicate record_id values.")

    required_columns = set(VERSION1_FEATURES + TARGET_COLUMNS)
    missing_columns = required_columns.difference(master_df.columns)
    if missing_columns:
        raise ValueError(
            f"Stage 2 is missing required columns: {sorted(missing_columns)}"
        )

    wrongly_selected = set(EXCLUDED_MODEL_COLUMNS).intersection(
        VERSION1_FEATURES
    )
    if wrongly_selected:
        raise ValueError(
            f"Excluded columns entered Version 1: {sorted(wrongly_selected)}"
        )

    for target in TARGET_COLUMNS:
        valid_values = master_df[target].dropna()
        if not valid_values.between(
            TEMP_PLAUSIBLE_MIN_C,
            TEMP_PLAUSIBLE_MAX_C,
        ).all():
            raise ValueError(f"{target} contains an implausible value.")

    if reports["target_summary"]["available_rows"].eq(0).any():
        raise ValueError("At least one future target has no valid rows.")


# Save final outputs
def save_stage2_outputs(
    master_df: pd.DataFrame,
    ml_df: pd.DataFrame,
    reports: dict[str, pd.DataFrame],
    master_output_file: Path = MASTER_OUTPUT_FILE,
    ml_output_file: Path = ML_OUTPUT_FILE,
    reports_output_dir: Path = REPORTS_OUTPUT_DIR,
) -> dict[str, Path]:
    """Save both data tables and all reports, replacing existing files."""
    master_output_file.parent.mkdir(parents=True, exist_ok=True)
    ml_output_file.parent.mkdir(parents=True, exist_ok=True)
    reports_output_dir.mkdir(parents=True, exist_ok=True)

    master_df.to_csv(master_output_file, index=False)
    ml_df.to_csv(ml_output_file, index=False)

    report_filenames = {
        "sequence_summary": "sequence_summary.csv",
        "sensor_quality_summary": "sensor_quality_summary.csv",
        "sensor_issue_rows": "sensor_issue_rows.csv",
        "controller_quality_summary": "controller_quality_summary.csv",
        "controller_issue_rows": "controller_issue_rows.csv",
        "feature_manifest": "version1_feature_manifest.csv",
        "target_summary": "target_summary.csv",
    }

    saved_paths = {
        "master": master_output_file,
        "ml_table": ml_output_file,
    }

    for report_name, filename in report_filenames.items():
        output_path = reports_output_dir / filename
        save_index = report_name in {
            "sensor_quality_summary",
            "controller_quality_summary",
            "target_summary",
        }
        reports[report_name].to_csv(output_path, index=save_index)
        saved_paths[report_name] = output_path

    return saved_paths


def main() -> None:
    """Run Stage 2 from the project root."""
    master_df, reports = build_stage2_master(INPUT_FILE)
    ml_df = build_ml_table(master_df)
    saved_paths = save_stage2_outputs(master_df, ml_df, reports)

    print("Stage 1 input shape:", (EXPECTED_STAGE1_ROWS, 27))
    print("Stage 2 master shape:", master_df.shape)
    print("Stage 3 ML table shape:", ml_df.shape)
    print("Number of sequences:", master_df["sequence_id"].nunique())
    print(
        "Rows with unexpected sensor issues:",
        len(reports["sensor_issue_rows"]),
    )
    print(
        "Rows with controller issues:",
        len(reports["controller_issue_rows"]),
    )
    print("\nTarget coverage:")
    print(reports["target_summary"].to_string())
    print("\nAvailable reports:", list(reports))
    print("\nSaved outputs:")
    for name, path in saved_paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
