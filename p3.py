"""Stage 3, Part 1: create future thermal-risk targets.

The script reads the compact Stage 2 table and creates three targets:
temperature_risk_5m, temperature_risk_10m, and temperature_risk_15m.
"""

from pathlib import Path

import numpy as np
import pandas as pd


INPUT_FILE = Path("data/processed/incubator_stage2_ml.csv")
OUTPUT_FILE = Path("data/processed/incubator_stage3_targets.csv")
TARGET_REPORT_FILE = Path("reports/stage3/temperature_risk_summary.csv")

HORIZONS_MINUTES = (5, 10, 15)

MODE_LIMITS = {
    "NRML": (37.3, 37.7),
    "HTCH": (37.1, 37.5),
}

REQUIRED_COLUMNS = [
    "record_id",
    "sequence_id",
    "estimated_timestamp",
    "sensor_quality_issue_count",
    "avg_t",
    "operating_mode",
]


def load_stage2_data(input_file=INPUT_FILE):
    """Load the Stage 2 ML table and check the required columns."""
    data = pd.read_csv(input_file, parse_dates=["estimated_timestamp"])

    missing_columns = set(REQUIRED_COLUMNS) - set(data.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    if not data["operating_mode"].isin(MODE_LIMITS).all():
        raise ValueError("Unexpected operating mode found.")

    return data


def classify_window(future_temperatures, mode):
    """Classify all valid temperatures inside one future window."""
    low_limit, high_limit = MODE_LIMITS[mode]
    future_min = future_temperatures.min()
    future_max = future_temperatures.max()

    crossed_low = future_min < low_limit
    crossed_high = future_max > high_limit

    if crossed_low and crossed_high:
        return np.nan, True
    if crossed_low:
        return "LOW", False
    if crossed_high:
        return "HIGH", False
    return "NORMAL", False

def calculate_temperature_slope(times, temperatures):
    """Calculate the overall temperature slope in °C/minute."""

    elapsed_minutes = (
        times - times[0]
    ) / (60 * 1_000_000_000)

    slope = np.polyfit(
        elapsed_minutes,
        temperatures,
        1,
    )[0]

    return slope

def add_temperature_slopes(data):
    """Add temperature slopes over the previous 5, 10, and 15 minutes."""

    result = data.copy()

    for minutes in HORIZONS_MINUTES:
        result[f"avg_t_slope_last_{minutes}m"] = np.nan

    for _, group in result.groupby("sequence_id", sort=False):
        positions = group.index.to_numpy()

        times = (
            group["estimated_timestamp"]
            .to_numpy(dtype="datetime64[ns]")
            .astype("int64")
        )

        temperatures = group["avg_t"].to_numpy(dtype="float64")

        quality_ok = (
            group["sensor_quality_issue_count"]
            .eq(0)
            .to_numpy()
        )

        for local_row, position in enumerate(positions):
            for minutes in HORIZONS_MINUTES:
                window_start = (
                    times[local_row]
                    - minutes * 60 * 1_000_000_000
                )

                # Require a complete historical window.
                if window_start < times[0]:
                    continue

                start_row = np.searchsorted(
                    times,
                    window_start,
                    side="left",
                )

                window_slice = slice(start_row, local_row + 1)

                valid = (
                    quality_ok[window_slice]
                    & np.isfinite(temperatures[window_slice])
                )

                window_times = times[window_slice][valid]
                window_temperatures = temperatures[window_slice][valid]

                if window_temperatures.size < 2:
                    continue

                slope = calculate_temperature_slope(
                    window_times,
                    window_temperatures,
                )

                result.at[
                    position,
                    f"avg_t_slope_last_{minutes}m",
                ] = slope

    return result

def add_risk_targets(data):
    """Create 5-, 10-, and 15-minute future-window risk targets."""
    result = data.copy()
    target_columns = [
        f"temperature_risk_{minutes}m"
        for minutes in HORIZONS_MINUTES
    ]

    for column in target_columns:
        result[column] = pd.NA

    ambiguous_counts = {minutes: 0 for minutes in HORIZONS_MINUTES}

    # A new segment prevents a target from crossing a sequence or mode change.
    segment_start = (
        result["sequence_id"].ne(result["sequence_id"].shift())
        | result["operating_mode"].ne(result["operating_mode"].shift())
    )
    result["target_segment_id"] = segment_start.cumsum()

    for _, group in result.groupby("target_segment_id", sort=False):
        positions = group.index.to_numpy()
        times = (group["estimated_timestamp"]
            .to_numpy(dtype="datetime64[ns]")
            .astype("int64"))
        temperatures = group["avg_t"].to_numpy(dtype="float64")
        quality_ok = group["sensor_quality_issue_count"].eq(0).to_numpy()
        mode = group["operating_mode"].iloc[0]

        for local_row, position in enumerate(positions):
            for minutes in HORIZONS_MINUTES:
                window_end = times[local_row] + minutes * 60 * 1_000_000_000

                # A full future window must exist inside the same segment.
                if window_end > times[-1]:
                    continue

                end_row = np.searchsorted(times, window_end, side="right")
                window_slice = slice(local_row + 1, end_row)
                valid = quality_ok[window_slice] & np.isfinite(
                    temperatures[window_slice]
                )
                future_values = temperatures[window_slice][valid]

                if future_values.size == 0:
                    continue

                risk, ambiguous = classify_window(future_values, mode)
                if ambiguous:
                    ambiguous_counts[minutes] += 1
                    continue

                result.at[
                    position,
                    f"temperature_risk_{minutes}m",
                ] = risk

    return result, ambiguous_counts

def create_target_summary(data, ambiguous_counts):
    """Create class counts and percentages for every target."""

    summary_rows = []

    for minutes in HORIZONS_MINUTES:
        column = f"temperature_risk_{minutes}m"

        counts = data[column].value_counts()
        available_rows = data[column].notna().sum()
        missing_rows = data[column].isna().sum()

        summary_rows.append({
            "target": column,
            "horizon_minutes": minutes,
            "low_count": counts.get("LOW", 0),
            "normal_count": counts.get("NORMAL", 0),
            "high_count": counts.get("HIGH", 0),
            "missing_count": missing_rows,
            "ambiguous_low_high_count": ambiguous_counts[minutes],
            "available_rows": available_rows,
            "low_pct": counts.get("LOW", 0) / available_rows * 100,
            "normal_pct": counts.get("NORMAL", 0) / available_rows * 100,
            "high_pct": counts.get("HIGH", 0) / available_rows * 100,
        })

    return pd.DataFrame(summary_rows)

def main(input_file=INPUT_FILE, output_file=OUTPUT_FILE):
    data = load_stage2_data(input_file)

    data_with_slopes = add_temperature_slopes(data)

    data_with_targets, ambiguous_counts = add_risk_targets(
        data_with_slopes
    )

    slope_columns = [
        f"avg_t_slope_last_{minutes}m"
        for minutes in HORIZONS_MINUTES
    ]

    print("\nTemperature-slope availability:")
    print(
        data_with_targets[slope_columns]
        .notna()
        .sum()
        .to_string()
    )

    target_summary = create_target_summary(
        data_with_targets,
        ambiguous_counts,
    )

    print("\nTemperature-risk target summary:")
    print(target_summary.to_string(index=False))

    TARGET_REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
    target_summary.to_csv(TARGET_REPORT_FILE, index=False)

    print(f"\nSaved report: {TARGET_REPORT_FILE}")

    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    data_with_targets.to_csv(output_file, index=False)
    print(f"\nSaved: {output_file}")


if __name__ == "__main__":
    main()
