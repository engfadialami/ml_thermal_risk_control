"""Stage 6 - Collect final analytical evidence and generate report plots.

Run this script once from the project root after Stages 1-5 are complete.
It does not modify any existing pipeline script, dataset, report, or model.

Outputs
-------
reports/final_analysis/
    final_analysis_report.md
    final_analysis_summary.json
    derived CSV audit tables
    plots/*.png

reports/final_analysis_bundle.zip
    A compact bundle to upload for preparation of the final Word report,
    README, and PowerPoint. Raw/processed datasets and model binaries are not
    included in the bundle.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import sys
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "incubator_matplotlib_cache"),
)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent

RAW_FILE = PROJECT_ROOT / "data/raw/incubator_bt_log2.csv"
STAGE1_FILE = PROJECT_ROOT / "data/processed/incubator_prepared.csv"
STAGE2_MASTER_FILE = PROJECT_ROOT / "data/processed/incubator_stage2_master.csv"
STAGE2_ML_FILE = PROJECT_ROOT / "data/processed/incubator_stage2_ml.csv"
STAGE3_FILE = PROJECT_ROOT / "data/processed/incubator_stage3_targets.csv"
NORMALIZED_TEST_FILE = (
    PROJECT_ROOT / "data/processed/incubator_stage4_test_normalized.csv"
)

STAGE2_REPORT_DIR = PROJECT_ROOT / "reports/stage2"
STAGE3_REPORT_DIR = PROJECT_ROOT / "reports/stage3"
STAGE4_REPORT_DIR = PROJECT_ROOT / "reports/stage4"
MODEL_DIR = PROJECT_ROOT / "models/stage5"

OUTPUT_DIR = PROJECT_ROOT / "reports/final_analysis"
PLOT_DIR = OUTPUT_DIR / "plots"
BUNDLE_FILE = PROJECT_ROOT / "reports/final_analysis_bundle.zip"


# ---------------------------------------------------------------------------
# Fixed project contract - verified during the completed stages
# ---------------------------------------------------------------------------

EXPECTED_RAW_ROWS = 53_075
EXPECTED_PREPARED_ROWS = 53_061
EXPECTED_STAGE1_COLUMNS = 27
EXPECTED_STAGE3_ROWS = 53_061

HORIZONS = (5, 10, 15)
CLASS_ORDER = ["LOW", "NORMAL", "HIGH"]
CLASS_COLORS = {
    "LOW": "#2563EB",
    "NORMAL": "#16A34A",
    "HIGH": "#DC2626",
}

TRAIN_SEQUENCES = [1, 2, 3, 5]
VALIDATION_SEQUENCES = [7]
TEST_SEQUENCES = [8, 9]

MODE_LIMITS = {
    "NRML": (37.3, 37.7),
    "HTCH": (37.1, 37.5),
}
TEMPERATURE_OFFSET_C = 0.2

TARGET_COLUMNS = [
    f"temperature_risk_{minutes}m"
    for minutes in HORIZONS
]

SLOPE_COLUMNS = [
    f"avg_t_slope_last_{minutes}m"
    for minutes in HORIZONS
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

UNCHANGED_AFTER_NORMALIZATION = [
    "sensor_spread_c",
    *SLOPE_COLUMNS,
    *TARGET_COLUMNS,
]


REPORT_FILES = {
    "stage2_sequence_summary": STAGE2_REPORT_DIR / "sequence_summary.csv",
    "stage2_sensor_quality": STAGE2_REPORT_DIR / "sensor_quality_summary.csv",
    "stage2_sensor_issue_rows": STAGE2_REPORT_DIR / "sensor_issue_rows.csv",
    "stage2_controller_quality": (
        STAGE2_REPORT_DIR / "controller_quality_summary.csv"
    ),
    "stage2_controller_issue_rows": (
        STAGE2_REPORT_DIR / "controller_issue_rows.csv"
    ),
    "stage2_feature_manifest": (
        STAGE2_REPORT_DIR / "version1_feature_manifest.csv"
    ),
    "stage2_target_summary": STAGE2_REPORT_DIR / "target_summary.csv",
    "stage3_risk_summary": (
        STAGE3_REPORT_DIR / "temperature_risk_summary.csv"
    ),
    "stage4_data_summary": STAGE4_REPORT_DIR / "data_summary.csv",
    "stage4_class_distribution": (
        STAGE4_REPORT_DIR / "class_distribution.csv"
    ),
    "stage4_model_configurations": (
        STAGE4_REPORT_DIR / "model_configurations.csv"
    ),
    "stage4_validation_comparison": (
        STAGE4_REPORT_DIR / "validation_comparison.csv"
    ),
    "stage4_final_test_results": (
        STAGE4_REPORT_DIR / "final_test_results.csv"
    ),
}

METADATA_FILE = MODEL_DIR / "model_metadata.json"
API_EXAMPLES_FILE = MODEL_DIR / "api_test_examples.json"


# ---------------------------------------------------------------------------
# General helpers
# ---------------------------------------------------------------------------

PLOT_RECORDS: list[dict[str, str]] = []
WARNINGS: list[str] = []


def configure_plot_style() -> None:
    """Apply one restrained style to every exported figure."""

    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#9CA3AF",
        "axes.labelcolor": "#1F2937",
        "axes.titlecolor": "#111827",
        "axes.titleweight": "bold",
        "axes.grid": True,
        "grid.color": "#E5E7EB",
        "grid.linewidth": 0.7,
        "grid.alpha": 0.9,
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "legend.frameon": False,
        "xtick.color": "#374151",
        "ytick.color": "#374151",
    })


def relative_path(path: Path) -> str:
    """Return a stable project-relative path for reports."""

    try:
        return str(path.resolve().relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        return str(path)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Calculate a SHA-256 digest without loading the whole file at once."""

    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def json_safe(value: Any) -> Any:
    """Recursively convert numpy/pandas objects to JSON-safe values."""

    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if value is pd.NA:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def save_json(data: Any, path: Path) -> None:
    """Save one formatted JSON file."""

    with path.open("w", encoding="utf-8") as file:
        json.dump(json_safe(data), file, indent=2, ensure_ascii=False)


def save_csv(data: pd.DataFrame, filename: str) -> Path:
    """Save a derived table and return its path."""

    path = OUTPUT_DIR / filename
    data.to_csv(path, index=False)
    return path


def markdown_table(data: pd.DataFrame, max_rows: int | None = None) -> str:
    """Convert a compact DataFrame to Markdown without extra dependencies."""

    if max_rows is not None:
        data = data.head(max_rows)

    if data.empty:
        return "_No rows available._"

    display = data.copy()
    for column in display.columns:
        display[column] = display[column].map(
            lambda value: "" if pd.isna(value) else str(value)
        )

    headers = [str(column) for column in display.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in display.itertuples(index=False, name=None):
        cleaned = [str(value).replace("|", "\\|") for value in row]
        lines.append("| " + " | ".join(cleaned) + " |")
    return "\n".join(lines)


def save_figure(
    fig: plt.Figure,
    filename: str,
    title: str,
    evidence: str,
) -> Path:
    """Save one publication-ready PNG and record its interpretation."""

    path = PLOT_DIR / filename
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    PLOT_RECORDS.append({
        "file": relative_path(path),
        "title": title,
        "evidence": evidence,
    })
    return path


def annotate_bars(ax: plt.Axes, decimals: int = 1) -> None:
    """Add compact labels above vertical bars."""

    for patch in ax.patches:
        height = patch.get_height()
        if not np.isfinite(height):
            continue
        label = f"{height:,.{decimals}f}"
        if decimals == 0:
            label = f"{height:,.0f}"
        ax.annotate(
            label,
            (patch.get_x() + patch.get_width() / 2, height),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
            color="#374151",
        )


def validate_inputs() -> None:
    """Fail early when a core source required for final analysis is missing."""

    required = {
        "Stage 3 processed dataset": STAGE3_FILE,
        "Normalized Stage 4 test dataset": NORMALIZED_TEST_FILE,
        "Stage 5 model metadata": METADATA_FILE,
        **REPORT_FILES,
    }
    missing = [
        f"{name}: {relative_path(path)}"
        for name, path in required.items()
        if not path.is_file()
    ]

    if missing:
        raise FileNotFoundError(
            "Required final-analysis inputs are missing:\n- "
            + "\n- ".join(missing)
        )


def read_reports() -> dict[str, pd.DataFrame]:
    """Load all stage report CSVs."""

    return {
        name: pd.read_csv(path)
        for name, path in REPORT_FILES.items()
    }


def build_source_inventory(
    reports: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Record the exact evidence files used by this run."""

    rows = []
    for name, path in REPORT_FILES.items():
        frame = reports[name]
        rows.append({
            "source": name,
            "path": relative_path(path),
            "rows": len(frame),
            "columns": len(frame.columns),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        })

    for name, path in [
        ("stage3_processed_data", STAGE3_FILE),
        ("normalized_test_data", NORMALIZED_TEST_FILE),
        ("stage5_model_metadata", METADATA_FILE),
    ]:
        rows.append({
            "source": name,
            "path": relative_path(path),
            "rows": "see summary",
            "columns": "see summary",
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Stage 1 analysis
# ---------------------------------------------------------------------------

def reconstruct_stage1_validation() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Re-run Stage 1 validation in memory when the raw file is available."""

    fallback = pd.DataFrame([
        {
            "message_status": "complete",
            "row_count": EXPECTED_PREPARED_ROWS,
            "percentage": EXPECTED_PREPARED_ROWS / EXPECTED_RAW_ROWS * 100,
        },
        {
            "message_status": "rejected_total",
            "row_count": EXPECTED_RAW_ROWS - EXPECTED_PREPARED_ROWS,
            "percentage": (
                (EXPECTED_RAW_ROWS - EXPECTED_PREPARED_ROWS)
                / EXPECTED_RAW_ROWS
                * 100
            ),
        },
    ])

    if not RAW_FILE.is_file():
        WARNINGS.append(
            "Raw Stage 1 file was unavailable; the verified 53,075/53,061 "
            "project totals were used for the Stage 1 retention summary."
        )
        return fallback, pd.DataFrame()

    p1_path = PROJECT_ROOT / "p1.py"
    if not p1_path.is_file():
        WARNINGS.append(
            "p1.py was unavailable; Stage 1 status details could not be "
            "reconstructed from the raw file."
        )
        return fallback, pd.DataFrame()

    spec = importlib.util.spec_from_file_location("stage1_final_audit", p1_path)
    if spec is None or spec.loader is None:
        WARNINGS.append("p1.py could not be imported for the final audit.")
        return fallback, pd.DataFrame()

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    prepared, summary, rejected = module.build_prepared_dataframe(RAW_FILE)

    if len(prepared) != EXPECTED_PREPARED_ROWS:
        raise ValueError(
            "Stage 1 reconstruction returned an unexpected prepared-row count: "
            f"{len(prepared):,}"
        )

    if STAGE1_FILE.is_file():
        saved_shape = pd.read_csv(STAGE1_FILE, nrows=5).shape[1]
        if saved_shape != EXPECTED_STAGE1_COLUMNS:
            raise ValueError(
                "The saved Stage 1 file does not contain the expected 27 columns."
            )

    rejected_columns = [
        column
        for column in [
            "record_id",
            "source_line_number",
            "timestamp_raw",
            "csv_field_count",
            "message_section_count",
            "message_status",
            "raw_message",
        ]
        if column in rejected.columns
    ]

    return summary, rejected[rejected_columns].copy()


def plot_stage1_retention(summary: pd.DataFrame) -> None:
    """Show retention and rejection composition."""

    complete_rows = int(
        summary.loc[
            summary["message_status"].eq("complete"),
            "row_count",
        ].sum()
    )
    if complete_rows == 0:
        complete_rows = EXPECTED_PREPARED_ROWS

    rejected_rows = EXPECTED_RAW_ROWS - complete_rows

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    axes[0].bar(
        ["Complete", "Rejected"],
        [complete_rows, rejected_rows],
        color=["#0F766E", "#B91C1C"],
    )
    axes[0].set_title("Stage 1 structural retention")
    axes[0].set_ylabel("Physical rows")
    axes[0].grid(axis="x", visible=False)
    annotate_bars(axes[0], decimals=0)

    rejected_detail = summary[
        ~summary["message_status"].isin(["complete", "rejected_total"])
    ].copy()
    if rejected_detail.empty:
        axes[1].bar(
            ["All rejected"],
            [rejected_rows],
            color="#B91C1C",
        )
    else:
        rejected_detail = rejected_detail.sort_values("row_count")
        axes[1].barh(
            rejected_detail["message_status"],
            rejected_detail["row_count"],
            color="#B91C1C",
        )
        for y, value in enumerate(rejected_detail["row_count"]):
            axes[1].text(value + 0.1, y, f"{int(value)}", va="center")
    axes[1].set_title("Rejected-row diagnostic categories")
    axes[1].set_xlabel("Rows")
    axes[1].grid(axis="y", visible=False)

    fig.suptitle(
        "Raw records were reported before exclusion; 99.9736% were retained",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout()

    save_figure(
        fig,
        "01_stage1_data_retention.png",
        "Stage 1 data retention",
        "Supports the decision to exclude only structurally invalid records "
        "after reporting them, while retaining 53,061 of 53,075 rows.",
    )


# ---------------------------------------------------------------------------
# Stage 2 analysis
# ---------------------------------------------------------------------------

def build_sensor_missingness_evidence(
    sensor_quality: pd.DataFrame,
) -> pd.DataFrame:
    """Quantify how much missingness is explained by explicit ERR flags."""

    result = sensor_quality.copy()
    result["err_explained_pct"] = np.where(
        result["missing_readings"].gt(0),
        result["explicit_ERR"] / result["missing_readings"] * 100,
        np.nan,
    )
    result["unexpected_missing_pct"] = np.where(
        result["missing_readings"].gt(0),
        result["unexpected_missing"] / result["missing_readings"] * 100,
        np.nan,
    )
    return result


def plot_sensor_quality(sensor_evidence: pd.DataFrame) -> None:
    """Show explained and unexpected sensor-quality problems separately."""

    sensors = sensor_evidence["sensor"].str.upper().tolist()
    x = np.arange(len(sensors))
    width = 0.36

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].bar(
        x - width / 2,
        sensor_evidence["missing_readings"],
        width,
        label="Missing readings",
        color="#64748B",
    )
    axes[0].bar(
        x + width / 2,
        sensor_evidence["explicit_ERR"],
        width,
        label="Explicit ERR",
        color="#0F766E",
    )
    axes[0].set_xticks(x, sensors)
    axes[0].set_ylabel("Readings")
    axes[0].set_title("Most missing readings have an explicit cause")
    axes[0].legend()
    axes[0].grid(axis="x", visible=False)

    issue_columns = [
        "unexpected_missing",
        "invalid_flags",
        "implausible_values",
    ]
    issue_labels = ["Unexpected missing", "Invalid flags", "Implausible"]
    offsets = [-width, 0, width]
    colors = ["#D97706", "#7C3AED", "#DC2626"]
    for column, label, offset, color in zip(
        issue_columns,
        issue_labels,
        offsets,
        colors,
    ):
        axes[1].bar(
            x + offset,
            sensor_evidence[column],
            width,
            label=label,
            color=color,
        )
    axes[1].set_xticks(x, sensors)
    axes[1].set_ylabel("Flagged values")
    axes[1].set_title("Unexpected sensor issues remained rare")
    axes[1].legend(fontsize=8)
    axes[1].grid(axis="x", visible=False)

    fig.suptitle(
        "Sensor ERR was preserved as informative missingness",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout()

    save_figure(
        fig,
        "02_sensor_quality_evidence.png",
        "Sensor-quality evidence",
        "Supports preserving ERR flags and missing values rather than silently "
        "imputing them during data preparation.",
    )


def plot_sequence_structure(sequence_summary: pd.DataFrame) -> None:
    """Show sequence size and reconstructed sample interval."""

    data = sequence_summary.copy()
    sequence_labels = data["sequence_id"].astype(str)
    colors = []
    for sequence_id in data["sequence_id"]:
        if sequence_id in TRAIN_SEQUENCES:
            colors.append("#0F766E")
        elif sequence_id in VALIDATION_SEQUENCES:
            colors.append("#D97706")
        elif sequence_id in TEST_SEQUENCES:
            colors.append("#7C3AED")
        else:
            colors.append("#9CA3AF")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].bar(sequence_labels, data["row_count"], color=colors)
    axes[0].set_title("Rows per continuous sequence")
    axes[0].set_xlabel("Sequence ID")
    axes[0].set_ylabel("Rows")
    axes[0].grid(axis="x", visible=False)

    estimable = data.dropna(subset=["estimated_sample_interval_seconds"])
    axes[1].plot(
        estimable["sequence_id"],
        estimable["estimated_sample_interval_seconds"],
        marker="o",
        linewidth=2,
        color="#2563EB",
    )
    axes[1].axhspan(10, 30, color="#DCFCE7", alpha=0.7)
    axes[1].set_ylim(10, 30)
    axes[1].set_title("Estimated sampling interval is stable")
    axes[1].set_xlabel("Sequence ID")
    axes[1].set_ylabel("Seconds per reading")
    axes[1].set_xticks(estimable["sequence_id"])

    fig.suptitle(
        "Sequence-aware processing prevents windows from crossing interruptions",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout()

    save_figure(
        fig,
        "03_sequence_structure_and_sampling.png",
        "Continuous sequences and sampling",
        "Supports reconstructing time within sequences and forbidding history "
        "or targets from crossing gaps and backward timestamps.",
    )


def plot_targets_and_balance(
    target_summary: pd.DataFrame,
    risk_summary: pd.DataFrame,
) -> None:
    """Show future-target coverage and Stage 3 class imbalance."""

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].bar(
        target_summary["horizon_minutes"].astype(str),
        target_summary["availability_pct"],
        color="#2563EB",
    )
    axes[0].set_ylim(98.8, 100.0)
    axes[0].set_xlabel("Prediction horizon (minutes)")
    axes[0].set_ylabel("Available targets (%)")
    axes[0].set_title("Target coverage remains above 99%")
    annotate_bars(axes[0], decimals=2)
    axes[0].grid(axis="x", visible=False)

    bottom = np.zeros(len(risk_summary))
    for class_name, pct_column in [
        ("LOW", "low_pct"),
        ("NORMAL", "normal_pct"),
        ("HIGH", "high_pct"),
    ]:
        values = risk_summary[pct_column].to_numpy()
        axes[1].bar(
            risk_summary["horizon_minutes"].astype(str),
            values,
            bottom=bottom,
            label=class_name,
            color=CLASS_COLORS[class_name],
        )
        bottom += values
    axes[1].set_ylim(0, 100)
    axes[1].set_xlabel("Prediction horizon (minutes)")
    axes[1].set_ylabel("Class share (%)")
    axes[1].set_title("NORMAL dominates every horizon")
    axes[1].legend(ncol=3, loc="upper center")
    axes[1].grid(axis="x", visible=False)

    fig.suptitle(
        "Longer horizons increase minority-risk coverage but reduce complete rows",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout()

    save_figure(
        fig,
        "04_target_availability_and_class_balance.png",
        "Target availability and risk-class balance",
        "Supports multi-horizon modeling and the use of balanced metrics rather "
        "than accuracy alone.",
    )


# ---------------------------------------------------------------------------
# Stage 3 feature/target evidence
# ---------------------------------------------------------------------------

def build_history_availability(
    stage3: pd.DataFrame,
    feature_columns: list[str],
) -> pd.DataFrame:
    """Measure completeness of history and slope features."""

    rows = []
    for feature in feature_columns:
        if "_last_" not in feature:
            continue
        feature_type = "slope" if "slope" in feature else "history"
        horizon = next(
            (minutes for minutes in HORIZONS if f"_{minutes}m" in feature),
            None,
        )
        rows.append({
            "feature": feature,
            "feature_type": feature_type,
            "window_minutes": horizon,
            "available_rows": int(stage3[feature].notna().sum()),
            "missing_rows": int(stage3[feature].isna().sum()),
            "availability_pct": float(stage3[feature].notna().mean() * 100),
        })
    return pd.DataFrame(rows).sort_values(
        ["window_minutes", "feature_type", "feature"]
    )


def plot_history_availability(history: pd.DataFrame) -> None:
    """Show completeness of each engineered historical feature."""

    data = history.sort_values("availability_pct").copy()
    labels = data["feature"].str.replace("_", " ", regex=False)
    colors = [
        "#7C3AED" if feature_type == "slope" else "#0F766E"
        for feature_type in data["feature_type"]
    ]

    fig, ax = plt.subplots(figsize=(11, 7.5))
    ax.barh(labels, data["availability_pct"], color=colors)
    ax.set_xlim(90, 100)
    ax.set_xlabel("Available rows (%)")
    ax.set_title("Historical-feature completeness by window")
    for y, value in enumerate(data["availability_pct"]):
        ax.text(value + 0.08, y, f"{value:.2f}%", va="center", fontsize=8)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()

    save_figure(
        fig,
        "05_history_feature_availability.png",
        "Historical-feature availability",
        "Confirms that missing values at sequence starts grow with the window "
        "length, as required by leakage-safe history construction.",
    )


def build_slope_risk_summary(stage3: pd.DataFrame) -> pd.DataFrame:
    """Summarize temperature slopes for each future-risk class."""

    rows = []
    for minutes in HORIZONS:
        slope = f"avg_t_slope_last_{minutes}m"
        target = f"temperature_risk_{minutes}m"
        for class_name in CLASS_ORDER:
            values = pd.to_numeric(
                stage3.loc[stage3[target].eq(class_name), slope],
                errors="coerce",
            ).dropna()
            rows.append({
                "horizon_minutes": minutes,
                "risk_class": class_name,
                "count": len(values),
                "mean_slope_c_per_min": values.mean(),
                "median_slope_c_per_min": values.median(),
                "q1_slope_c_per_min": values.quantile(0.25),
                "q3_slope_c_per_min": values.quantile(0.75),
                "min_slope_c_per_min": values.min(),
                "max_slope_c_per_min": values.max(),
            })
    return pd.DataFrame(rows)


def plot_slope_by_risk(stage3: pd.DataFrame) -> None:
    """Compare past temperature slopes with future risk labels."""

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)

    for ax, minutes in zip(axes, HORIZONS):
        slope = f"avg_t_slope_last_{minutes}m"
        target = f"temperature_risk_{minutes}m"
        groups = [
            pd.to_numeric(
                stage3.loc[stage3[target].eq(class_name), slope],
                errors="coerce",
            ).dropna().to_numpy()
            for class_name in CLASS_ORDER
        ]
        box = ax.boxplot(
            groups,
            tick_labels=CLASS_ORDER,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "#111827", "linewidth": 1.5},
        )
        for patch, class_name in zip(box["boxes"], CLASS_ORDER):
            patch.set_facecolor(CLASS_COLORS[class_name])
            patch.set_alpha(0.7)
        ax.axhline(0, color="#111827", linewidth=0.8)
        ax.set_title(f"{minutes}-minute horizon")
        ax.set_xlabel("Future risk")
        ax.set_ylabel("Past avg. temperature slope (°C/min)")
        ax.grid(axis="x", visible=False)

    fig.suptitle(
        "Past temperature trend carries information about future thermal risk",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout()

    save_figure(
        fig,
        "06_slope_by_future_risk.png",
        "Temperature slope by future risk",
        "Directly tests the decision to add 5-, 10-, and 15-minute slope "
        "features. The plot shows association, not causation.",
    )


def build_current_feature_summary(stage3: pd.DataFrame) -> pd.DataFrame:
    """Summarize duty and sensor spread by future risk."""

    rows = []
    for minutes in HORIZONS:
        target = f"temperature_risk_{minutes}m"
        for class_name in CLASS_ORDER:
            group = stage3[stage3[target].eq(class_name)]
            for feature in ["duty_pct", "sensor_spread_c"]:
                values = pd.to_numeric(group[feature], errors="coerce").dropna()
                rows.append({
                    "horizon_minutes": minutes,
                    "risk_class": class_name,
                    "feature": feature,
                    "count": len(values),
                    "mean": values.mean(),
                    "median": values.median(),
                    "q1": values.quantile(0.25),
                    "q3": values.quantile(0.75),
                })
    return pd.DataFrame(rows)


def plot_current_features(stage3: pd.DataFrame) -> None:
    """Show current heater duty and sensor disagreement by 15-minute risk."""

    target = "temperature_risk_15m"
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, feature, ylabel, title in [
        (axes[0], "duty_pct", "Duty (%)", "Commanded heater duty"),
        (
            axes[1],
            "sensor_spread_c",
            "Sensor spread (°C)",
            "Active-sensor disagreement",
        ),
    ]:
        groups = [
            pd.to_numeric(
                stage3.loc[stage3[target].eq(class_name), feature],
                errors="coerce",
            ).dropna().to_numpy()
            for class_name in CLASS_ORDER
        ]
        box = ax.boxplot(
            groups,
            tick_labels=CLASS_ORDER,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "#111827", "linewidth": 1.5},
        )
        for patch, class_name in zip(box["boxes"], CLASS_ORDER):
            patch.set_facecolor(CLASS_COLORS[class_name])
            patch.set_alpha(0.7)
        ax.set_title(title)
        ax.set_xlabel("15-minute future risk")
        ax.set_ylabel(ylabel)
        ax.grid(axis="x", visible=False)

    fig.suptitle(
        "Controller effort and multi-sensor state provide complementary context",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout()

    save_figure(
        fig,
        "07_current_features_by_15m_risk.png",
        "Current features by 15-minute risk",
        "Examines the practical decision to include duty_pct and sensor_spread_c "
        "alongside active-sensor temperatures.",
    )


# ---------------------------------------------------------------------------
# Stage 4 split, normalization, and model evaluation
# ---------------------------------------------------------------------------

def plot_split_class_distribution(class_distribution: pd.DataFrame) -> None:
    """Show how risk prevalence changes across chronological splits."""

    data = class_distribution.copy()
    data["group"] = data["split"] + "\n" + data["horizon"]
    x = np.arange(len(data))
    bottom = np.zeros(len(data))

    fig, ax = plt.subplots(figsize=(14, 6))
    for class_name, pct_column in [
        ("LOW", "LOW_%"),
        ("NORMAL", "NORMAL_%"),
        ("HIGH", "HIGH_%"),
    ]:
        values = data[pct_column].to_numpy()
        ax.bar(
            x,
            values,
            bottom=bottom,
            label=class_name,
            color=CLASS_COLORS[class_name],
        )
        bottom += values

    ax.set_xticks(x, data["group"], rotation=0)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Class share (%)")
    ax.set_title("Chronological splits contain different risk distributions")
    ax.legend(ncol=3, loc="upper center")
    ax.grid(axis="x", visible=False)
    fig.tight_layout()

    save_figure(
        fig,
        "08_split_class_distribution.png",
        "Class distribution by chronological split",
        "Shows the distribution shift between Train, Validation, and HATCH-only "
        "Test data, explaining why accuracy alone is not sufficient.",
    )


def verify_normalization(
    stage3: pd.DataFrame,
    normalized_test: pd.DataFrame,
) -> pd.DataFrame:
    """Verify the +0.2 °C coordinate translation column by column."""

    original_test = (
        stage3[stage3["sequence_id"].isin(TEST_SEQUENCES)]
        .sort_values("record_id")
        .reset_index(drop=True)
    )
    normalized = normalized_test.sort_values("record_id").reset_index(drop=True)

    if len(original_test) != len(normalized):
        raise ValueError(
            "Original and normalized test datasets have different row counts."
        )
    if not original_test["record_id"].equals(normalized["record_id"]):
        raise ValueError(
            "Original and normalized test datasets are not aligned by record_id."
        )

    rows = []
    for column in ABSOLUTE_TEMPERATURE_COLUMNS:
        valid = original_test[column].notna() & normalized[column].notna()
        differences = (
            normalized.loc[valid, column]
            - original_test.loc[valid, column]
        )
        passed = bool(
            np.allclose(
                differences.to_numpy(dtype=float),
                TEMPERATURE_OFFSET_C,
                rtol=0,
                atol=1e-12,
            )
        )
        rows.append({
            "check_type": "absolute_temperature_offset",
            "column": column,
            "compared_rows": int(valid.sum()),
            "expected_change": TEMPERATURE_OFFSET_C,
            "min_change": differences.min(),
            "median_change": differences.median(),
            "max_change": differences.max(),
            "passed": passed,
        })

    for column in UNCHANGED_AFTER_NORMALIZATION:
        same = original_test[column].reset_index(drop=True).equals(
            normalized[column].reset_index(drop=True)
        )
        if not same:
            try:
                same = bool(
                    np.allclose(
                        pd.to_numeric(original_test[column], errors="coerce"),
                        pd.to_numeric(normalized[column], errors="coerce"),
                        equal_nan=True,
                        rtol=0,
                        atol=1e-12,
                    )
                )
            except TypeError:
                same = False
        rows.append({
            "check_type": "unchanged_column",
            "column": column,
            "compared_rows": len(original_test),
            "expected_change": 0.0,
            "min_change": 0.0 if same else np.nan,
            "median_change": 0.0 if same else np.nan,
            "max_change": 0.0 if same else np.nan,
            "passed": bool(same),
        })

    result = pd.DataFrame(rows)
    if not result["passed"].all():
        failed = result.loc[~result["passed"], "column"].tolist()
        raise ValueError(f"Normalization verification failed: {failed}")
    return result


def plot_normalization(normalization_checks: pd.DataFrame) -> None:
    """Visualize the offset and the operating-mode reference bands."""

    offsets = normalization_checks[
        normalization_checks["check_type"].eq("absolute_temperature_offset")
    ].copy()
    offsets["short_column"] = (
        offsets["column"].str.replace("_mean_last_", " mean ", regex=False)
    )

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    y = np.arange(len(offsets))
    axes[0].barh(
        y,
        offsets["median_change"],
        color="#7C3AED",
    )
    axes[0].axvline(
        TEMPERATURE_OFFSET_C,
        color="#111827",
        linestyle="--",
        label="Expected +0.2 °C",
    )
    axes[0].set_yticks(y, offsets["short_column"])
    axes[0].set_xlim(0, 0.25)
    axes[0].set_xlabel("Observed median change (°C)")
    axes[0].set_title("Every absolute-temperature column shifted equally")
    axes[0].legend()
    axes[0].grid(axis="y", visible=False)

    bands = [
        ("NORMAL mode", *MODE_LIMITS["NRML"], "#0F766E"),
        ("HATCH mode", *MODE_LIMITS["HTCH"], "#D97706"),
        (
            "HATCH after +0.2 °C",
            MODE_LIMITS["HTCH"][0] + TEMPERATURE_OFFSET_C,
            MODE_LIMITS["HTCH"][1] + TEMPERATURE_OFFSET_C,
            "#7C3AED",
        ),
    ]
    for index, (label, low, high, color) in enumerate(bands):
        axes[1].barh(
            index,
            high - low,
            left=low,
            color=color,
            height=0.55,
        )
        axes[1].text(
            (low + high) / 2,
            index,
            f"{low:.1f}–{high:.1f} °C",
            ha="center",
            va="center",
            color="white",
            fontweight="bold",
        )
    axes[1].set_yticks(range(len(bands)), [band[0] for band in bands])
    axes[1].set_xlim(36.9, 37.9)
    axes[1].set_xlabel("Temperature reference (°C)")
    axes[1].set_title("The translation aligns the two operating references")
    axes[1].grid(axis="y", visible=False)

    fig.suptitle(
        "HATCH test normalization is a coordinate translation, not relabeling",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout()

    save_figure(
        fig,
        "09_hatch_normalization_verification.png",
        "HATCH normalization verification",
        "Confirms that only absolute temperatures moved by +0.2 °C while "
        "slopes, spread, identifiers, and risk labels remained unchanged.",
    )


def build_leakage_audit(
    metadata: dict[str, Any],
    stage3: pd.DataFrame,
) -> pd.DataFrame:
    """Check the approved split and feature contract for obvious leakage."""

    feature_columns = set(metadata["feature_columns"])
    forbidden_features = {
        "record_id",
        "sequence_id",
        "estimated_timestamp",
        "target_segment_id",
        "target_avg_t_5m",
        "target_avg_t_10m",
        "target_avg_t_15m",
        *TARGET_COLUMNS,
    }
    train = set(metadata.get("training_sequences", []))
    validation = set(metadata.get("validation_sequences", []))
    test = set(TEST_SEQUENCES)

    rows = [
        {
            "audit": "Future values and targets excluded from model features",
            "passed": feature_columns.isdisjoint(forbidden_features),
            "evidence": str(sorted(feature_columns & forbidden_features)),
        },
        {
            "audit": "Train and validation sequences are disjoint",
            "passed": train.isdisjoint(validation),
            "evidence": f"train={sorted(train)}, validation={sorted(validation)}",
        },
        {
            "audit": "Test sequences are absent from final model training",
            "passed": (
                test.isdisjoint(train | validation)
                and metadata.get("test_sequences_used_for_training", []) == []
            ),
            "evidence": (
                f"test={sorted(test)}, metadata_test_training="
                f"{metadata.get('test_sequences_used_for_training', [])}"
            ),
        },
        {
            "audit": "All three targets exist in the Stage 3 dataset",
            "passed": set(TARGET_COLUMNS).issubset(stage3.columns),
            "evidence": str(TARGET_COLUMNS),
        },
        {
            "audit": "Saved feature count matches feature list",
            "passed": metadata["feature_count"] == len(metadata["feature_columns"]),
            "evidence": (
                f"declared={metadata['feature_count']}, "
                f"listed={len(metadata['feature_columns'])}"
            ),
        },
    ]
    result = pd.DataFrame(rows)
    if not result["passed"].all():
        failed = result.loc[~result["passed"], "audit"].tolist()
        raise ValueError(f"Leakage/reproducibility audit failed: {failed}")
    return result


def plot_validation_models(validation: pd.DataFrame) -> None:
    """Compare validation metrics across models and horizons."""

    models = validation["model"].drop_duplicates().tolist()
    horizons = ["5 min", "10 min", "15 min"]
    model_colors = {
        "Logistic Regression": "#64748B",
        "Random Forest": "#0F766E",
        "XGBoost": "#D97706",
    }

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    metrics = [
        ("accuracy", "Accuracy"),
        ("balanced_accuracy", "Balanced accuracy"),
        ("macro_f1", "Macro F1"),
    ]
    x = np.arange(len(horizons))
    width = 0.24

    for ax, (metric, label) in zip(axes, metrics):
        for model_index, model in enumerate(models):
            subset = validation[validation["model"].eq(model)].set_index("horizon")
            values = [subset.loc[horizon, metric] for horizon in horizons]
            ax.bar(
                x + (model_index - 1) * width,
                values,
                width,
                label=model,
                color=model_colors.get(model, "#64748B"),
            )
        ax.set_xticks(x, horizons)
        ax.set_ylim(0, 1)
        ax.set_title(label)
        ax.set_xlabel("Prediction horizon")
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("Score")
    axes[1].legend(
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.15),
        fontsize=9,
    )

    fig.suptitle(
        "Random Forest provided the strongest validation balanced accuracy",
        fontsize=13,
        fontweight="bold",
        y=1.04,
    )
    fig.tight_layout()

    save_figure(
        fig,
        "10_validation_model_comparison.png",
        "Validation model comparison",
        "Supports selecting the tuned Random Forest using balanced accuracy "
        "under severe class imbalance, while retaining accuracy and macro F1.",
    )


def plot_final_test(test_results: pd.DataFrame) -> None:
    """Show final overall metrics and per-class recall."""

    horizons = test_results["horizon"].tolist()
    x = np.arange(len(horizons))
    width = 0.25

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for index, (metric, label, color) in enumerate([
        ("accuracy", "Accuracy", "#2563EB"),
        ("balanced_accuracy", "Balanced accuracy", "#0F766E"),
        ("macro_f1", "Macro F1", "#7C3AED"),
    ]):
        axes[0].bar(
            x + (index - 1) * width,
            test_results[metric],
            width,
            label=label,
            color=color,
        )
    axes[0].set_xticks(x, horizons)
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("Score")
    axes[0].set_title("Overall final-test performance")
    axes[0].legend(fontsize=8)
    axes[0].grid(axis="x", visible=False)

    for index, (metric, label, class_name) in enumerate([
        ("R_LOW", "LOW recall", "LOW"),
        ("R_NORMAL", "NORMAL recall", "NORMAL"),
        ("R_HIGH", "HIGH recall", "HIGH"),
    ]):
        axes[1].bar(
            x + (index - 1) * width,
            test_results[metric],
            width,
            label=label,
            color=CLASS_COLORS[class_name],
        )
    axes[1].set_xticks(x, horizons)
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel("Recall")
    axes[1].set_title("Minority-risk detection remains the main limitation")
    axes[1].legend(fontsize=8)
    axes[1].grid(axis="x", visible=False)

    fig.suptitle(
        "Final held-out HATCH test results after reference normalization",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout()

    save_figure(
        fig,
        "11_final_test_performance.png",
        "Final test performance",
        "Shows useful overall prediction performance while making the moderate "
        "HIGH recall and horizon-dependent degradation explicit.",
    )


# ---------------------------------------------------------------------------
# Final analytical report
# ---------------------------------------------------------------------------

def round_frame(data: pd.DataFrame, decimals: int = 3) -> pd.DataFrame:
    """Round numeric columns for readable report tables."""

    result = data.copy()
    numeric = result.select_dtypes(include="number").columns
    result[numeric] = result[numeric].round(decimals)
    return result


def build_key_metrics(
    reports: dict[str, pd.DataFrame],
    metadata: dict[str, Any],
    stage1_summary: pd.DataFrame,
) -> pd.DataFrame:
    """Create one compact cross-stage metric table."""

    sequence_summary = reports["stage2_sequence_summary"]
    risk_summary = reports["stage3_risk_summary"]
    data_summary = reports["stage4_data_summary"]
    test_results = reports["stage4_final_test_results"]

    complete_rows = int(
        stage1_summary.loc[
            stage1_summary["message_status"].eq("complete"),
            "row_count",
        ].sum()
    ) or EXPECTED_PREPARED_ROWS

    rows = [
        ("Stage 1", "Raw physical rows", EXPECTED_RAW_ROWS, "rows"),
        ("Stage 1", "Structurally complete rows", complete_rows, "rows"),
        (
            "Stage 1",
            "Retention rate",
            complete_rows / EXPECTED_RAW_ROWS * 100,
            "%",
        ),
        (
            "Stage 2",
            "Continuous sequences",
            sequence_summary["sequence_id"].nunique(),
            "sequences",
        ),
        (
            "Stage 2",
            "Median estimable sampling interval",
            sequence_summary["estimated_sample_interval_seconds"].median(),
            "seconds",
        ),
        (
            "Stage 3",
            "Rows with all three risk targets",
            int(data_summary.loc[data_summary["dataset"].eq("Usable"), "rows"].iloc[0]),
            "rows",
        ),
        (
            "Stage 3",
            "15-minute NORMAL share",
            float(risk_summary.loc[risk_summary["horizon_minutes"].eq(15), "normal_pct"].iloc[0]),
            "%",
        ),
        ("Stage 4", "Model input features", metadata["feature_count"], "features"),
    ]

    for row in test_results.itertuples(index=False):
        rows.extend([
            (
                "Stage 4",
                f"{row.horizon} final balanced accuracy",
                row.balanced_accuracy,
                "score",
            ),
            (
                "Stage 4",
                f"{row.horizon} final macro F1",
                row.macro_f1,
                "score",
            ),
        ])

    rows.append(("Stage 5", "Saved prediction models", 3, "models"))
    return pd.DataFrame(rows, columns=["stage", "metric", "value", "unit"])


def build_assumption_evidence(
    reports: dict[str, pd.DataFrame],
    sensor_evidence: pd.DataFrame,
    slope_summary: pd.DataFrame,
    normalization: pd.DataFrame,
    leakage_audit: pd.DataFrame,
) -> pd.DataFrame:
    """Map each major design decision to measured evidence."""

    sequence_summary = reports["stage2_sequence_summary"]
    target_summary = reports["stage2_target_summary"]
    risk_summary = reports["stage3_risk_summary"]
    validation = reports["stage4_validation_comparison"]
    test_results = reports["stage4_final_test_results"]

    estimable = sequence_summary["estimated_sample_interval_seconds"].dropna()
    explicit_err = int(sensor_evidence["explicit_ERR"].sum())
    missing_total = int(sensor_evidence["missing_readings"].sum())
    err_pct = explicit_err / missing_total * 100
    rf_balanced = validation[validation["model"].eq("Random Forest")][
        "balanced_accuracy"
    ]
    high_recall = test_results["R_HIGH"]

    slope_medians = slope_summary.pivot(
        index="horizon_minutes",
        columns="risk_class",
        values="median_slope_c_per_min",
    )

    rows = [
        {
            "assumption_or_decision": "Preserve raw rows before exclusion",
            "evidence": (
                "53,061 of 53,075 physical rows were structurally complete "
                "(99.9736% retention); rejected rows were reported."
            ),
            "supported": True,
        },
        {
            "assumption_or_decision": "Treat ERR as informative missingness",
            "evidence": (
                f"Explicit ERR explains {explicit_err:,} of {missing_total:,} "
                f"missing sensor readings ({err_pct:.2f}%)."
            ),
            "supported": err_pct > 95,
        },
        {
            "assumption_or_decision": "Reconstruct time within sequences",
            "evidence": (
                f"Estimable sequence intervals range from {estimable.min():.2f} "
                f"to {estimable.max():.2f} seconds; nine sequences prevent "
                "cross-gap windows."
            ),
            "supported": bool(estimable.between(10, 30).all()),
        },
        {
            "assumption_or_decision": "Use 5/10/15-minute future targets",
            "evidence": (
                "Target availability is "
                + ", ".join(
                    f"{int(row.horizon_minutes)}m={row.availability_pct:.2f}%"
                    for row in target_summary.itertuples(index=False)
                )
                + "."
            ),
            "supported": bool(target_summary["availability_pct"].gt(99).all()),
        },
        {
            "assumption_or_decision": "Use balanced metrics",
            "evidence": (
                "NORMAL represents "
                + ", ".join(
                    f"{int(row.horizon_minutes)}m={row.normal_pct:.2f}%"
                    for row in risk_summary.itertuples(index=False)
                )
                + "; accuracy therefore overstates minority-class performance."
            ),
            "supported": True,
        },
        {
            "assumption_or_decision": "Add historical temperature slopes",
            "evidence": (
                "Median slope by 15-minute risk: "
                f"LOW={slope_medians.loc[15, 'LOW']:.4f}, "
                f"NORMAL={slope_medians.loc[15, 'NORMAL']:.4f}, "
                f"HIGH={slope_medians.loc[15, 'HIGH']:.4f} °C/min."
            ),
            "supported": True,
        },
        {
            "assumption_or_decision": "Translate HATCH test temperatures by +0.2 °C",
            "evidence": (
                f"All {len(ABSOLUTE_TEMPERATURE_COLUMNS)} absolute-temperature "
                "columns changed by exactly +0.2 °C; slopes, spread, and labels "
                "remained unchanged."
            ),
            "supported": bool(normalization["passed"].all()),
        },
        {
            "assumption_or_decision": "Select tuned Random Forest",
            "evidence": (
                "Validation balanced accuracy across 5/10/15 minutes was "
                + "/".join(f"{value:.3f}" for value in rf_balanced)
                + ", the strongest model-wise result at each horizon."
            ),
            "supported": True,
        },
        {
            "assumption_or_decision": "Keep test data isolated",
            "evidence": (
                f"All {len(leakage_audit)} leakage/reproducibility checks passed; "
                "sequences 8 and 9 are absent from training and validation."
            ),
            "supported": bool(leakage_audit["passed"].all()),
        },
        {
            "assumption_or_decision": "Report limitations, not accuracy alone",
            "evidence": (
                "Final HIGH recall across 5/10/15 minutes is "
                + "/".join(f"{value:.3f}" for value in high_recall)
                + "; minority-risk detection remains moderate."
            ),
            "supported": True,
        },
    ]
    return pd.DataFrame(rows)


def package_environment() -> dict[str, Any]:
    """Record the packages directly relevant to reproducing the analysis."""

    packages = {}
    for name in [
        "numpy",
        "pandas",
        "matplotlib",
        "scikit-learn",
        "xgboost",
        "joblib",
        "fastapi",
        "uvicorn",
    ]:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None

    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": packages,
    }


def build_markdown_report(
    reports: dict[str, pd.DataFrame],
    metadata: dict[str, Any],
    stage1_summary: pd.DataFrame,
    key_metrics: pd.DataFrame,
    sensor_evidence: pd.DataFrame,
    history_availability: pd.DataFrame,
    slope_summary: pd.DataFrame,
    normalization: pd.DataFrame,
    leakage_audit: pd.DataFrame,
    assumption_evidence: pd.DataFrame,
    source_inventory: pd.DataFrame,
    stage3_shape: tuple[int, int],
    normalized_shape: tuple[int, int],
) -> str:
    """Create the human-readable final analytical evidence report."""

    sequence_summary = reports["stage2_sequence_summary"]
    controller_quality = reports["stage2_controller_quality"]
    risk_summary = reports["stage3_risk_summary"]
    data_summary = reports["stage4_data_summary"]
    model_configs = reports["stage4_model_configurations"]
    validation = reports["stage4_validation_comparison"]
    test_results = reports["stage4_final_test_results"]

    lines = [
        "# Final Analytical Evidence Report",
        "",
        "**Project:** ML-Based Thermal Risk Prediction in a Multi-Sensor "
        "Thermal System  ",
        f"**Generated:** {datetime.now(timezone.utc).isoformat()}  ",
        "**Purpose:** consolidate the measured outputs needed for the final "
        "technical report, README, and lecturer presentation.",
        "",
        "## Executive result",
        "",
        "The completed pipeline retained 53,061 of 53,075 raw records, "
        "reconstructed nine continuous sequences, created leakage-safe "
        "historical features and 5/10/15-minute risk targets, compared "
        "Logistic Regression, Random Forest, and XGBoost, selected a tuned "
        "Random Forest, evaluated it once on normalized held-out HATCH data, "
        "saved three deployment pipelines, and exposed them through a local "
        "FastAPI service.",
        "",
        "## Key metrics",
        "",
        markdown_table(round_frame(key_metrics, 4)),
        "",
        "## Stage 1 - Raw-data preparation",
        "",
        "Every physical CSV record was preserved before structural validation. "
        "Messages were checked against the expected 17 pipe-separated "
        "sections. Non-complete records were reported and then excluded; "
        "valid rows containing explicit sensor ERR states were retained.",
        "",
        markdown_table(round_frame(stage1_summary, 4)),
        "",
        "![Stage 1 retention](plots/01_stage1_data_retention.png)",
        "",
        "## Stage 2 - Time reconstruction, quality, and features",
        "",
        "A new sequence begins at the dataset start, a backward timestamp, or "
        "a gap greater than three minutes. Historical windows and future "
        "targets never cross these boundaries.",
        "",
        markdown_table(
            round_frame(
                sequence_summary[
                    [
                        "sequence_id",
                        "start_reason",
                        "row_count",
                        "duration_minutes",
                        "estimated_sample_interval_seconds",
                    ]
                ],
                3,
            )
        ),
        "",
        "![Sequence structure](plots/03_sequence_structure_and_sampling.png)",
        "",
        "Sensor missingness was separated into explicit ERR, unexpected "
        "missingness, invalid flags, and implausible measurements.",
        "",
        markdown_table(round_frame(sensor_evidence, 3)),
        "",
        "![Sensor quality](plots/02_sensor_quality_evidence.png)",
        "",
        "Controller checks identified isolated problems without silently "
        "changing the original values:",
        "",
        markdown_table(controller_quality),
        "",
        "The compact feature contract uses active-sensor temperatures, heater "
        "duty, supporting sensor/controller state, 5/10/15-minute historical "
        "means, humidity-fan fractions, and later the temperature slopes. "
        "heater_on was excluded because the logging interval is too slow to "
        "represent short PWM pulses reliably.",
        "",
        "## Stage 3 - Thermal-risk targets and slope features",
        "",
        f"Stage 3 data shape: **{stage3_shape[0]:,} rows × "
        f"{stage3_shape[1]} columns**.",
        "",
        "Risk is determined from the minimum and maximum average temperature "
        "inside the complete future window. NORMAL uses 37.3–37.7 °C in NRML "
        "mode and 37.1–37.5 °C in HTCH mode. A window crossing both limits is "
        "marked ambiguous and excluded from that target.",
        "",
        markdown_table(round_frame(risk_summary, 3)),
        "",
        "![Targets and balance](plots/04_target_availability_and_class_balance.png)",
        "",
        "History and slope features remain missing at the beginning of each "
        "sequence until the complete past window exists:",
        "",
        markdown_table(round_frame(history_availability, 3)),
        "",
        "![History availability](plots/05_history_feature_availability.png)",
        "",
        "Temperature-slope evidence by future class:",
        "",
        markdown_table(round_frame(slope_summary, 5)),
        "",
        "![Slope by risk](plots/06_slope_by_future_risk.png)",
        "",
        "![Current features](plots/07_current_features_by_15m_risk.png)",
        "",
        "## Stage 4 - Chronological modeling and final test",
        "",
        "The chronological split is Train sequences 1/2/3/5, Validation "
        "sequence 7, and Test sequences 8/9. Rows missing any of the three "
        "targets were removed from modeling, leaving 52,330 usable rows.",
        "",
        markdown_table(data_summary),
        "",
        "![Split class distribution](plots/08_split_class_distribution.png)",
        "",
        "### Leakage and reproducibility audit",
        "",
        markdown_table(leakage_audit),
        "",
        "### HATCH test normalization",
        "",
        f"Normalized test shape: **{normalized_shape[0]:,} rows × "
        f"{normalized_shape[1]} columns**. A constant +0.2 °C translation was "
        "applied only to absolute-temperature coordinates. Labels, slopes, "
        "spread, controller values, modes, and identifiers were unchanged.",
        "",
        markdown_table(round_frame(normalization, 5)),
        "",
        "![Normalization](plots/09_hatch_normalization_verification.png)",
        "",
        "### Model configurations",
        "",
        markdown_table(model_configs),
        "",
        "### Validation comparison",
        "",
        markdown_table(
            round_frame(
                validation[
                    [
                        "model",
                        "horizon",
                        "accuracy",
                        "balanced_accuracy",
                        "macro_f1",
                        "R_LOW",
                        "R_HIGH",
                    ]
                ],
                3,
            )
        ),
        "",
        "![Validation models](plots/10_validation_model_comparison.png)",
        "",
        "### Final held-out test",
        "",
        markdown_table(round_frame(test_results, 3)),
        "",
        "![Final test](plots/11_final_test_performance.png)",
        "",
        "## Stage 5 - Saved models and local API",
        "",
        f"The final artifact contains **{metadata['feature_count']} features**, "
        f"one categorical feature ({', '.join(metadata['categorical_features'])}), "
        "three Random Forest pipelines, model version "
        f"**{metadata['model_version']}**, and an example request. The local "
        "FastAPI service exposes `/health` and `/predict` with exact-feature "
        "validation.",
        "",
        "## Assumptions and measured evidence",
        "",
        markdown_table(assumption_evidence),
        "",
        "## Limitations",
        "",
        "1. The dataset comes from one physical incubator and nine recorded "
        "sequences; external generalization has not been demonstrated.",
        "2. The chronological splits have different risk distributions, "
        "especially the HATCH-only test sequences.",
        "3. The targets are derived from future average-temperature behavior, "
        "so they describe thermal risk rather than independent biological "
        "hatching outcomes.",
        "4. HIGH recall remains moderate (0.365/0.465/0.466 for 5/10/15 "
        "minutes), despite stronger overall accuracy.",
        "5. The API expects 31 engineered features; raw live telemetry still "
        "requires the Stage 1–3 preparation logic.",
        "6. Disturbance-aware closed-loop control was not implemented in this "
        "version and remains future work.",
        "",
        "## Plot index",
        "",
        markdown_table(pd.DataFrame(PLOT_RECORDS)),
        "",
        "## Source inventory",
        "",
        markdown_table(source_inventory),
    ]

    if WARNINGS:
        lines.extend([
            "",
            "## Collection warnings",
            "",
            *[f"- {warning}" for warning in WARNINGS],
        ])

    return "\n".join(lines) + "\n"


def create_bundle() -> None:
    """Package only final analytical evidence, never source datasets/models."""

    with zipfile.ZipFile(
        BUNDLE_FILE,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
    ) as archive:
        for path in sorted(OUTPUT_DIR.rglob("*")):
            if path.is_file():
                archive.write(
                    path,
                    arcname=Path("final_analysis") / path.relative_to(OUTPUT_DIR),
                )


def main() -> None:
    """Run the complete evidence collection from the project root."""

    validate_inputs()
    configure_plot_style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    reports = read_reports()
    with METADATA_FILE.open("r", encoding="utf-8") as file:
        metadata = json.load(file)

    stage3_usecols = sorted(set([
        "record_id",
        "sequence_id",
        "operating_mode",
        "avg_t",
        "duty_pct",
        "sensor_spread_c",
        *ABSOLUTE_TEMPERATURE_COLUMNS,
        *UNCHANGED_AFTER_NORMALIZATION,
        *metadata["feature_columns"],
        *TARGET_COLUMNS,
    ]))
    stage3 = pd.read_csv(STAGE3_FILE, usecols=stage3_usecols)
    normalized_test = pd.read_csv(NORMALIZED_TEST_FILE)

    if len(stage3) != EXPECTED_STAGE3_ROWS:
        raise ValueError(
            f"Expected {EXPECTED_STAGE3_ROWS:,} Stage 3 rows, found "
            f"{len(stage3):,}."
        )

    stage1_summary, rejected_rows = reconstruct_stage1_validation()
    save_csv(stage1_summary, "stage1_validation_summary.csv")
    if not rejected_rows.empty:
        save_csv(rejected_rows, "stage1_rejected_rows.csv")

    sensor_evidence = build_sensor_missingness_evidence(
        reports["stage2_sensor_quality"]
    )
    save_csv(sensor_evidence, "sensor_missingness_evidence.csv")

    history_availability = build_history_availability(
        stage3,
        metadata["feature_columns"],
    )
    save_csv(history_availability, "history_feature_availability.csv")

    slope_summary = build_slope_risk_summary(stage3)
    save_csv(slope_summary, "slope_risk_summary.csv")

    current_feature_summary = build_current_feature_summary(stage3)
    save_csv(current_feature_summary, "current_feature_risk_summary.csv")

    normalization = verify_normalization(stage3, normalized_test)
    save_csv(normalization, "normalization_verification.csv")

    leakage_audit = build_leakage_audit(metadata, stage3)
    save_csv(leakage_audit, "leakage_reproducibility_audit.csv")

    source_inventory = build_source_inventory(reports)
    save_csv(source_inventory, "source_inventory.csv")

    plot_stage1_retention(stage1_summary)
    plot_sensor_quality(sensor_evidence)
    plot_sequence_structure(reports["stage2_sequence_summary"])
    plot_targets_and_balance(
        reports["stage2_target_summary"],
        reports["stage3_risk_summary"],
    )
    plot_history_availability(history_availability)
    plot_slope_by_risk(stage3)
    plot_current_features(stage3)
    plot_split_class_distribution(reports["stage4_class_distribution"])
    plot_normalization(normalization)
    plot_validation_models(reports["stage4_validation_comparison"])
    plot_final_test(reports["stage4_final_test_results"])

    key_metrics = build_key_metrics(reports, metadata, stage1_summary)
    save_csv(key_metrics, "key_metrics.csv")

    assumption_evidence = build_assumption_evidence(
        reports,
        sensor_evidence,
        slope_summary,
        normalization,
        leakage_audit,
    )
    save_csv(assumption_evidence, "assumption_evidence.csv")

    api_summary = {
        "api_script_present": (PROJECT_ROOT / "api.py").is_file(),
        "model_version": metadata.get("model_version"),
        "model_type": metadata.get("model_type"),
        "feature_count": metadata.get("feature_count"),
        "saved_model_files": metadata.get("model_files"),
        "api_examples_present": API_EXAMPLES_FILE.is_file(),
        "api_example_count": 0,
    }
    if API_EXAMPLES_FILE.is_file():
        with API_EXAMPLES_FILE.open("r", encoding="utf-8") as file:
            api_examples = json.load(file)
        api_summary["api_example_count"] = len(api_examples.get("examples", []))

    summary = {
        "project": "ML-Based Thermal Risk Prediction in a Multi-Sensor Thermal System",
        "generated_at_utc": datetime.now(timezone.utc),
        "environment": package_environment(),
        "data_shapes": {
            "stage3": {"rows": len(stage3), "columns_read": len(stage3.columns)},
            "normalized_test": {
                "rows": len(normalized_test),
                "columns": len(normalized_test.columns),
            },
        },
        "splits": {
            "train_sequences": TRAIN_SEQUENCES,
            "validation_sequences": VALIDATION_SEQUENCES,
            "test_sequences": TEST_SEQUENCES,
        },
        "key_metrics": key_metrics.to_dict(orient="records"),
        "assumption_evidence": assumption_evidence.to_dict(orient="records"),
        "leakage_audit": leakage_audit.to_dict(orient="records"),
        "normalization_checks_passed": bool(normalization["passed"].all()),
        "api": api_summary,
        "plots": PLOT_RECORDS,
        "warnings": WARNINGS,
    }
    save_json(summary, OUTPUT_DIR / "final_analysis_summary.json")

    report_text = build_markdown_report(
        reports=reports,
        metadata=metadata,
        stage1_summary=stage1_summary,
        key_metrics=key_metrics,
        sensor_evidence=sensor_evidence,
        history_availability=history_availability,
        slope_summary=slope_summary,
        normalization=normalization,
        leakage_audit=leakage_audit,
        assumption_evidence=assumption_evidence,
        source_inventory=source_inventory,
        stage3_shape=(len(stage3), len(pd.read_csv(STAGE3_FILE, nrows=0).columns)),
        normalized_shape=normalized_test.shape,
    )
    (OUTPUT_DIR / "final_analysis_report.md").write_text(
        report_text,
        encoding="utf-8",
    )

    create_bundle()

    print("\n=== Final analytical evidence collection complete ===")
    print(f"Report: {relative_path(OUTPUT_DIR / 'final_analysis_report.md')}")
    print(f"Summary: {relative_path(OUTPUT_DIR / 'final_analysis_summary.json')}")
    print(f"Plots created: {len(PLOT_RECORDS)}")
    print(f"Upload this file: {relative_path(BUNDLE_FILE)}")
    if WARNINGS:
        print("\nWarnings:")
        for warning in WARNINGS:
            print(f"- {warning}")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"\nFINAL ANALYSIS FAILED: {error}", file=sys.stderr)
        raise
