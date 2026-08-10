import csv
import logging
from pathlib import Path

import pandas as pd


# --------------------------------------------------
# File settings
# --------------------------------------------------

INPUT_FILE = Path("data/raw/incubator_bt_log2.csv")
OUTPUT_FILE = Path("data/processed/incubator_prepared.csv")
EXPECTED_SECTIONS = 17


# --------------------------------------------------
# Logging configuration
# --------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(message)s",
    force=True
)

logger = logging.getLogger("thermal_pipeline")


# --------------------------------------------------
# 1. Load and validate the raw CSV
# --------------------------------------------------

def load_raw_data(file_path):
    file_path = Path(file_path)

    if not file_path.is_file():
        logger.error("Raw data file not found: %s", file_path)
        raise FileNotFoundError(f"File not found: {file_path}")

    records = []

    with file_path.open(
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as file:
        reader = csv.reader(file)
        header = next(reader, None)

        if header is None:
            logger.error("The CSV file is empty: %s", file_path)
            raise ValueError("The CSV file has no header")

        header = [column.strip() for column in header]
        expected_header = ["timestamp", "message"]

        if header != expected_header:
            logger.error(
                "Expected CSV header %s but found %s",
                expected_header,
                header
            )
            raise ValueError(
                f"Expected CSV header {expected_header}, found {header}"
            )

        expected_field_count = len(expected_header)

        for fields in reader:
            field_count = len(fields)

            timestamp = fields[0] if field_count >= 1 else ""

            # Preserve all text after the timestamp, even when
            # a malformed CSV row contains additional commas.
            raw_message = (
                ",".join(fields[1:])
                if field_count >= 2
                else ""
            )

            records.append(
                {
                    "source_line_number": reader.line_num,
                    "timestamp": timestamp,
                    "message": raw_message,
                    "csv_field_count": field_count,
                    "csv_structure_status": (
                        "valid"
                        if field_count == expected_field_count
                        else "malformed"
                    )
                }
            )

    raw_columns = [
        "source_line_number",
        "timestamp",
        "message",
        "csv_field_count",
        "csv_structure_status"
    ]

    raw_df = pd.DataFrame.from_records(
        records,
        columns=raw_columns
    )

    malformed_count = (
        raw_df["csv_structure_status"]
        .ne("valid")
        .sum()
    )

    logger.info(
        "Loaded %d physical rows from %s",
        len(raw_df),
        file_path.name
    )

    if malformed_count:
        logger.warning(
            "Found %d malformed CSV rows",
            malformed_count
        )

    return raw_df


# --------------------------------------------------
# 2. Create and validate the working table
# --------------------------------------------------

def create_working_table(raw_df):
    working_df = raw_df.copy()

    working_df = working_df.rename(
        columns={
            "timestamp": "timestamp_raw",
            "message": "raw_message"
        }
    )

    working_df.insert(
        0,
        "record_id",
        range(1, len(working_df) + 1)
    )

    logger.info(
        "Created working table with %d rows",
        len(working_df)
    )

    return working_df


def classify_messages(working_df, expected_sections=EXPECTED_SECTIONS):
    working_df = working_df.copy()

    message_text = working_df["raw_message"].str.strip()

    working_df["message_is_empty"] = message_text.eq("")

    working_df["message_section_count"] = (
        message_text.str.count(r"\|") + 1
    )

    working_df.loc[
        working_df["message_is_empty"],
        "message_section_count"
    ] = 0

    # Begin with the expected case, then overwrite rows
    # that match one of the rejection conditions.
    working_df["message_status"] = "complete"

    working_df.loc[
        working_df["message_is_empty"],
        "message_status"
    ] = "empty"

    incomplete_condition = (
        ~working_df["message_is_empty"]
        & (
            working_df["message_section_count"]
            < expected_sections
        )
    )

    working_df.loc[
        incomplete_condition,
        "message_status"
    ] = "incomplete"

    extra_sections_condition = (
        working_df["message_section_count"]
        > expected_sections
    )

    working_df.loc[
        extra_sections_condition,
        "message_status"
    ] = "extra_sections"

    malformed_csv_condition = (
        working_df["csv_structure_status"] != "valid"
    )

    # Run this condition last so malformed CSV rows keep
    # the most useful diagnostic status.
    working_df.loc[
        malformed_csv_condition,
        "message_status"
    ] = "malformed_csv_row"

    return working_df


def verify_raw_data_preservation(raw_df, working_df):
    if len(raw_df) != len(working_df):
        logger.error(
            "Row preservation failed: raw=%d, working=%d",
            len(raw_df),
            len(working_df)
        )
        raise ValueError("The number of rows changed")

    if not working_df["timestamp_raw"].equals(
        raw_df["timestamp"]
    ):
        logger.error("Raw timestamp values were changed")
        raise ValueError("Raw timestamp preservation failed")

    if not working_df["raw_message"].equals(
        raw_df["message"]
    ):
        logger.error("Raw message values were changed")
        raise ValueError("Raw message preservation failed")

    logger.info(
        "Raw-data preservation verified: %d rows",
        len(raw_df)
    )


def summarize_message_validation(working_df):
    message_summary = (
        working_df["message_status"]
        .value_counts()
        .rename_axis("message_status")
        .reset_index(name="row_count")
    )

    message_summary["percentage"] = (
        message_summary["row_count"]
        / len(working_df)
        * 100
    ).round(4)

    return message_summary


def separate_complete_and_rejected(working_df):
    complete_condition = (
        working_df["message_status"] == "complete"
    )

    clean_columns = [
        "record_id",
        "timestamp_raw",
        "raw_message"
    ]

    clean_df = working_df.loc[
        complete_condition,
        clean_columns
    ].copy()

    rejected_rows = working_df.loc[
        ~complete_condition
    ].copy()

    logger.info(
        "Retained %d complete rows and excluded %d rejected rows",
        len(clean_df),
        len(rejected_rows)
    )

    return clean_df, rejected_rows


# --------------------------------------------------
# 3. Convert timestamps and extract message values
# --------------------------------------------------

def convert_timestamps(timestamp_text):
    timestamp_text = timestamp_text.str.strip()

    # Format used by incubator_bt_log2.csv
    timestamps = pd.to_datetime(
        timestamp_text,
        format="%m/%d/%Y %H:%M",
        errors="coerce"
    )

    # Format used by incubator_bt_log_N1.csv
    still_invalid = timestamps.isna()

    timestamps.loc[still_invalid] = pd.to_datetime(
        timestamp_text.loc[still_invalid],
        format="%Y-%m-%d %H:%M:%S",
        errors="coerce"
    )

    return timestamps


def extract_labeled_value(section, label):
    pattern = rf"^\s*{label}\s*=\s*(.*?)\s*$"

    return section.str.extract(
        pattern,
        expand=False
    )


def prepare_data_columns(
    clean_df,
    expected_sections=EXPECTED_SECTIONS
):
    required_columns = {
        "record_id",
        "timestamp_raw",
        "raw_message"
    }

    missing_columns = required_columns - set(clean_df.columns)

    if missing_columns:
        logger.error(
            "Missing preparation columns: %s",
            sorted(missing_columns)
        )
        raise ValueError(
            f"Missing preparation columns: {sorted(missing_columns)}"
        )

    # record_id preserves acquisition order, including
    # readings that have the same minute-level timestamp.
    if not clean_df["record_id"].is_monotonic_increasing:
        logger.warning("Input rows were reordered using record_id")

    clean_df = clean_df.sort_values(
        "record_id",
        kind="stable"
    ).copy()

    message_parts = clean_df["raw_message"].str.split(
        "|",
        regex=False,
        expand=True
    )

    if message_parts.shape[1] != expected_sections:
        logger.error(
            "Expected %d message sections but found %d",
            expected_sections,
            message_parts.shape[1]
        )
        raise ValueError("Unexpected number of message sections")

    message_parts = message_parts.apply(
        lambda column: column.str.strip()
    )

    prepared_df = pd.DataFrame(index=clean_df.index)

    prepared_df["record_id"] = clean_df["record_id"]
    prepared_df["timestamp"] = convert_timestamps(
        clean_df["timestamp_raw"]
    )

    invalid_timestamp_count = (
        prepared_df["timestamp"].isna().sum()
    )

    if invalid_timestamp_count:
        logger.warning(
            "Found %d invalid timestamps",
            invalid_timestamp_count
        )

    number_pattern = r"[-+]?\d+(?:\.\d+)?"
    sensor_error_counts = {}
    unparsed_sensor_counts = {}

    for sensor_number in range(1, 5):
        sensor_pattern = (
            rf"^S{sensor_number}:\s*"
            rf"(?:(?P<temperature>{number_pattern})C\s+"
            rf"(?P<humidity>{number_pattern})%\s+"
            rf"(?P<flag>[AP])|"
            rf"(?P<error>ERR))$"
        )

        sensor_values = message_parts[
            sensor_number - 1
        ].str.extract(sensor_pattern)

        temperature = pd.to_numeric(
            sensor_values["temperature"],
            errors="coerce"
        ).astype("Float64")

        humidity = pd.to_numeric(
            sensor_values["humidity"],
            errors="coerce"
        ).astype("Float64")

        sensor_flag = (
            sensor_values["flag"]
            .astype("string")
            .mask(
                sensor_values["error"].eq("ERR"),
                "ERR"
            )
        )

        prepared_df[f"s{sensor_number}_t"] = temperature
        prepared_df[f"s{sensor_number}_h"] = humidity
        prepared_df[f"s{sensor_number}_flag"] = sensor_flag

        error_count = sensor_values["error"].eq("ERR").sum()

        if error_count:
            sensor_error_counts[f"S{sensor_number}"] = int(
                error_count
            )

        recognized_sensor = (
            sensor_values["error"].eq("ERR")
            | (
                temperature.notna()
                & humidity.notna()
                & sensor_values["flag"].notna()
            )
        )

        unparsed_count = (~recognized_sensor).sum()

        if unparsed_count:
            unparsed_sensor_counts[f"S{sensor_number}"] = int(
                unparsed_count
            )

    if sensor_error_counts:
        logger.warning(
            "Retained sensor ERR readings: %s",
            sensor_error_counts
        )

    if unparsed_sensor_counts:
        logger.warning(
            "Found unrecognized sensor sections: %s",
            unparsed_sensor_counts
        )

    numeric_sections = {
        "avg_t": (4, "T", "Float64"),
        "avg_h": (5, "H", "Float64"),
        "duty_pct": (6, "Duty", "Int64"),
        "duty_correction_pct": (7, "Corr", "Int64"),
        "heater_on": (8, "Htr", "Int64"),
        "humidity_fan_on": (9, "HFan", "Int64"),
        "sensor_spread_c": (12, "Sprd", "Float64"),
        "active_sensors_count": (13, "AvgSens", "Int64"),
        "rotation_minutes_remaining": (14, "RotMin", "Int64"),
        "low_edge_counter": (15, "LEC", "Int64"),
        "high_edge_counter": (16, "HEC", "Int64")
    }

    for output_column, (
        section_position,
        raw_label,
        data_type
    ) in numeric_sections.items():
        values = extract_labeled_value(
            message_parts[section_position],
            raw_label
        )

        prepared_df[output_column] = pd.to_numeric(
            values,
            errors="coerce"
        ).astype(data_type)

    prepared_df["servo_position"] = (
        extract_labeled_value(
            message_parts[10],
            "Servo"
        ).astype("string")
    )

    prepared_df["operating_mode"] = (
        extract_labeled_value(
            message_parts[11],
            "Mod"
        ).astype("string")
    )

    final_columns = [
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
        "high_edge_counter"
    ]

    prepared_df = (
        prepared_df[final_columns]
        .reset_index(drop=True)
    )

    missing_counts = prepared_df.isna().sum()
    missing_counts = missing_counts[missing_counts > 0]

    if not missing_counts.empty:
        logger.warning(
            "Prepared columns containing missing values: %s",
            missing_counts.to_dict()
        )

    logger.info(
        "Prepared %d rows with %d columns",
        len(prepared_df),
        prepared_df.shape[1]
    )

    return prepared_df


# --------------------------------------------------
# 4. Build and save the final prepared DataFrame
# --------------------------------------------------

def build_prepared_dataframe(
    file_path,
    expected_sections=EXPECTED_SECTIONS
):
    raw_df = load_raw_data(file_path)
    working_df = create_working_table(raw_df)

    working_df = classify_messages(
        working_df,
        expected_sections=expected_sections
    )

    # Verify preservation before rejected rows are removed.
    verify_raw_data_preservation(raw_df, working_df)

    message_summary = summarize_message_validation(
        working_df
    )

    clean_df, rejected_rows = (
        separate_complete_and_rejected(working_df)
    )

    prepared_df = prepare_data_columns(
        clean_df,
        expected_sections=expected_sections
    )

    return prepared_df, message_summary, rejected_rows


def save_prepared_dataframe(prepared_df, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    file_already_exists = output_path.is_file()

    prepared_df.to_csv(
        output_path,
        index=False,
        date_format="%Y-%m-%d %H:%M:%S"
    )

    action = "Replaced" if file_already_exists else "Saved"

    logger.info(
        "%s prepared data file: %s",
        action,
        output_path
    )

    return output_path


# --------------------------------------------------
# 5. Run Stage 1
# --------------------------------------------------

def main():
    prepared_df, message_summary, rejected_rows = (
        build_prepared_dataframe(
            INPUT_FILE,
            expected_sections=EXPECTED_SECTIONS
        )
    )

    saved_path = save_prepared_dataframe(
        prepared_df,
        OUTPUT_FILE
    )

    print("\nMessage validation summary:")
    print(message_summary.to_string(index=False))

    print("\nRejected rows:")

    if rejected_rows.empty:
        print("No rejected rows found.")
    else:
        rejected_columns = [
            "record_id",
            "source_line_number",
            "timestamp_raw",
            "csv_field_count",
            "message_section_count",
            "message_status",
            "raw_message"
        ]

        print(
            rejected_rows[rejected_columns]
            .to_string(index=False)
        )

    print("\nPrepared shape:", prepared_df.shape)
    print("Saved to:", saved_path)

    print("\nFirst three prepared rows:")
    print(prepared_df.head(3).to_string(index=False))


if __name__ == "__main__":
    main()
