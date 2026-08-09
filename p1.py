import logging
from pathlib import Path

import pandas as pd
import csv

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

        required_columns = {"timestamp", "message"}
        missing_columns = required_columns - set(header)

        if missing_columns:
            logger.error(
                "Missing required columns: %s",
                sorted(missing_columns)
            )
            raise ValueError(
                f"Missing required columns: {sorted(missing_columns)}"
            )

        expected_field_count = len(header)

        for fields in reader:
            field_count = len(fields)

            timestamp = fields[0] if field_count >= 1 else ""

            # Joining preserves all text after the first CSV field
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

    raw_df = pd.DataFrame(records)

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
# 2. Create a separate working table
# --------------------------------------------------

def create_working_table(raw_df):
    clean_df = raw_df.copy()

    clean_df = clean_df.rename(
        columns={
            "timestamp": "timestamp_raw",
            "message": "raw_message"
        }
    )

    clean_df.insert(
        0,
        "record_id",
        range(1, len(clean_df) + 1)
    )

    logger.info(
        "Created working table with %d rows",
        len(clean_df)
    )

    return clean_df


# --------------------------------------------------
# 3. Classify complete and problematic messages
# --------------------------------------------------

def classify_messages(clean_df, expected_sections=17):
    clean_df = clean_df.copy()

    message_text = clean_df["raw_message"].str.strip()

    clean_df["message_is_empty"] = message_text.eq("")

    clean_df["message_section_count"] = (
        message_text.str.count(r"\|") + 1
    )

    clean_df.loc[
        clean_df["message_is_empty"],
        "message_section_count"
    ] = 0

    # Start by considering every message complete
    clean_df["message_status"] = "complete"

    # Change the status only for rows matching each condition
    clean_df.loc[
        clean_df["message_is_empty"],
        "message_status"
    ] = "empty"

    incomplete_condition = (
        ~clean_df["message_is_empty"]
        & (
            clean_df["message_section_count"]
            < expected_sections
        )
    )

    clean_df.loc[
        incomplete_condition,
        "message_status"
    ] = "incomplete"

    extra_sections_condition = (
        clean_df["message_section_count"]
        > expected_sections
    )

    clean_df.loc[
        extra_sections_condition,
        "message_status"
    ] = "extra_sections"

    malformed_csv_condition = (
        clean_df["csv_structure_status"] != "valid"
    )

    clean_df.loc[
        malformed_csv_condition,
        "message_status"
    ] = "malformed_csv_row"

    return clean_df


# --------------------------------------------------
# 4. Verify that the original rows were preserved
# --------------------------------------------------

def verify_raw_data_preservation(raw_df, clean_df):
    if len(raw_df) != len(clean_df):
        logger.error(
            "Row preservation failed: raw=%d, working=%d",
            len(raw_df),
            len(clean_df)
        )
        raise ValueError("The number of rows changed")

    if not clean_df["timestamp_raw"].equals(raw_df["timestamp"]):
        logger.error("Raw timestamp values were changed")
        raise ValueError("Raw timestamp preservation failed")

    if not clean_df["raw_message"].equals(raw_df["message"]):
        logger.error("Raw message values were changed")
        raise ValueError("Raw message preservation failed")

    logger.info(
        "Raw-data preservation verified: %d rows",
        len(raw_df)
    )


# --------------------------------------------------
# 5. Summarize message validation
# --------------------------------------------------

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
# 6. Run the current preparation pipeline
# --------------------------------------------------

def run_validation_pipeline(file_path, expected_sections=17):
    raw_df = load_raw_data(file_path)

    working_df = create_working_table(raw_df)

    working_df = classify_messages(
        working_df,
        expected_sections=expected_sections
    )

    # This check must happen before excluding any rows
    verify_raw_data_preservation(raw_df, working_df)

    message_summary = summarize_message_validation(
        working_df
    )

    clean_df, rejected_rows = (
        separate_complete_and_rejected(working_df)
    )

    return (
        raw_df,
        working_df,
        clean_df,
        message_summary,
        rejected_rows
    )

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

def prepare_data_columns(clean_df):
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

    # record_id preserves acquisition order,
    # including readings with the same timestamp
    if not clean_df["record_id"].is_monotonic_increasing:
        logger.warning(
            "Input rows were reordered using record_id"
        )

    clean_df = clean_df.sort_values(
        "record_id",
        kind="stable"
    ).copy()

    # The message still has 17 raw sections internally
    message_parts = clean_df["raw_message"].str.split(
        "|",
        regex=False,
        expand=True
    )

    if message_parts.shape[1] != 17:
        logger.error(
            "Expected 17 message sections but found %d",
            message_parts.shape[1]
        )
        raise ValueError(
            "Unexpected number of message sections"
        )

    message_parts = message_parts.apply(
        lambda column: column.str.strip()
    )

    prepared_df = pd.DataFrame(index=clean_df.index)

    # Identification and time
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

    # --------------------------------------------------
    # Extract the four sensor sections
    # --------------------------------------------------

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
            |
            (
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

    # --------------------------------------------------
    # Extract numeric controller values
    # --------------------------------------------------

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

    # --------------------------------------------------
    # Extract text controller values
    # --------------------------------------------------

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

    # --------------------------------------------------
    # Arrange the final 27 columns
    # --------------------------------------------------

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

    logger.info(
        "Prepared %d rows with %d columns",
        len(prepared_df),
        prepared_df.shape[1]
    )

    return prepared_df

# --------------------------------------------------
# Execute the pipeline
# --------------------------------------------------

file_path = "data/raw/incubator_bt_log2.csv"

(
    raw_df,
    working_df,
    clean_df,
    message_summary,
    rejected_rows
) = run_validation_pipeline(
    file_path,
    expected_sections=17
)

prepared_df = prepare_data_columns(clean_df)

# --------------------------------------------------
# Display the results
# --------------------------------------------------

print("\nMessage validation summary:")
print(message_summary.to_string(index=False))

print("\nRejected rows:")
print(
    rejected_rows[
        [
            "record_id",
            "source_line_number",
            "timestamp_raw",
            "csv_field_count",
            "message_section_count",
            "message_status",
            "raw_message"
        ]
    ].to_string(index=False)
)

print("Prepared shape:", prepared_df.shape)

print("\nPrepared columns:")
print(prepared_df.columns.tolist())

print("\nFirst three prepared rows:")
print(prepared_df.head(3).to_string(index=False))