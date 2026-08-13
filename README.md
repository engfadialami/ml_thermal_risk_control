# ml_thermal_risk_control
Machine learning for early thermal risk prediction, system-lag estimation, and disturbance-aware control using multi-sensor time-series data.
# Project Workflow and Current Progress

The project is organized into seven main stages. The current implementation completes the data-preparation and ML-dataset-construction stages. Model training, evaluation, and control integration remain future work.

## Main Project Stages

| Stage | Description                                                                                   
| ----- | --------------------------------------------------------------------------------------------
| 1     | Raw data preparation and structural validation                                                
| 2     | Time-aware dataset construction, quality analysis, feature engineering, and target generation 
| 3     | Version 1: current-state temperature prediction and model comparison                         
| 4     | Version 2: history-aware prediction using lag features and thermal behavior                   
| 5     | Version 3: thermal-risk prediction and disturbance-aware bounded control                      
| 6     | Model integration, API development, versioning, and testing                                   
| 7     | Final evaluation, documentation, and presentation                                             

Stage 3 will compare regression models covered in the GSG Advanced Course, including Ridge Regression, Decision Tree, Random Forest, and XGBoost. A Dummy Regressor will be used as the minimum baseline. The models will predict the average system temperature 5, 10, and 15 minutes ahead.

---

## Stage 1: Raw Data Preparation and Validation

### Objective

Stage 1 converts the original incubator telemetry log into a structured and validated dataset without silently deleting problematic records. It is implemented as a standalone script named `p1.py`.

### Main Operations

* Loaded the raw CSV while safely handling malformed CSV rows.
* Preserved the original acquisition order using a unique `record_id`.
* Checked every telemetry message against the expected 17 pipe-separated sections.
* Classified structural problems before excluding rejected records.
* Supported both timestamp formats found in the raw log.
* Extracted temperature, humidity, and status flags for all four sensors.
* Preserved the sensor states:

  * `A`: included in the controller average.
  * `P`: excluded from the controller average.
  * `ERR`: sensor reading unavailable because of an explicit sensor error.
* Extracted the controller measurements and operating states.
* Retained structurally valid records containing `ERR` flags. Their unavailable measurements remain missing rather than being replaced with invented values.
* Generated validation and rejected-row reports.
* Saved the prepared dataset to a stable output file that is replaced when the stage is rerun.

### Results

| Result                              |    Value |
| ----------------------------------- | -------: |
| Raw physical rows                   |   53,075 |
| Accepted structurally complete rows |   53,061 |
| Rejected rows                       |       14 |
| Data-retention rate                 | 99.9736% |
| Prepared dataset columns            |       27 |

The final Stage 1 dataframe is `prepared_df`, and its saved output is:

```text
data/processed/incubator_prepared.csv
```

This file is the official input to Stage 2.

---

## Stage 2: Time-Aware Dataset and Feature Construction

### Objective

Stage 2 transforms the prepared records into a time-aware dataset suitable for analysis and future ML experiments. It is implemented in `p2.py`.

The stage preserves a comprehensive master table for reporting while also producing a compact table containing only the information required for model development.

### Main Operations

#### 1. Continuous-sequence identification

The readings were divided into continuous operating sequences. A new sequence begins when the time gap between consecutive records exceeds three minutes.

The three-minute threshold was selected because data acquisition may temporarily stop during egg rotation. This avoids incorrectly treating short operational interruptions as major breaks while preventing historical features or future targets from crossing real recording gaps.

#### 2. Sensor-quality analysis

Sensor readings were checked by distinguishing between:

* Missing values caused by explicit `ERR` flags.
* Unexpected missing measurements.
* Invalid sensor-status flags.
* Implausible temperature or humidity values.

The quality analysis identified nine rows containing unexpected sensor-quality issues. These rows were reported for inspection instead of being silently deleted.

#### 3. Controller-quality analysis

Controller fields were checked separately from sensor measurements. This produced a compact controller-quality summary and a report containing records with unexpected controller issues.

#### 4. Current-state feature definition

The primary predictive features are:

* `s2_t`
* `s3_t`
* `duty_pct`

These represent the two active temperature sensors and the effective heater duty cycle.

Supporting features include:

* Individual temperatures from all four sensors.
* Individual humidity measurements from all four sensors.
* Duty-cycle correction.
* Average temperature and humidity.
* Temperature spread.
* Humidity-fan state.
* Operating mode.

The recorded `heater_on` field was not selected as a main feature because its logging resolution is too low to represent the heater pulses reliably. The recorded `duty_pct` provides a more useful representation of heater activity.

#### 5. Historical-feature construction

Historical features were generated over 5-, 10-, and 15-minute windows for:

* `s2_t`
* `s3_t`
* `duty_pct`
* Humidity-fan state

These features will allow the Version 2 models to represent recent thermal behavior and thermal inertia rather than depending only on the current reading.

All historical features are restricted to their original continuous sequence.

#### 6. Future-target generation

Three regression targets were generated:

* `target_avg_t_5m`
* `target_avg_t_10m`
* `target_avg_t_15m`

They represent the average system temperature 5, 10, and 15 minutes into the future. Targets are not allowed to cross sequence boundaries.

### Stage 2 Outputs

Stage 2 produces two main datasets:

1. **Master dataframe (`master_df`)**

   A comprehensive 47-column table containing metadata, sequence information, quality indicators, current features, historical features, and future targets. It is intended for traceability, reporting, and detailed analysis.

2. **Compact ML table (`ml_table`)**

   A smaller table containing only the metadata, selected model features, and future targets required for the next modeling stage. It preserves the same records as the master table while avoiding unnecessary columns in the ML pipeline.

The following supporting reports are also generated:

* `sequence_summary`
* `sensor_quality_summary`
* `sensor_issue_rows`
* `controller_quality_summary`
* `controller_issue_rows`
* `feature_manifest`
* `target_summary`

### Stage 2 Result

Stage 2 successfully produced a sequence-aware, quality-checked, and ML-ready dataset without allowing historical information or future targets to cross discontinuous recording periods.

The comprehensive master table remains available for detailed reports, while the compact ML table provides a clear interface for Stage 3 model development.

---

## Current Project Status

The completed workflow is:

```text
Raw incubator log
        ↓
Stage 1: Structural validation and data preparation
        ↓
incubator_prepared.csv
        ↓
Stage 2: Sequences, quality checks, features, and future targets
        ↓
Master dataset + Compact ML dataset + Quality reports
```

The project is therefore ready to begin Stage 3: chronological data splitting, baseline evaluation, regression-model training, and comparison for the 5-, 10-, and 15-minute prediction horizons.
