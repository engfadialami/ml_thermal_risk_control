# ML-Based Thermal Risk Prediction in a Multi-Sensor Thermal System

An end-to-end machine-learning capstone that converts noisy incubator telemetry into leakage-safe, multi-horizon thermal-risk predictions and serves the final models through a local FastAPI interface.

**Author:** Fady Alami  
**Course:** GSG-PSSAR Advanced Training  
**Final model:** Random Forest classifier  
**Prediction horizons:** 5, 10, and 15 minutes  
**Risk classes:** `LOW`, `NORMAL`, `HIGH`  
**Model version:** `1.0`

## 1. Executive summary

The project began with a raw incubator log containing sensor readings, controller states, malformed records, explicit sensor errors, interruptions, and two operating modes. The final workflow:

1. preserves every physical CSV row before making exclusions;
2. validates and parses the telemetry message structure;
3. reconstructs continuous time sequences;
4. creates history features and future risk targets without crossing gaps;
5. compares Logistic Regression, Random Forest, and XGBoost;
6. selects and evaluates a tuned Random Forest using balanced metrics;
7. saves three complete preprocessing-and-model pipelines; and
8. exposes the models through a local REST API.

The pipeline retained **53,061 of 53,075 raw rows (99.9736%)**. After requiring all three prediction targets, **52,330 rows** remained for modeling. On the held-out normalized HATCH test sequences, the final Random Forest achieved balanced accuracy of **0.650**, **0.648**, and **0.595** at 5, 10, and 15 minutes respectively.

The implementation predicts thermal risk. It does **not** yet implement disturbance-aware closed-loop control; that is retained as future work.

## 2. Problem definition

The physical system records four temperature/humidity sensors together with heater, fan, servo, operating-mode, and controller information. A stable current reading is not enough to describe near-future risk because the system has thermal inertia, sensor failures, control actions, and operating interruptions.

The machine-learning task is therefore:

> Given the current state and recent 5/10/15-minute history, predict whether the future average temperature will become `LOW`, remain `NORMAL`, or become `HIGH` during the next 5, 10, or 15 minutes.

The project treats this as three related classification problems, one for each prediction horizon.

## 3. Final results at a glance

| Item | Result |
| --- | ---: |
| Raw physical records | 53,075 |
| Structurally complete records | 53,061 |
| Data retention | 99.9736% |
| Continuous sequences | 9 |
| Median sampling interval | 16.349 seconds |
| Rows with all three risk targets | 52,330 |
| Model input features | 31 |
| Candidate model families | 3 |
| Saved deployment models | 3 |
| Leakage/reproducibility checks passed | 5 of 5 |

### Final held-out test performance

| Horizon | Accuracy | Balanced accuracy | Macro F1 | LOW recall | HIGH recall |
| --- | ---: | ---: | ---: | ---: | ---: |
| 5 min | 0.831 | 0.650 | 0.639 | 0.692 | 0.365 |
| 10 min | 0.766 | 0.648 | 0.639 | 0.641 | 0.465 |
| 15 min | 0.669 | 0.595 | 0.596 | 0.581 | 0.466 |

The decrease with horizon is expected: longer forecasts accumulate more uncertainty. The moderate `HIGH` recall is the main remaining modeling limitation and is reported explicitly rather than hidden by overall accuracy.

## 4. Repository structure

```text
ml_thermal_risk_control/
├── p1.py                              # Raw-data parsing and validation
├── p2.py                              # Sequences, quality checks, history, future averages
├── p3.py                              # Risk targets and temperature slopes
├── p4.py                              # Final model comparison and held-out evaluation
├── p4_tunning.py                      # Tuning experiments and diagnostics
├── prepare_stage4_normalized_test.py  # HATCH test reference translation
├── p5.py                              # Final training and model serialization
├── api.py                             # FastAPI service
├── api_test_examples.py               # Five real held-out API examples
├── p6_final_analysis.py               # Final evidence tables, plots, and bundle
├── requirements.txt
├── data/
│   ├── raw/                           # Not distributed in the compact submission
│   └── processed/                     # Generated intermediate datasets
├── models/stage5/
│   ├── random_forest_5m.joblib
│   ├── random_forest_10m.joblib
│   ├── random_forest_15m.joblib
│   ├── model_metadata.json
│   └── api_test_examples.json
└── reports/
    ├── stage2/
    ├── stage3/
    ├── stage4/
    └── final_analysis/
        ├── final_analysis_report.md
        ├── final_analysis_summary.json
        ├── supporting CSV files
        └── plots/
```

The raw and processed datasets are intentionally omitted from the compact shared archive because of their size. They are regenerated locally by running the stages in order.

## 5. Environment setup

The final evidence bundle was generated with:

- Python 3.13.3
- NumPy 2.4.6
- pandas 3.0.3
- scikit-learn 1.9.0
- XGBoost 3.4.1
- joblib 1.5.3
- Matplotlib 3.11.0
- FastAPI 0.141.1
- Uvicorn 0.52.3

### Windows setup

```powershell
py -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If a minimal environment is preferred, the core project packages are:

```powershell
pip install numpy pandas scipy scikit-learn xgboost joblib matplotlib seaborn fastapi "uvicorn[standard]"
```

## 6. Required input

Place the raw CSV at:

```text
data/raw/incubator_bt_log2.csv
```

The file must contain the header:

```text
timestamp,message
```

The telemetry message is expected to contain 17 pipe-separated sections describing four sensors and controller state.

## 7. Reproduce the complete workflow

Run all commands from the project root.

### Stage 1 - validate and prepare the raw log

```powershell
python p1.py
```

Main output:

```text
data/processed/incubator_prepared.csv
```

Stage 1 loads the CSV row by row so an extra comma does not silently destroy acquisition order. It accepts two timestamp formats, records the original source line, checks the expected 17 message sections, reports rejected rows, and retains valid rows containing explicit `ERR` sensor states.

Measured result:

- 53,075 raw physical rows;
- 53,061 structurally complete rows;
- 14 rejected rows;
- 99.9736% retention;
- 27 prepared columns.

### Stage 2 - reconstruct sequences and build time-aware features

```powershell
python p2.py
```

Main outputs:

```text
data/processed/incubator_stage2_master.csv
data/processed/incubator_stage2_ml.csv
reports/stage2/*.csv
```

A new continuous sequence starts at:

- the beginning of the dataset;
- a backward timestamp; or
- a time gap greater than three minutes.

Nine sequences were identified. Estimable sampling intervals ranged from approximately **16.32 to 16.44 seconds**, supporting time-window construction inside each sequence.

Quality checks separate:

- explicit `ERR` readings;
- unexpected missing values;
- invalid sensor flags;
- implausible measurements; and
- isolated controller-field problems.

The explicit `ERR` state explains **1,456 of 1,462 missing sensor readings (99.59%)**. This is why `ERR` is preserved as informative missingness and numerical replacement is deferred to the model pipeline.

### Stage 3 - create leakage-safe thermal-risk targets

```powershell
python p3.py
```

Main outputs:

```text
data/processed/incubator_stage3_targets.csv
reports/stage3/temperature_risk_summary.csv
```

The stage adds linear temperature slopes over the previous 5, 10, and 15 minutes. Risk is calculated from the minimum and maximum `avg_t` within the complete future window.

| Mode | LOW boundary | NORMAL band | HIGH boundary |
| --- | ---: | ---: | ---: |
| `NRML` | below 37.3 °C | 37.3-37.7 °C | above 37.7 °C |
| `HTCH` | below 37.1 °C | 37.1-37.5 °C | above 37.5 °C |

If a future window crosses both the LOW and HIGH boundaries, it is marked ambiguous and excluded from that target instead of forcing an unreliable label.

| Horizon | LOW | NORMAL | HIGH | Available rows | Ambiguous |
| --- | ---: | ---: | ---: | ---: | ---: |
| 5 min | 2,884 | 48,962 | 1,053 | 52,899 | 13 |
| 10 min | 4,290 | 46,642 | 1,715 | 52,647 | 118 |
| 15 min | 5,558 | 44,502 | 2,270 | 52,330 | 290 |

### Stage 4 - compare models and evaluate the selected model

First create the normalized held-out HATCH test dataset:

```powershell
python prepare_stage4_normalized_test.py
```

Then run the final model comparison:

```powershell
python p4.py
```

Main outputs:

```text
data/processed/incubator_stage4_test_normalized.csv
reports/stage4/data_summary.csv
reports/stage4/class_distribution.csv
reports/stage4/model_configurations.csv
reports/stage4/validation_comparison.csv
reports/stage4/final_test_results.csv
```

#### Chronological split

| Split | Sequences | Usable rows |
| --- | --- | ---: |
| Train | 1, 2, 3, 5 | 28,264 |
| Validation | 7 | 19,759 |
| Test | 8, 9 | 4,307 |

The split is sequence-based, not random. This preserves chronology and prevents neighboring rows from the same physical run appearing on both sides of a split.

#### Candidate models

- Logistic Regression: interpretable linear baseline with scaling and class balancing.
- Random Forest: nonlinear tree ensemble with bounded depth and balanced class weights.
- XGBoost: boosted-tree comparison with tuned depth and iteration count.

Because the `NORMAL` class dominates all horizons, **balanced accuracy** is the primary selection metric. Macro F1, overall accuracy, and minority-class recall are also reported.

#### Selected configuration

| Horizon | Trees | Max depth | Minimum leaf size |
| --- | ---: | ---: | ---: |
| 5 min | 200 | 6 | 20 |
| 10 min | 200 | 6 | 20 |
| 15 min | 200 | 6 | 10 |

Random Forest produced the strongest validation balanced accuracy at every horizon: **0.653**, **0.628**, and **0.627**.

#### Why the HATCH test set is translated by +0.2 °C

The held-out sequences use the `HTCH` operating band, whose center is 0.2 °C below the `NRML` band used in the earlier sequences. A coordinate translation aligns the absolute-temperature reference without changing the physical trend or label definition.

The normalization script verifies that:

- exactly +0.2 °C is applied to 14 absolute-temperature columns;
- slopes remain unchanged;
- sensor spread remains unchanged;
- controller features remain unchanged;
- identifiers and operating mode remain unchanged; and
- risk labels remain unchanged.

This operation is performed only on the held-out HATCH test file; it does not modify the original Stage 3 dataset.

### Stage 5 - train and save final deployment pipelines

```powershell
python p5.py
```

The final models are trained on the approved Train + Validation sequences. Test sequences 8 and 9 are not read by `p5.py` and are not used for training.

Each `.joblib` file contains the complete preprocessing and Random Forest pipeline, including:

- median imputation for numerical values;
- most-frequent imputation for categorical values;
- one-hot encoding for `operating_mode`; and
- the trained classifier.

The metadata file records the ordered 31-feature contract, model files, model version, configurations, sequence assignments, and a real prepared request example.

### Stage 6 - generate final analytical evidence

```powershell
python p6_final_analysis.py --project-root .
```

This creates a consolidated analytical report, machine-readable summary, supporting CSV files, and 11 plots under:

```text
reports/final_analysis/
```

The plots support the main design assumptions: row preservation, informative missingness, sequence-aware windows, target availability, class imbalance, slope usefulness, feature context, split shift, HATCH normalization, model selection, and final test performance.

## 8. Feature contract

The final API requires **31 engineered features**. They include:

- four sensor temperatures and four sensor humidities;
- sensor flags;
- current average temperature and humidity;
- current heater duty and duty correction;
- humidity-fan state;
- sensor spread;
- operating mode;
- 5/10/15-minute means for `s2_t`, `s3_t`, and `duty_pct`;
- 5/10/15-minute humidity-fan fractions; and
- 5/10/15-minute average-temperature slopes.

The API validates the exact feature names. Missing or unexpected features return HTTP status `422` with a clear error list.

### Why `heater_on` was not used

The logging interval is approximately 16 seconds, while heater PWM pulses may be shorter. A single logged `heater_on` state can therefore misrepresent heater activity. `duty_pct` is retained because it describes the commanded heating effort more reliably at this logging resolution.

### Leakage controls

The following columns are excluded from model input:

- `record_id` and sequence identifiers;
- estimated timestamps and target segment identifiers;
- future average-temperature columns; and
- the three risk targets.

The final audit confirmed:

1. no future values or targets are present among model features;
2. Train and Validation sequences are disjoint;
3. Test sequences are absent from final training;
4. all three targets exist; and
5. the saved feature count matches the 31-column list.

## 9. Run and use the API

Train/save the models first, then start the local server:

```powershell
uvicorn api:app --reload --host 127.0.0.1 --port 8000
```

Open the automatic interactive documentation:

```text
http://127.0.0.1:8000/docs
```

### Check API health

1. Expand `GET /health`.
2. Click **Try it out**.
3. Click **Execute**.

Expected response structure:

```json
{
  "status": "ready",
  "model_type": "RandomForestClassifier",
  "model_version": "1.0",
  "loaded_models": 3,
  "feature_count": 31
}
```

### Make a prediction

1. Expand `POST /predict`.
2. Click **Try it out**.
3. Keep or replace the provided `features` object.
4. Click **Execute**.

Expected response structure:

```json
{
  "temperature_risk_5m": "NORMAL",
  "temperature_risk_10m": "NORMAL",
  "temperature_risk_15m": "NORMAL"
}
```

The output is a prediction from the saved model pipelines. It is not a direct heater command.

## 10. Test the API with five held-out examples

Generate the five examples after preparing the normalized test file and model metadata:

```powershell
python api_test_examples.py
```

The script selects real rows from held-out test sequences, prioritizing:

- LOW outcomes;
- HIGH outcomes;
- large negative slopes;
- large positive slopes; and
- a stable comparison case.

The examples are saved to:

```text
models/stage5/api_test_examples.json
```

They are for inference testing only and are never added to training.

## 11. Important development difficulties and solutions

### Malformed CSV rows

**Symptom:** a conventional CSV load could fail or shift columns when messages contained extra commas.  
**Diagnosis:** the physical record was valid as a line even when the CSV field count was not exactly two.  
**Solution:** parse row by row, preserve the first field as timestamp, rejoin all remaining fields as the raw message, and report structural status before excluding anything.

### Sensor errors mixed with missing data

**Symptom:** numerical sensor readings were missing, especially for sensor 3.  
**Diagnosis:** most missing values were explicitly explained by the `ERR` status rather than random corruption.  
**Solution:** preserve `ERR`, leave its measurement missing, carry sensor flags as categorical context, and let the saved pipeline impute numerical values consistently.

### Time windows crossing interruptions

**Symptom:** ordinary rolling features could join unrelated operating periods or cross backward timestamps.  
**Diagnosis:** the raw log contains real gaps and two backward-timestamp events.  
**Solution:** reconstruct nine continuous sequences and calculate every historical feature and future target inside its original sequence only.

### Class imbalance made accuracy misleading

**Symptom:** high overall accuracy could coexist with poor LOW/HIGH detection because `NORMAL` represents 85-93% of available labels.  
**Diagnosis:** the minority risk classes matter operationally but contribute little to raw accuracy.  
**Solution:** use balanced accuracy as the main selection metric, class-weighted models, macro F1, and separate LOW/HIGH recalls.

### Initial tree models overfit

**Symptom:** highly flexible trees could fit training data much better than chronological validation data.  
**Diagnosis:** deep trees learned sequence-specific patterns and majority-class structure.  
**Solution:** tune maximum depth and minimum leaf size, keep a fixed random seed, and select the constrained Random Forest on validation balanced accuracy.

### Operating-mode reference shift

**Symptom:** the held-out test sequences were HATCH-only and used a temperature band centered 0.2 °C lower than NRML data.  
**Diagnosis:** comparing absolute temperatures directly would mix coordinate-reference shift with model generalization.  
**Solution:** create a separate normalized test copy, translate only absolute-temperature coordinates by +0.2 °C, verify unchanged labels/slopes/spread/controller values, and evaluate the finalist once.

### Turning a model into a usable API

**Symptom:** a trained estimator alone did not guarantee correct input ordering or preprocessing at inference time.  
**Diagnosis:** deployment must reproduce imputation, categorical encoding, and the exact feature contract.  
**Solution:** save complete pipelines, store ordered metadata and a real example, validate exact inputs, and expose `/health` plus `/predict` through FastAPI and Swagger UI.

## 12. Evidence supporting the main assumptions

| Assumption or decision | Measured evidence |
| --- | --- |
| Preserve rows before exclusion | 53,061 of 53,075 physical rows retained; 14 rejected rows reported |
| Treat `ERR` as informative | 99.59% of sensor missingness explained by explicit `ERR` |
| Use sequence-aware windows | Nine sequences; stable estimable sampling around 16.35 seconds |
| Use three forecast horizons | Target availability remains 99.78%, 99.54%, and 99.29% |
| Use balanced metrics | `NORMAL` is 92.56%, 88.59%, and 85.04% by horizon |
| Add slope features | Median 15-minute slope is about 0.0039 °C/min before HIGH risk versus approximately zero for LOW/NORMAL |
| Normalize HATCH coordinates | All 14 absolute-temperature columns changed exactly +0.2 °C; protected columns stayed unchanged |
| Select Random Forest | Best validation balanced accuracy at all three horizons |
| Protect the test set | All five leakage/reproducibility checks passed |

Association in the slope and feature plots is evidence of predictive relevance, not proof of physical causation.

## 13. Limitations

- Data comes from one physical incubator and nine recorded sequences.
- The chronological splits have different class distributions; the test set is HATCH-only.
- Risk labels are derived from future average-temperature behavior, not independent biological hatch outcomes.
- `HIGH` recall remains moderate, especially at 5 minutes.
- The API accepts engineered features, not raw live telemetry.
- Model version `1.0` is the only active version in this implementation.
- The service is verified locally; public cloud deployment is not required for the current capstone.
- Disturbance-aware closed-loop control has not been implemented.

## 14. Recommended improvements

The most useful next improvements are:

1. collect more LOW and HIGH events under both operating modes;
2. validate on another incubator or a later independent recording period;
3. move Stage 1-3 transformations into a reusable online feature builder;
4. tune class-specific thresholds or costs to improve `HIGH` recall;
5. add probability calibration and confidence reporting;
6. add API versioning and model-monitoring endpoints if required for production;
7. monitor data drift and sensor-error rates after deployment; and
8. evaluate any future controller in simulation with safety bounds before hardware use.

## 15. Project conclusion

The capstone demonstrates a complete and reproducible path from imperfect telemetry to a locally deployable multi-horizon classification service. The strongest parts of the work are the row-preserving validation, sequence-aware feature construction, explicit leakage controls, balanced evaluation, transparent HATCH normalization, and saved end-to-end pipelines.

The final Random Forest is useful as an early-warning model, but it should be interpreted with the reported class-specific limitations. The appropriate next step is better minority-event data and online preprocessing, not unnecessary additional tuning on the existing test set.

