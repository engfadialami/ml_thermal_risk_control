# =========================================================
# Stage 4 - ML Classification
# =========================================================

from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

INPUT_FILE = "data/processed/incubator_stage3_targets.csv"

REPORT_DIR = Path("reports/stage4")
REPORT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_COLUMNS = [
    "temperature_risk_5m",
    "temperature_risk_10m",
    "temperature_risk_15m",
]

HORIZON_NAMES = {
    "temperature_risk_5m": "5 min",
    "temperature_risk_10m": "10 min",
    "temperature_risk_15m": "15 min",
}

CLASS_ORDER = ["LOW", "NORMAL", "HIGH"]

TRAIN_SEQUENCES = [1, 2, 3, 5]
VAL_SEQUENCES = [7]
TEST_SEQUENCES = [8, 9]


# ---------------------------------------------------------
# Load Stage 3 dataset
# ---------------------------------------------------------

df = pd.read_csv(INPUT_FILE)
original_rows = len(df)

# All three future targets must exist
df = df.dropna(subset=TARGET_COLUMNS).copy()

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

FEATURE_COLUMNS = [
    column
    for column in df.columns
    if column not in DROP_FROM_FEATURES
]

df = df.sort_values("record_id").reset_index(drop=True)


# ---------------------------------------------------------
# Chronological split by continuous sequences
# ---------------------------------------------------------

train_df = df[df["sequence_id"].isin(TRAIN_SEQUENCES)].copy()
val_df = df[df["sequence_id"].isin(VAL_SEQUENCES)].copy()
# test_df = df[df["sequence_id"].isin(TEST_SEQUENCES)].copy()
test_df = (
    pd.read_csv(
        "data/processed/incubator_stage4_test_normalized.csv"
    )
    .dropna(subset=TARGET_COLUMNS)
    .sort_values("record_id")
    .reset_index(drop=True)
)

X_train = train_df[FEATURE_COLUMNS].copy()
Y_train = train_df[TARGET_COLUMNS].copy()

X_val = val_df[FEATURE_COLUMNS].copy()
Y_val = val_df[TARGET_COLUMNS].copy()

X_test = test_df[FEATURE_COLUMNS].copy()
Y_test = test_df[TARGET_COLUMNS].copy()


# ---------------------------------------------------------
# Stage 4 data summary
# ---------------------------------------------------------

data_summary = pd.DataFrame([
    {"dataset": "Full Stage 3", "rows": original_rows, "sequences": "All"},
    {"dataset": "Usable", "rows": len(df), "sequences": "Valid targets"},
    {"dataset": "Train", "rows": len(train_df), "sequences": str(TRAIN_SEQUENCES)},
    {"dataset": "Validation", "rows": len(val_df), "sequences": str(VAL_SEQUENCES)},
    {"dataset": "Test", "rows": len(test_df), "sequences": str(TEST_SEQUENCES)},
])

data_summary["features"] = len(FEATURE_COLUMNS)
data_summary["outputs"] = len(TARGET_COLUMNS)


# ---------------------------------------------------------
# Class distribution report
# ---------------------------------------------------------

class_distribution_rows = []

for split_name, y_data in [
    ("Train", Y_train),
    ("Validation", Y_val),
    ("Test", Y_test),
]:
    for target in TARGET_COLUMNS:
        counts = (
            y_data[target]
            .value_counts()
            .reindex(CLASS_ORDER, fill_value=0)
        )

        percentages = (
            y_data[target]
            .value_counts(normalize=True)
            .reindex(CLASS_ORDER, fill_value=0)
            .mul(100)
        )

        if (counts == 0).any():
            missing = counts[counts == 0].index.tolist()
            raise ValueError(
                f"{split_name} - {target} is missing classes: {missing}"
            )

        class_distribution_rows.append({
            "split": split_name,
            "horizon": HORIZON_NAMES[target],
            "LOW_n": counts["LOW"],
            "NORMAL_n": counts["NORMAL"],
            "HIGH_n": counts["HIGH"],
            "LOW_%": percentages["LOW"],
            "NORMAL_%": percentages["NORMAL"],
            "HIGH_%": percentages["HIGH"],
        })

class_distribution_report = pd.DataFrame(class_distribution_rows)


# ---------------------------------------------------------
# Feature types
# ---------------------------------------------------------

NUMERIC_FEATURES = (
    X_train
    .select_dtypes(include="number")
    .columns
    .tolist()
)

CATEGORICAL_FEATURES = (
    X_train
    .select_dtypes(exclude="number")
    .columns
    .tolist()
)


# ---------------------------------------------------------
# Logistic Regression preprocessing
# ---------------------------------------------------------

logistic_numeric = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

logistic_categorical = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore")),
])

logistic_preprocessor = ColumnTransformer([
    ("numeric", logistic_numeric, NUMERIC_FEATURES),
    ("categorical", logistic_categorical, CATEGORICAL_FEATURES),
])


# ---------------------------------------------------------
# Tree-model preprocessing
# ---------------------------------------------------------

tree_numeric = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
])

tree_categorical = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore")),
])

tree_preprocessor = ColumnTransformer([
    ("numeric", tree_numeric, NUMERIC_FEATURES),
    ("categorical", tree_categorical, CATEGORICAL_FEATURES),
])


# ---------------------------------------------------------
# Tuned Logistic Regression settings
# ---------------------------------------------------------

LOGISTIC_CONFIG = {
    "temperature_risk_5m": {"C": 0.1},
    "temperature_risk_10m": {"C": 0.01},
    "temperature_risk_15m": {"C": 0.01},
}

logistic_models = {}

for target in TARGET_COLUMNS:
    model = Pipeline([
        ("preprocessor", clone(logistic_preprocessor)),
        (
            "classifier",
            LogisticRegression(
                C=LOGISTIC_CONFIG[target]["C"],
                solver="lbfgs",
                class_weight="balanced",
                max_iter=1000,
            ),
        ),
    ])

    model.fit(X_train, Y_train[target])
    logistic_models[target] = model


# ---------------------------------------------------------
# Tuned Random Forest settings
# ---------------------------------------------------------

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

rf_models = {}

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

    model.fit(X_train, Y_train[target])
    rf_models[target] = model


# ---------------------------------------------------------
# Tuned XGBoost settings
# ---------------------------------------------------------

XGB_CONFIG = {
    "temperature_risk_5m": {"max_depth": 4},
    "temperature_risk_10m": {"max_depth": 4},
    "temperature_risk_15m": {"max_depth": 2},
}

xgb_preprocessor = clone(tree_preprocessor)
X_train_xgb = xgb_preprocessor.fit_transform(X_train)
X_val_xgb = xgb_preprocessor.transform(X_val)

CLASS_TO_INT = {
    "LOW": 0,
    "NORMAL": 1,
    "HIGH": 2,
}

INT_TO_CLASS = {
    0: "LOW",
    1: "NORMAL",
    2: "HIGH",
}

xgb_models = {}
xgb_val_predictions = {}
xgb_best_iterations = {}

for target in TARGET_COLUMNS:
    config = XGB_CONFIG[target]

    y_train_encoded = Y_train[target].map(CLASS_TO_INT)
    y_val_encoded = Y_val[target].map(CLASS_TO_INT)

    train_weights = compute_sample_weight(
        class_weight="balanced",
        y=Y_train[target],
    )

    val_weights = compute_sample_weight(
        class_weight="balanced",
        y=Y_val[target],
    )

    model = xgb.XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=config["max_depth"],
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="mlogloss",
        early_stopping_rounds=30,
        tree_method="hist",
        n_jobs=-1,
        random_state=42,
    )

    model.fit(
        X_train_xgb,
        y_train_encoded,
        sample_weight=train_weights,
        eval_set=[(X_val_xgb, y_val_encoded)],
        sample_weight_eval_set=[val_weights],
        verbose=False,
    )

    xgb_models[target] = model
    xgb_best_iterations[target] = model.best_iteration

    encoded_prediction = model.predict(X_val_xgb).astype(int)

    xgb_val_predictions[target] = np.array([
        INT_TO_CLASS[value]
        for value in encoded_prediction
    ])


# ---------------------------------------------------------
# Evaluation
# ---------------------------------------------------------

def evaluate_predictions(
    model_name,
    target,
    y_true,
    y_pred,
):
    precision = precision_score(
        y_true,
        y_pred,
        labels=CLASS_ORDER,
        average=None,
        zero_division=0,
    )

    recall = recall_score(
        y_true,
        y_pred,
        labels=CLASS_ORDER,
        average=None,
        zero_division=0,
    )

    class_f1 = f1_score(
        y_true,
        y_pred,
        labels=CLASS_ORDER,
        average=None,
        zero_division=0,
    )

    return {
        "model": model_name,
        "horizon": HORIZON_NAMES[target],
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        ),
        "P_LOW": precision[0],
        "R_LOW": recall[0],
        "F1_LOW": class_f1[0],
        "R_NORMAL": recall[1],
        "P_HIGH": precision[2],
        "R_HIGH": recall[2],
        "F1_HIGH": class_f1[2],
    }


# ---------------------------------------------------------
# Final validation comparison
# ---------------------------------------------------------

validation_rows = []

for target in TARGET_COLUMNS:
    logistic_pred = logistic_models[target].predict(X_val)
    validation_rows.append(
        evaluate_predictions(
            "Logistic Regression",
            target,
            Y_val[target],
            logistic_pred,
        )
    )

    rf_pred = rf_models[target].predict(X_val)
    validation_rows.append(
        evaluate_predictions(
            "Random Forest",
            target,
            Y_val[target],
            rf_pred,
        )
    )

    validation_rows.append(
        evaluate_predictions(
            "XGBoost",
            target,
            Y_val[target],
            xgb_val_predictions[target],
        )
    )

validation_report = pd.DataFrame(validation_rows)


# ---------------------------------------------------------
# Model configuration report
# ---------------------------------------------------------

configuration_rows = []

for target in TARGET_COLUMNS:
    configuration_rows.append({
        "model": "Logistic Regression",
        "horizon": HORIZON_NAMES[target],
        "configuration": f"C={LOGISTIC_CONFIG[target]['C']}",
    })

    configuration_rows.append({
        "model": "Random Forest",
        "horizon": HORIZON_NAMES[target],
        "configuration": (
            f"depth={RF_CONFIG[target]['max_depth']}, "
            f"leaf={RF_CONFIG[target]['min_samples_leaf']}, "
            f"trees=200"
        ),
    })

    configuration_rows.append({
        "model": "XGBoost",
        "horizon": HORIZON_NAMES[target],
        "configuration": (
            f"depth={XGB_CONFIG[target]['max_depth']}, "
            f"best_iteration={xgb_best_iterations[target]}"
        ),
    })

configuration_report = pd.DataFrame(configuration_rows)


# ---------------------------------------------------------
# Save Stage 4 reports
# ---------------------------------------------------------

data_summary.to_csv(
    REPORT_DIR / "data_summary.csv",
    index=False,
)

class_distribution_report.to_csv(
    REPORT_DIR / "class_distribution.csv",
    index=False,
)

configuration_report.to_csv(
    REPORT_DIR / "model_configurations.csv",
    index=False,
)

validation_report.to_csv(
    REPORT_DIR / "validation_comparison.csv",
    index=False,
)


# ---------------------------------------------------------
# Console summary
# ---------------------------------------------------------

print("\n=== Stage 4 Data Summary ===")
print(data_summary.to_string(index=False))

print("\n=== Final Validation Comparison ===")
print(validation_report.round(3).to_string(index=False))

print(f"\nStage 4 reports saved to: {REPORT_DIR}")
print("\nTEST SET HAS NOT BEEN USED.")


# =========================================================
# Final model training
# =========================================================

final_train_df = pd.concat(
    [train_df, val_df],
    ignore_index=True,
)

X_final_train = final_train_df[FEATURE_COLUMNS].copy()
Y_final_train = final_train_df[TARGET_COLUMNS].copy()

# ---------------------------------------------------------
# Retrain selected models on Train + Validation
# ---------------------------------------------------------

final_rf_models = {}

for target in TARGET_COLUMNS:

    config = RF_CONFIG[target]

    model = Pipeline([
        (
            "preprocessor",
            clone(tree_preprocessor),
        ),
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

    model.fit(
        X_final_train,
        Y_final_train[target],
    )

    final_rf_models[target] = model

# ---------------------------------------------------------
# Final test evaluation
# ---------------------------------------------------------

test_rows = []

for target in TARGET_COLUMNS:

    y_pred = final_rf_models[target].predict(
        X_test
    )

    test_rows.append(
        evaluate_predictions(
            "Random Forest",
            target,
            Y_test[target],
            y_pred,
        )
    )

final_test_report = pd.DataFrame(
    test_rows
)

final_test_report.to_csv(
    REPORT_DIR / "final_test_results.csv",
    index=False,
)

print("\n=== FINAL TEST RESULTS ===")
print(
    final_test_report
    .round(3)
    .to_string(index=False)
)

