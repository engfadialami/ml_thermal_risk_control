import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    recall_score,
)
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import xgboost as xgb
from sklearn.base import clone
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import precision_score

INPUT_FILE = "data\processed\incubator_stage3_targets.csv"

df = pd.read_csv(INPUT_FILE)

print(f"Dataset shape: {df.shape}")

TARGET_COLUMNS = [
    "temperature_risk_5m",
    "temperature_risk_10m",
    "temperature_risk_15m",
]

# All three future targets must exist
df = df.dropna(subset=TARGET_COLUMNS).copy()

DROP_FROM_FEATURES = [
    "record_id",
    "sequence_id",
    "estimated_timestamp",
    "target_segment_id",

    # Future temperature values — DATA LEAKAGE if used as inputs
    "target_avg_t_5m",
    "target_avg_t_10m",
    "target_avg_t_15m",

    # Outputs
    "temperature_risk_5m",
    "temperature_risk_10m",
    "temperature_risk_15m",
]

X = df.drop(columns=DROP_FROM_FEATURES)
Y = df[TARGET_COLUMNS]

print(f"Usable rows: {len(df)}")
print(f"Number of input features: {X.shape[1]}")
print(f"Number of outputs: {Y.shape[1]}")

print("\nTarget class distributions:")

for target in TARGET_COLUMNS:
    print(f"\n{target}")
    print(Y[target].value_counts())
    print((Y[target].value_counts(normalize=True) * 100).round(2))

# ---------------------------------------------------------
# Chronological split by continuous sequences
# ---------------------------------------------------------

TRAIN_SEQUENCES = [1, 2, 3, 5]
VAL_SEQUENCES = [7]
TEST_SEQUENCES = [8, 9]

# Keep acquisition order
df = df.sort_values("record_id").reset_index(drop=True)

train_df = df[df["sequence_id"].isin(TRAIN_SEQUENCES)].copy()
val_df = df[df["sequence_id"].isin(VAL_SEQUENCES)].copy()
test_df = df[df["sequence_id"].isin(TEST_SEQUENCES)].copy()

print("\nSequence-based split:")
print(f"Train:      {len(train_df)} rows - sequences {TRAIN_SEQUENCES}")
print(f"Validation: {len(val_df)} rows - sequences {VAL_SEQUENCES}")
print(f"Test:       {len(test_df)} rows - sequences {TEST_SEQUENCES}")

# ---------------------------------------------------------
# Build Train / Validation / Test feature and target sets
# ---------------------------------------------------------

FEATURE_COLUMNS = X.columns.tolist()

X_train = train_df[FEATURE_COLUMNS].copy()
Y_train = train_df[TARGET_COLUMNS].copy()

X_val = val_df[FEATURE_COLUMNS].copy()
Y_val = val_df[TARGET_COLUMNS].copy()

X_test = test_df[FEATURE_COLUMNS].copy()
Y_test = test_df[TARGET_COLUMNS].copy()

print("\nML dataset shapes:")
print(f"X_train: {X_train.shape} | Y_train: {Y_train.shape}")
print(f"X_val:   {X_val.shape} | Y_val:   {Y_val.shape}")
print(f"X_test:  {X_test.shape} | Y_test:  {Y_test.shape}")


# ---------------------------------------------------------
# Check class distributions in each split
# ---------------------------------------------------------

EXPECTED_CLASSES = {"LOW", "NORMAL", "HIGH"}

splits = {
    "TRAIN": Y_train,
    "VALIDATION": Y_val,
    "TEST": Y_test,
}

for split_name, y_split in splits.items():

    print(f"\n--- {split_name} ---")

    for target in TARGET_COLUMNS:

        counts = y_split[target].value_counts()
        percentages = (
            y_split[target]
            .value_counts(normalize=True)
            .mul(100)
            .round(2)
        )

        print(f"\n{target}")
        print(counts)
        print("Percentages:")
        print(percentages)

        missing_classes = EXPECTED_CLASSES - set(counts.index)

        if missing_classes:
            raise ValueError(
                f"{split_name} - {target} missing classes: "
                f"{missing_classes}"
            )

# ---------------------------------------------------------
# Preprocessing for Logistic Regression
# ---------------------------------------------------------

NUMERIC_FEATURES = X_train.select_dtypes(include="number").columns.tolist()
CATEGORICAL_FEATURES = X_train.select_dtypes(exclude="number").columns.tolist()

numeric_preprocessor = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

categorical_preprocessor = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore")),
])

logistic_preprocessor = ColumnTransformer([
    ("numeric", numeric_preprocessor, NUMERIC_FEATURES),
    ("categorical", categorical_preprocessor, CATEGORICAL_FEATURES),
])


# ---------------------------------------------------------
# Multinomial Logistic Regression
# ---------------------------------------------------------

base_logistic = LogisticRegression(
    solver="lbfgs",
    class_weight="balanced",
    max_iter=1000,
)

logistic_model = Pipeline([
    ("preprocessor", logistic_preprocessor),
    ("classifier", MultiOutputClassifier(base_logistic)),
])

logistic_model.fit(X_train, Y_train)

# ---------------------------------------------------------
# Logistic Regression validation
# ---------------------------------------------------------

Y_val_pred = logistic_model.predict(X_val)

validation_results = []

for i, target in enumerate(TARGET_COLUMNS):

    y_true = Y_val[target]
    y_pred = Y_val_pred[:, i]

    recalls = recall_score(
        y_true,
        y_pred,
        labels=["LOW", "NORMAL", "HIGH"],
        average=None,
        zero_division=0,
    )

    validation_results.append({
        "target": target,
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        ),
        "recall_LOW": recalls[0],
        "recall_NORMAL": recalls[1],
        "recall_HIGH": recalls[2],
    })

logistic_validation_report = pd.DataFrame(validation_results)

print("\nLogistic Regression - Validation Report")
print(
    logistic_validation_report
    .round(3)
    .to_string(index=False)
)

# ---------------------------------------------------------
# Logistic Regression confusion matrices
# ---------------------------------------------------------

CLASS_ORDER = ["LOW", "NORMAL", "HIGH"]

for i, target in enumerate(TARGET_COLUMNS):

    cm = confusion_matrix(
        Y_val[target],
        Y_val_pred[:, i],
        labels=CLASS_ORDER,
    )

    cm_df = pd.DataFrame(
        cm,
        index=[f"Actual_{c}" for c in CLASS_ORDER],
        columns=[f"Pred_{c}" for c in CLASS_ORDER],
    )

    cm_pct = (
        cm
        / cm.sum(axis=1, keepdims=True)
        * 100
    )

    cm_pct_df = pd.DataFrame(
        cm_pct,
        index=[f"Actual_{c}" for c in CLASS_ORDER],
        columns=[f"Pred_{c}" for c in CLASS_ORDER],
    )

    print("Row percentages:")
    print(cm_pct_df.round(1))

    print(f"\n{target}")
    print(cm_df)

    # ---------------------------------------------------------
# Preprocessing for Random Forest
# ---------------------------------------------------------

rf_numeric_preprocessor = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
])

rf_categorical_preprocessor = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore")),
])

rf_preprocessor = ColumnTransformer([
    ("numeric", rf_numeric_preprocessor, NUMERIC_FEATURES),
    ("categorical", rf_categorical_preprocessor, CATEGORICAL_FEATURES),
])

# ---------------------------------------------------------
# Random Forest
# ---------------------------------------------------------

base_rf = RandomForestClassifier(
    n_estimators=200,
    max_features="sqrt",
    class_weight="balanced",
    n_jobs=-1,
    random_state=42,
)

rf_model = Pipeline([
    ("preprocessor", rf_preprocessor),
    ("classifier", MultiOutputClassifier(base_rf)),
])

rf_model.fit(X_train, Y_train)

# ---------------------------------------------------------
# Compact classification evaluation
# ---------------------------------------------------------

def evaluate_multiclass_model(model, X_data, Y_data):

    predictions = model.predict(X_data)
    results = []

    for i, target in enumerate(TARGET_COLUMNS):

        y_true = Y_data[target]
        y_pred = predictions[:, i]

        recalls = recall_score(
            y_true,
            y_pred,
            labels=["LOW", "NORMAL", "HIGH"],
            average=None,
            zero_division=0,
        )

        results.append({
            "target": target,
            "accuracy": accuracy_score(y_true, y_pred),
            "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
            "macro_f1": f1_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0,
            ),
            "recall_LOW": recalls[0],
            "recall_NORMAL": recalls[1],
            "recall_HIGH": recalls[2],
        })

    return pd.DataFrame(results)

rf_validation_report = evaluate_multiclass_model(
    rf_model,
    X_val,
    Y_val,
)

print("\nRandom Forest - Validation Report")
print(
    rf_validation_report
    .round(3)
    .to_string(index=False)
)

rf_train_report = evaluate_multiclass_model(
    rf_model,
    X_train,
    Y_train,
)

rf_fit_check = pd.DataFrame({
    "target": TARGET_COLUMNS,

    "train_bal_acc":
        rf_train_report["balanced_accuracy"],

    "val_bal_acc":
        rf_validation_report["balanced_accuracy"],

    "train_macro_f1":
        rf_train_report["macro_f1"],

    "val_macro_f1":
        rf_validation_report["macro_f1"],
})

print("\nRandom Forest - Fit Check")
print(
    rf_fit_check
    .round(3)
    .to_string(index=False)
)



# ---------------------------------------------------------
# Random Forest with limited depth
# ---------------------------------------------------------

base_rf_d6 = RandomForestClassifier(
    n_estimators=200,
    max_features="sqrt",
    max_depth=6,
    class_weight="balanced",
    n_jobs=-1,
    random_state=42,
)

rf_model_d6 = Pipeline([
    ("preprocessor", rf_preprocessor),
    ("classifier", MultiOutputClassifier(base_rf_d6)),
])

rf_model_d6.fit(X_train, Y_train)

rf_d6_train_report = evaluate_multiclass_model(
    rf_model_d6,
    X_train,
    Y_train,
)

rf_d6_validation_report = evaluate_multiclass_model(
    rf_model_d6,
    X_val,
    Y_val,
)

rf_depth_comparison = pd.DataFrame({
    "target": TARGET_COLUMNS,

    "baseline_train_bal":
        rf_train_report["balanced_accuracy"],

    "baseline_val_bal":
        rf_validation_report["balanced_accuracy"],

    "depth6_train_bal":
        rf_d6_train_report["balanced_accuracy"],

    "depth6_val_bal":
        rf_d6_validation_report["balanced_accuracy"],

    "depth6_macro_f1":
        rf_d6_validation_report["macro_f1"],

    "depth6_LOW":
        rf_d6_validation_report["recall_LOW"],

    "depth6_NORMAL":
        rf_d6_validation_report["recall_NORMAL"],

    "depth6_HIGH":
        rf_d6_validation_report["recall_HIGH"],
})

print("\nRandom Forest - Depth Comparison")
print(
    rf_depth_comparison
    .round(3)
    .to_string(index=False)
)

# ---------------------------------------------------------
# Preprocessing for XGBoost
# ---------------------------------------------------------

xgb_preprocessor = clone(rf_preprocessor)

X_train_xgb = xgb_preprocessor.fit_transform(X_train)
X_val_xgb = xgb_preprocessor.transform(X_val)

# ---------------------------------------------------------
# XGBoost models
# ---------------------------------------------------------

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
xgb_val_predictions = pd.DataFrame(index=Y_val.index)
xgb_best_iterations = {}

for target in TARGET_COLUMNS:

    y_train_encoded = Y_train[target].map(CLASS_TO_INT)
    y_val_encoded = Y_val[target].map(CLASS_TO_INT)

    balanced_weights = compute_sample_weight(
        class_weight="balanced",
        y=Y_train[target],
    )

    train_weights = np.sqrt(balanced_weights)

    val_weights = compute_sample_weight(
        class_weight="balanced",
        y=Y_val[target],
    )

    model = xgb.XGBClassifier(
        objective="multi:softprob",
        num_class=3,

        n_estimators=1000,
        learning_rate=0.05,
        max_depth=4,

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

        eval_set=[
            (X_val_xgb, y_val_encoded)
        ],

        sample_weight_eval_set=[
            val_weights
        ],

        verbose=False,
    )

    xgb_models[target] = model

    pred_encoded = model.predict(X_val_xgb).astype(int)

    xgb_val_predictions[target] = [
        INT_TO_CLASS[p] for p in pred_encoded
    ]

    xgb_best_iterations[target] = model.best_iteration

    # ---------------------------------------------------------
# XGBoost validation report
# ---------------------------------------------------------

xgb_results = []

for target in TARGET_COLUMNS:

    y_true = Y_val[target]
    y_pred = xgb_val_predictions[target]

    recalls = recall_score(
        y_true,
        y_pred,
        labels=["LOW", "NORMAL", "HIGH"],
        average=None,
        zero_division=0,
    )

    xgb_results.append({
        "target": target,

        "accuracy":
            accuracy_score(y_true, y_pred),

        "balanced_accuracy":
            balanced_accuracy_score(y_true, y_pred),

        "macro_f1":
            f1_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0,
            ),

        "recall_LOW": recalls[0],
        "recall_NORMAL": recalls[1],
        "recall_HIGH": recalls[2],

        "best_iteration":
            xgb_best_iterations[target],
    })

xgb_validation_report = pd.DataFrame(xgb_results)

print("\nXGBoost - Validation Report")
print(
    xgb_validation_report
    .round(3)
    .to_string(index=False)
)




# ---------------------------------------------------------
# ---------------   Tuning    -----------------------------
# ---------------------------------------------------------

# ---------------------------------------------------------
# Tuning 1 - Logistic Regression C
# ---------------------------------------------------------

logistic_c_reports = []

for c in [0.01, 0.1, 1.0, 10.0]:

    base_lr = LogisticRegression(
        C=c,
        solver="lbfgs",
        class_weight="balanced",
        max_iter=1000,
    )

    model = Pipeline([
        ("preprocessor", clone(logistic_preprocessor)),
        ("classifier", MultiOutputClassifier(base_lr)),
    ])

    model.fit(X_train, Y_train)

    report = evaluate_multiclass_model(
        model,
        X_val,
        Y_val,
    )

    report.insert(0, "C", c)
    logistic_c_reports.append(report)

logistic_c_report = pd.concat(
    logistic_c_reports,
    ignore_index=True,
)

print("\nTUNING REPORT 1 - Logistic Regression C")
print(logistic_c_report.round(3).to_string(index=False))

# ---------------------------------------------------------
# Tuning 2 - Random Forest max_depth
# ---------------------------------------------------------

rf_depth_reports = []

for depth in [4, 6, 8, 10]:

    base_rf_tune = RandomForestClassifier(
        n_estimators=200,
        max_depth=depth,
        max_features="sqrt",
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
    )

    model = Pipeline([
        ("preprocessor", clone(rf_preprocessor)),
        ("classifier", MultiOutputClassifier(base_rf_tune)),
    ])

    model.fit(X_train, Y_train)

    report = evaluate_multiclass_model(
        model,
        X_val,
        Y_val,
    )

    report.insert(0, "max_depth", depth)
    rf_depth_reports.append(report)

rf_depth_tuning_report = pd.concat(
    rf_depth_reports,
    ignore_index=True,
)

print("\nTUNING REPORT 2 - Random Forest max_depth")
print(rf_depth_tuning_report.round(3).to_string(index=False))

# ---------------------------------------------------------
# Tuning 3 - Random Forest min_samples_leaf
# ---------------------------------------------------------

rf_leaf_reports = []

for leaf in [1, 5, 10, 20]:

    base_rf_leaf = RandomForestClassifier(
        n_estimators=200,
        max_depth=6,
        min_samples_leaf=leaf,
        max_features="sqrt",
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
    )

    model = Pipeline([
        ("preprocessor", clone(rf_preprocessor)),
        ("classifier", MultiOutputClassifier(base_rf_leaf)),
    ])

    model.fit(X_train, Y_train)

    report = evaluate_multiclass_model(
        model,
        X_val,
        Y_val,
    )

    report.insert(0, "min_samples_leaf", leaf)
    rf_leaf_reports.append(report)

rf_leaf_tuning_report = pd.concat(
    rf_leaf_reports,
    ignore_index=True,
)

print("\nTUNING REPORT 3 - Random Forest min_samples_leaf")
print(rf_leaf_tuning_report.round(3).to_string(index=False))

# ---------------------------------------------------------
# Tuning 4 - XGBoost max_depth
# ---------------------------------------------------------

xgb_depth_rows = []

for depth in [2, 3, 4, 5]:

    for target in TARGET_COLUMNS:

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
            max_depth=depth,

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

            eval_set=[
                (X_val_xgb, y_val_encoded)
            ],

            sample_weight_eval_set=[
                val_weights
            ],

            verbose=False,
        )

        pred_encoded = model.predict(
            X_val_xgb
        ).astype(int)

        y_pred = pd.Series(
            [INT_TO_CLASS[p] for p in pred_encoded],
            index=Y_val.index,
        )

        recalls = recall_score(
            Y_val[target],
            y_pred,
            labels=["LOW", "NORMAL", "HIGH"],
            average=None,
            zero_division=0,
        )

        xgb_depth_rows.append({
            "max_depth": depth,
            "target": target,
            "balanced_accuracy":
                balanced_accuracy_score(
                    Y_val[target],
                    y_pred,
                ),
            "macro_f1":
                f1_score(
                    Y_val[target],
                    y_pred,
                    average="macro",
                    zero_division=0,
                ),
            "recall_LOW": recalls[0],
            "recall_NORMAL": recalls[1],
            "recall_HIGH": recalls[2],
            "best_iteration":
                model.best_iteration,
        })

xgb_depth_tuning_report = pd.DataFrame(
    xgb_depth_rows
)

print("\nTUNING REPORT 4 - XGBoost max_depth")
print(
    xgb_depth_tuning_report
    .round(3)
    .to_string(index=False)
)


# ---------------------------------------------------------
# Tuning 5 - Random oversampling + Random Forest
# ---------------------------------------------------------

from imblearn.over_sampling import RandomOverSampler

ros_preprocessor = clone(rf_preprocessor)

X_train_ros = ros_preprocessor.fit_transform(X_train)
X_val_ros = ros_preprocessor.transform(X_val)

ros_results = []

for target in TARGET_COLUMNS:

    ros = RandomOverSampler(
        random_state=42
    )

    X_resampled, y_resampled = ros.fit_resample(
        X_train_ros,
        Y_train[target],
    )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=6,
        max_features="sqrt",

        # Do not double-correct the imbalance
        class_weight=None,

        n_jobs=-1,
        random_state=42,
    )

    model.fit(
        X_resampled,
        y_resampled,
    )

    y_pred = model.predict(X_val_ros)

    recalls = recall_score(
        Y_val[target],
        y_pred,
        labels=["LOW", "NORMAL", "HIGH"],
        average=None,
        zero_division=0,
    )

    ros_results.append({
        "target": target,
        "balanced_accuracy":
            balanced_accuracy_score(
                Y_val[target],
                y_pred,
            ),
        "macro_f1":
            f1_score(
                Y_val[target],
                y_pred,
                average="macro",
                zero_division=0,
            ),
        "recall_LOW": recalls[0],
        "recall_NORMAL": recalls[1],
        "recall_HIGH": recalls[2],
    })

ros_tuning_report = pd.DataFrame(
    ros_results
)

print("\nTUNING REPORT 5 - Random Oversampling + RF")
print(
    ros_tuning_report
    .round(3)
    .to_string(index=False)
)


# ---------------------------------------------------------
# Tuning 6 - Random Forest depth + min_samples_leaf
# ---------------------------------------------------------

rf_grid_reports = []

for depth in [4, 6]:
    for leaf in [10, 20]:

        base_rf_grid = RandomForestClassifier(
            n_estimators=200,
            max_depth=depth,
            min_samples_leaf=leaf,
            max_features="sqrt",
            class_weight="balanced",
            n_jobs=-1,
            random_state=42,
        )

        model = Pipeline([
            ("preprocessor", clone(rf_preprocessor)),
            ("classifier", MultiOutputClassifier(base_rf_grid)),
        ])

        model.fit(X_train, Y_train)

        report = evaluate_multiclass_model(
            model,
            X_val,
            Y_val,
        )

        report.insert(0, "min_samples_leaf", leaf)
        report.insert(0, "max_depth", depth)

        rf_grid_reports.append(report)

rf_grid_tuning_report = pd.concat(
    rf_grid_reports,
    ignore_index=True,
)

print("\nTUNING REPORT 6 - Random Forest depth + min_samples_leaf")
print(
    rf_grid_tuning_report
    .round(3)
    .to_string(index=False)
)


# %%
# ---------------------------------------------------------
# Diagnostic report - Best Random Forest candidates
# ---------------------------------------------------------

RF_CANDIDATES = [
    {
        "name": "5m_depth4_leaf20",
        "target": "temperature_risk_5m",
        "max_depth": 4,
        "min_samples_leaf": 20,
    },
    {
        "name": "5m_depth6_leaf20",
        "target": "temperature_risk_5m",
        "max_depth": 6,
        "min_samples_leaf": 20,
    },
    {
        "name": "10m_depth6_leaf20",
        "target": "temperature_risk_10m",
        "max_depth": 6,
        "min_samples_leaf": 20,
    },
    {
        "name": "15m_depth6_leaf10",
        "target": "temperature_risk_15m",
        "max_depth": 6,
        "min_samples_leaf": 10,
    },
]

diagnostic_rows = []

CLASS_ORDER = ["LOW", "NORMAL", "HIGH"]

for candidate in RF_CANDIDATES:

    target = candidate["target"]

    rf_candidate = RandomForestClassifier(
        n_estimators=200,
        max_depth=candidate["max_depth"],
        min_samples_leaf=candidate["min_samples_leaf"],
        max_features="sqrt",
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
    )

    candidate_model = Pipeline([
        ("preprocessor", clone(rf_preprocessor)),
        ("classifier", rf_candidate),
    ])

    candidate_model.fit(
        X_train,
        Y_train[target],
    )

    y_true = Y_val[target]
    y_pred = candidate_model.predict(X_val)

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

    diagnostic_rows.append({
        "candidate": candidate["name"],

        "balanced_acc":
            balanced_accuracy_score(y_true, y_pred),

        "macro_f1":
            f1_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0,
            ),

        "P_LOW": precision[0],
        "R_LOW": recall[0],
        "F1_LOW": class_f1[0],

        "P_NORMAL": precision[1],
        "R_NORMAL": recall[1],
        "F1_NORMAL": class_f1[1],

        "P_HIGH": precision[2],
        "R_HIGH": recall[2],
        "F1_HIGH": class_f1[2],
    })


rf_diagnostic_report = pd.DataFrame(
    diagnostic_rows
)

print("\nRandom Forest - Candidate Diagnostic Report")
print(
    rf_diagnostic_report
    .round(3)
    .to_string(index=False)
)

