"""Minimal FastAPI service for incubator thermal-risk prediction."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


MODEL_DIR = Path("models/stage5")
METADATA_FILE = MODEL_DIR / "model_metadata.json"


def load_artifacts() -> tuple[dict[str, Any], dict[str, Any]]:
    """Load the saved metadata and three complete model pipelines."""

    if not METADATA_FILE.exists():
        raise RuntimeError(
            "Model metadata was not found. Run p5.py before starting the API."
        )

    with METADATA_FILE.open("r", encoding="utf-8") as file:
        metadata = json.load(file)

    models = {}
    for target, filename in metadata["model_files"].items():
        model_path = MODEL_DIR / filename
        if not model_path.exists():
            raise RuntimeError(
                f"Saved model was not found: {model_path}. Run p5.py again."
            )
        models[target] = joblib.load(model_path)

    return metadata, models


METADATA, MODELS = load_artifacts()


class PredictionRequest(BaseModel):
    """One prepared Stage 3 feature row."""

    features: dict[str, Any] = Field(
        ...,
        description=(
            "The complete prepared feature row. Use the example generated "
            "from the Stage 3 data and change values as needed."
        ),
        examples=[METADATA["example_features"]],
    )


class PredictionResponse(BaseModel):
    temperature_risk_5m: str
    temperature_risk_10m: str
    temperature_risk_15m: str


app = FastAPI(
    title="Incubator Thermal-Risk Prediction API",
    description=(
        "Predicts LOW, NORMAL, or HIGH temperature risk for the next "
        "5, 10, and 15 minutes using the selected Random Forest models."
    ),
    version=METADATA["model_version"],
)


@app.get("/health", tags=["System"])
def health() -> dict[str, Any]:
    """Confirm that the API and all model artifacts are ready."""

    return {
        "status": "ready",
        "model_type": METADATA["model_type"],
        "model_version": METADATA["model_version"],
        "loaded_models": len(MODELS),
        "feature_count": METADATA["feature_count"],
    }


def prepare_input(features: dict[str, Any]) -> pd.DataFrame:
    """Validate, order, and type one API feature row."""

    expected = METADATA["feature_columns"]
    expected_set = set(expected)
    received_set = set(features)

    missing = sorted(expected_set - received_set)
    unexpected = sorted(received_set - expected_set)

    if missing or unexpected:
        details = {}
        if missing:
            details["missing_features"] = missing
        if unexpected:
            details["unexpected_features"] = unexpected
        raise HTTPException(status_code=422, detail=details)

    row = pd.DataFrame(
        [{feature: features[feature] for feature in expected}],
        columns=expected,
    )

    for feature in METADATA["numeric_features"]:
        row[feature] = pd.to_numeric(row[feature], errors="coerce")

    for feature in METADATA["categorical_features"]:
        row[feature] = row[feature].replace({None: np.nan})

    return row


@app.post(
    "/predict",
    response_model=PredictionResponse,
    tags=["Prediction"],
)
def predict(request: PredictionRequest) -> PredictionResponse:
    """Predict thermal risk for the three approved horizons."""

    row = prepare_input(request.features)

    predictions = {
        target: str(model.predict(row)[0])
        for target, model in MODELS.items()
    }

    return PredictionResponse(**predictions)