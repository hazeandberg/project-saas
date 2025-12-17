from __future__ import annotations

from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field


MODEL_PATH = Path("src/ml/models/churn_model_v1.joblib")

app = FastAPI(title="SaaS ML API", version="v1")


class PredictIn(BaseModel):
    paid_count_before_T: int = Field(..., ge=0)
    paid_sum_before_T: float = Field(..., ge=0)
    days_since_last_paid: int = Field(..., ge=0)
    plan: Literal["free", "basic", "pro"]
    ville: str = Field(..., min_length=1)


class PredictOut(BaseModel):
    churn_probability: float
    churn: bool


_model = None


def get_model():
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model not found at: {MODEL_PATH}")
        _model = joblib.load(MODEL_PATH)
    return _model


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictOut)
def predict(payload: PredictIn):
    model = get_model()

    X = pd.DataFrame(
        [
            {
                "paid_count_before_T": payload.paid_count_before_T,
                "paid_sum_before_T": payload.paid_sum_before_T,
                "days_since_last_paid": payload.days_since_last_paid,
                "plan": payload.plan,
                "ville": payload.ville,
            }
        ]
    )

    proba = float(model.predict_proba(X)[0][1])
    pred = bool(model.predict(X)[0])

    return {"churn_probability": proba, "churn": pred}
