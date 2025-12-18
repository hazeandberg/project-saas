from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# =========================
# Paths
# =========================
PATH_IN = "data/ml_ready/df_ml_ready.csv"
MODELS_DIR = Path("src/ml/models")
PATH_MODEL_BASELINE = MODELS_DIR / "model_baseline_v1.joblib"

METRICS_DIR = Path("data/ml_ready")
PATH_METRICS = METRICS_DIR / "metrics_v1.csv"


# =========================
# Contract (must match build_ml_ready.py output)
# =========================
TARGET = "ca_total"
NUM_FEATURES = ["nb_paiements", "actions_total", "sessions_total", "anciennete_jours"]
CAT_FEATURES = ["plan", "ville"]
ALL_FEATURES = NUM_FEATURES + CAT_FEATURES


def main() -> None:
    # --- Read ---
    print("IN :", PATH_IN)
    df = pd.read_csv(PATH_IN)
    print("shape_in :", df.shape)
    print("cols_in  :", df.columns.tolist())

    # --- Guards ---
    required = set(ALL_FEATURES + [TARGET])
    missing = sorted(required - set(df.columns))
    if missing:
        raise KeyError(f"Colonnes manquantes dans {PATH_IN}: {missing}")
    if df.empty:
        raise ValueError(f"Dataset vide: {PATH_IN}")

    # --- X / y (strict contract) ---
    X = df[ALL_FEATURES].copy()
    y = df[TARGET].copy()

    # --- Split ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print("X_train:", X_train.shape)
    print("X_test :", X_test.shape)
    print("y_train:", y_train.shape)
    print("y_test :", y_test.shape)

    # --- Preprocess ---
    preprocess = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUM_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_FEATURES),
        ],
        remainder="drop",
    )

    # --- Model + pipeline ---
    model = LinearRegression()
    pipeline = Pipeline(
        steps=[
            ("preprocess", preprocess),
            ("model", model),
        ]
    )

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    # --- Eval ---
    mae = mean_absolute_error(y_test, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))

    print(f"MAE  : {mae:.2f}")
    print(f"RMSE : {rmse:.2f}")

    # --- Ensure output dirs ---
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    # --- Save model ---
    joblib.dump(pipeline, PATH_MODEL_BASELINE)
    print("OUT model_baseline:", str(PATH_MODEL_BASELINE))

    # --- Append metrics (same file as other models) ---
    row = pd.DataFrame(
        [
            {
                "model": "LinearRegression",
                "target": TARGET,
                "MAE": float(mae),
                "RMSE": float(rmse),
                "n_rows": int(df.shape[0]),
                "n_features": int(len(ALL_FEATURES)),
            }
        ]
    )
    write_header = not PATH_METRICS.exists()
    row.to_csv(PATH_METRICS, mode="a", header=write_header, index=False)
    print("OUT metrics:", str(PATH_METRICS))


if __name__ == "__main__":
    main()
