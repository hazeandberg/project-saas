import pandas as pd
import joblib


# =========================
# Paths
# =========================
MODEL_PATH = "src/ml/models/model_current_v1.joblib"
DATA_PATH = "data/ml_ready/df_ml_ready.csv"


# =========================
# Contract (must match build_ml_ready.py output)
# =========================
TARGET = "ca_total"
NUM_FEATURES = ["nb_paiements", "actions_total", "sessions_total", "anciennete_jours"]
CAT_FEATURES = ["plan", "ville"]
ALL_FEATURES = NUM_FEATURES + CAT_FEATURES


def main() -> None:
    print("IN model:", MODEL_PATH)
    print("IN data :", DATA_PATH)

    df = pd.read_csv(DATA_PATH)
    print("shape_in:", df.shape)
    print("cols_in :", df.columns.tolist())

    # Guard: required feature columns
    missing = sorted(set(ALL_FEATURES) - set(df.columns))
    if missing:
        raise KeyError(f"Colonnes features manquantes dans {DATA_PATH}: {missing}")

    # Build X strictly on contract
    X = df[ALL_FEATURES].copy()

    model = joblib.load(MODEL_PATH)

    pred = model.predict(X)

    print("Pred shape:", pred.shape)
    print("First 5 predictions:", pred[:5])

    # Optional sanity check if target exists (dev-only)
    if TARGET in df.columns:
        y = df[TARGET].values
        print("First 5 actual:", y[:5])
    else:
        print("Note: target non présente dans le dataset (mode prod OK).")


if __name__ == "__main__":
    main()
