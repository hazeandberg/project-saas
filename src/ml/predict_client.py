import sys
import pandas as pd
import joblib


# =========================
# Paths
# =========================
MODEL_PATH = "src/ml/models/model_current_v1.joblib"
REPORT_PATH = "data/processed/report_oop.csv"


# =========================
# Contract (must match build_ml_ready.py output)
# =========================
FEATURE_COLS = [
    "nb_paiements",
    "actions_total",
    "sessions_total",
    "anciennete_jours",
    "plan",
    "ville",
]

TARGET_COL = "ca_total"  # optional for "actual"
ID_COL = "client_id"
DATE_COL = "last_activity_date"


def add_anciennete_jours(report_df: pd.DataFrame) -> pd.DataFrame:
    df = report_df.copy()

    # Guard: date column exists
    if DATE_COL not in df.columns:
        raise KeyError(f"Colonne manquante dans report: {DATE_COL}")

    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    ref_date = df[DATE_COL].max()

    if pd.isna(ref_date):
        raise ValueError(
            f"Impossible de calculer ref_date: toutes les valeurs de {DATE_COL} sont NaT après parsing."
        )

    df["anciennete_jours"] = (ref_date - df[DATE_COL]).dt.days
    return df


def build_features_for_client(report_df: pd.DataFrame, client_id: str) -> pd.DataFrame:
    # Guard: id column exists
    if ID_COL not in report_df.columns:
        raise KeyError(f"Colonne manquante dans report: {ID_COL}")

    row = report_df.loc[report_df[ID_COL] == client_id]
    if row.empty:
        raise ValueError(f"{ID_COL} not found: {client_id}")

    # Guard: all feature columns exist
    missing = sorted(set(FEATURE_COLS) - set(report_df.columns))
    if missing:
        raise KeyError(f"Colonnes features manquantes dans {REPORT_PATH}: {missing}")

    # One row DF, in the exact order
    X_one = row.iloc[[0]][FEATURE_COLS].copy()
    return X_one


def main() -> None:
    client_id = sys.argv[1] if len(sys.argv) > 1 else "C008"

    print("IN model :", MODEL_PATH)
    print("IN report:", REPORT_PATH)
    print("client_id:", client_id)

    report = pd.read_csv(REPORT_PATH)
    print("shape_report:", report.shape)
    print("cols_report :", report.columns.tolist())

    report = add_anciennete_jours(report)

    model = joblib.load(MODEL_PATH)

    X_one = build_features_for_client(report, client_id)
    print("X_one shape:", X_one.shape)
    print("X_one cols :", X_one.columns.tolist())
    print("X_one row  :", X_one.iloc[0].to_dict())

    pred = float(model.predict(X_one)[0])

    # actual is optional
    if TARGET_COL in report.columns:
        actual_series = report.loc[report[ID_COL] == client_id, TARGET_COL]
        actual = float(actual_series.iloc[0]) if not actual_series.empty else None
        if actual is not None:
            print(f"Client {client_id} -> pred: {pred:.2f} | actual: {actual:.2f}")
            return

    print(f"Client {client_id} -> pred: {pred:.2f} | actual: N/A (target absente)")


if __name__ == "__main__":
    main()
