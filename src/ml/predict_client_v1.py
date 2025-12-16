import pandas as pd
import joblib
import sys

MODEL_PATH = "src/ml/models/model_current_v1.joblib"
REPORT_PATH = "data/processed/report_oop.csv"

# 🔧 Mets ici EXACTEMENT les colonnes features utilisées pour entraîner le modèle
FEATURE_COLS = [
    "nb_paiements",
    "actions_total",
    "sessions_total",
    "anciennete_jours",
    "plan",
    "ville",
]

TARGET_COL = "ca_total"
ID_COL = "client_id"


def add_anciennete_jours(report_df: pd.DataFrame) -> pd.DataFrame:
    df = report_df.copy()
    df["last_activity_date"] = pd.to_datetime(df["last_activity_date"])
    ref_date = df["last_activity_date"].max()
    df["anciennete_jours"] = (ref_date - df["last_activity_date"]).dt.days
    return df


def build_features_for_client(report_df: pd.DataFrame, client_id: str) -> pd.DataFrame:
    row = report_df.loc[report_df[ID_COL] == client_id]
    if row.empty:
        raise ValueError(f"{ID_COL} not found: {client_id}")

    # On prend la 1ère occurrence (normalement unique)
    X_one = row.iloc[[0]][FEATURE_COLS]
    return X_one


def main():
    report = pd.read_csv(REPORT_PATH)
    report = add_anciennete_jours(report)
    model = joblib.load(MODEL_PATH)
    client_id = sys.argv[1] if len(sys.argv) > 1 else "C008"
    X_one = build_features_for_client(report, client_id)
    pred = model.predict(X_one)[0]

    actual = report.loc[report[ID_COL] == client_id, TARGET_COL].iloc[0]
    print(f"Client {client_id} -> pred: {pred:.2f} | actual: {actual:.2f}")


if __name__ == "__main__":
    main()
