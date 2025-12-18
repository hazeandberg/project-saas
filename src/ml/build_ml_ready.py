import os
import pandas as pd


# =========================
# Paths
# =========================
PATH_SRC = "data/processed/report_oop.csv"
PATH_OUT = "data/ml_ready/df_ml_ready.csv"


# =========================
# Config colonnes (contrat)
# =========================
FEATURES_NUM = [
    "nb_paiements",
    "actions_total",
    "sessions_total",
]

FEATURES_CAT = [
    "plan",
    "ville",
]

DATE_COL = "last_activity_date"
TARGET = "ca_total"


def main() -> None:
    # --- Read ---
    print("IN :", PATH_SRC)
    df = pd.read_csv(PATH_SRC)
    print("shape_in :", df.shape)
    print("cols_in  :", df.columns.tolist())

    # --- Guard: required columns ---
    required = set(FEATURES_NUM + FEATURES_CAT + [DATE_COL, TARGET])
    missing = sorted(required - set(df.columns))
    if missing:
        raise KeyError(f"Colonnes manquantes dans {PATH_SRC}: {missing}")

    # --- Parse dates ---
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")

    # Date de référence (dataset figé)
    date_ref = df[DATE_COL].max()
    if pd.isna(date_ref):
        raise ValueError(
            f"Impossible de calculer date_ref: toutes les valeurs de {DATE_COL} sont NaT après parsing."
        )

    # Feature temporelle
    df["anciennete_jours"] = (date_ref - df[DATE_COL]).dt.days

    # --- Build ML selection ---
    cols_out = FEATURES_NUM + FEATURES_CAT + ["anciennete_jours", TARGET]
    df_ml = df[cols_out].copy()

    # --- Type safety (num + target + engineered num) ---
    for c in FEATURES_NUM + ["anciennete_jours", TARGET]:
        df_ml[c] = pd.to_numeric(df_ml[c], errors="coerce")

    # --- Drop invalid rows (strict) ---
    df_ml = df_ml.dropna().reset_index(drop=True)
    if df_ml.empty:
        raise ValueError(
            "df_ml est vide après dropna(). Vérifie les dates (last_activity_date) "
            "et les colonnes numériques/target (coercion en NaN)."
        )

    # --- Ensure output dir exists ---
    out_dir = os.path.dirname(PATH_OUT)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # --- Export ---
    print("OUT:", PATH_OUT)
    print("shape_out:", df_ml.shape)
    print("cols_out :", df_ml.columns.tolist())
    df_ml.to_csv(PATH_OUT, index=False)

    print(f"ML-ready exporté : {PATH_OUT} | shape={df_ml.shape}")


if __name__ == "__main__":
    main()
