import pandas as pd

# Source
PATH_SRC = "data/processed/report_oop.csv"
PATH_OUT = "data/ml_ready/df_ml_ready.csv"

df = pd.read_csv(PATH_SRC)

# Parse dates
df["last_activity_date"] = pd.to_datetime(df["last_activity_date"], errors="coerce")

# Date de référence (dataset figé)
date_ref = df["last_activity_date"].max()

# Feature temporelle
df["anciennete_jours"] = (date_ref - df["last_activity_date"]).dt.days

# Sélection ML
features_num = [
    "nb_paiements",
    "actions_total",
    "sessions_total",
    "anciennete_jours",
]

features_cat = [
    "plan",
    "ville",
]

target = "ca_total"

df_ml = df[features_num + features_cat + [target]].copy()

# Sécurité types
for c in features_num + [target]:
    df_ml[c] = pd.to_numeric(df_ml[c], errors="coerce")

# Drop lignes invalides (strict)
df_ml = df_ml.dropna().reset_index(drop=True)

# Export
df_ml.to_csv(PATH_OUT, index=False)

print(f"ML-ready exporté : {PATH_OUT} | shape={df_ml.shape}")
