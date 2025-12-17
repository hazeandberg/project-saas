import pandas as pd

# =========================
# Paths
# =========================
REPORT_PATH = "data/processed/report_oop.csv"
SUBS_PATH = "data/raw/subscriptions.csv"
PATH_OUT = "data/ml_ready/df_ml_ready.csv"

# =========================
# Paramètres figés (Semaine 2)
# =========================
T = pd.Timestamp("2025-11-01")
BUFFER_DAYS = 7
HORIZON_DAYS = 30

# =========================
# Load
# =========================
report = pd.read_csv(REPORT_PATH)
subs = pd.read_csv(SUBS_PATH)

# Parse dates
subs["date_paiement"] = pd.to_datetime(subs["date_paiement"], errors="coerce")

# =========================
# Garde-fou couverture temporelle
# =========================
max_payment = subs["date_paiement"].max()
required = T + pd.Timedelta(days=HORIZON_DAYS)
if max_payment < required:
    raise ValueError(
        f"Couverture insuffisante: max(date_paiement)={max_payment.date()} < "
        f"T+{HORIZON_DAYS}j={required.date()}"
    )

# =========================
# Features subscriptions-only (pré-T)
# =========================
subs_pre_T = subs[subs["date_paiement"] < T].copy()

agg = subs_pre_T.groupby("client_id", as_index=False).agg(
    paid_count_before_T=("date_paiement", "count"),
    paid_sum_before_T=("montant", "sum"),
    last_paid_before_T=("date_paiement", "max"),
)

agg["days_since_last_paid"] = (T - agg["last_paid_before_T"]).dt.days

# =========================
# Label churn_7_30j (paiement-based)
# Fenêtre: [T+7, T+30] (borne haute incluse)
# =========================
start = T + pd.Timedelta(days=BUFFER_DAYS)
end = T + pd.Timedelta(days=HORIZON_DAYS)

paid_7_30 = subs[
    (subs["statut"] == "paid")
    & (subs["date_paiement"] >= start)
    & (subs["date_paiement"] <= end)
]["client_id"].unique()

agg["churn_7_30j"] = (~agg["client_id"].isin(paid_7_30)).astype(int)

# =========================
# Attributs stables (plan, ville)
# =========================
attrs = report[["client_id", "plan", "ville"]].drop_duplicates("client_id")

df_ml = agg.merge(attrs, on="client_id", how="left")

# =========================
# Sécurité types
# =========================
num_cols = [
    "paid_count_before_T",
    "paid_sum_before_T",
    "days_since_last_paid",
]
for c in num_cols:
    df_ml[c] = pd.to_numeric(df_ml[c], errors="coerce")

df_ml["churn_7_30j"] = df_ml["churn_7_30j"].astype(int)

# =========================
# Nettoyage final
# =========================
df_ml = df_ml.dropna().reset_index(drop=True)

# =========================
# Export
# =========================
df_ml.to_csv(PATH_OUT, index=False)

print(f"T (ref_date)       : {T.date()}")
print(f"max(date_paiement) : {max_payment.date()}")
print(f"Export             : {PATH_OUT} | shape={df_ml.shape}")
print("Distribution churn_7_30j:")
print(df_ml["churn_7_30j"].value_counts())
