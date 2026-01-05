# src/dashboards/app_e2_health_overview.py
import pandas as pd
import streamlit as st

st.header("E2 — Santé globale")
st.caption("E2 — Santé globale de l’activité")

# === Source unique (Dataset V2) ===
DATA_PATH = "data/ml_ready/df_ml_churn_ready.csv"

# Colonnes autorisées (strict)
REQ_COLS = [
    "client_id",
    "paid_count_before_T",
    "paid_sum_before_T",
    "days_since_last_paid",
    "plan",
    "ville",
]


@st.cache_data
def load_df(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def guard_columns(df: pd.DataFrame, required: list[str], where: str) -> None:
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise KeyError(f"Colonnes manquantes dans {where}: {missing}")


# =========================
# Chargement + garde-fous
# =========================
try:
    df = load_df(DATA_PATH)
except FileNotFoundError:
    st.error(f"Fichier introuvable : {DATA_PATH}")
    st.stop()
except Exception as e:
    st.error(f"Erreur de lecture : {e}")
    st.stop()

try:
    if df.empty:
        st.warning("Aucune donnée disponible pour évaluer la santé globale.")
        st.stop()
    guard_columns(df, REQ_COLS, f"dataset ({DATA_PATH})")
except Exception as e:
    st.error(str(e))
    st.stop()

# =========================
# Cadre (F7) — visible
# =========================
st.caption(
    "Cadre : cette vue fournit une lecture globale et descriptive de l’activité. "
    "Elle ne priorise aucun client et ne formule aucune recommandation."
)

st.divider()

# =========================
# Indicateurs globaux
# =========================
n_clients = len(df)

avg_days = df["days_since_last_paid"].mean()
med_days = df["days_since_last_paid"].median()

avg_paid_count = df["paid_count_before_T"].mean()
avg_paid_sum = df["paid_sum_before_T"].mean()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Clients observés", f"{n_clients}")
c2.metric("Jours depuis dernier paiement (médiane)", f"{med_days:.0f}")
c3.metric("Nb paiements moyen", f"{avg_paid_count:.1f}")
c4.metric("Montant payé moyen", f"{avg_paid_sum:.2f}")

st.divider()

# =========================
# Répartition simple
# =========================
left, right = st.columns([1, 1], gap="large")

with left:
    st.subheader("Répartition par plan")
    plan_dist = (
        df["plan"].value_counts().rename_axis("plan").reset_index(name="clients")
    )
    st.dataframe(plan_dist, use_container_width=True, hide_index=True)

with right:
    st.subheader("Lecture d’ensemble")
    st.write(
        "- Cette vue décrit **l’état global** du périmètre observé.\n"
        "- Les indicateurs sont **agrégés**, sans distinction individuelle.\n"
        "- Pour identifier des situations particulières, utiliser **E1 — À surveiller**."
    )

st.divider()

# =========================
# Navigation autorisée
# =========================
st.info(
    "Navigation autorisée :\n"
    "- Revenir à **E1 — Vue prioritaire « À surveiller »**\n"
    "- Préparer une **exploration guidée (E5)** (à venir)\n\n"
    "Aucune action métier n’est possible depuis cet écran."
)
