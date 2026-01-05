# src/dashboards/app_e3_at_risk_entities.py
import pandas as pd
import streamlit as st

st.header("E3 — Clients / entités à risque")
st.caption("E3 — Clients / entités à risque")

DATA_PATH = "data/ml_ready/df_ml_churn_ready.csv"

REQ_COLS = [
    "client_id",
    "paid_count_before_T",
    "paid_sum_before_T",
    "last_paid_before_T",
    "days_since_last_paid",
    "plan",
    "ville",
]

TOP_N_PER_CATEGORY = 20


@st.cache_data
def load_df(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def guard_columns(df: pd.DataFrame, required: list[str], where: str) -> None:
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise KeyError(f"Colonnes manquantes dans {where}: {missing}")


def prep_types(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["days_since_last_paid"] = pd.to_numeric(
        out["days_since_last_paid"], errors="coerce"
    )
    out["paid_count_before_T"] = pd.to_numeric(
        out["paid_count_before_T"], errors="coerce"
    )
    out["paid_sum_before_T"] = pd.to_numeric(out["paid_sum_before_T"], errors="coerce")
    out["last_paid_before_T"] = pd.to_datetime(
        out["last_paid_before_T"], errors="coerce"
    )
    return out


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
        st.warning("Aucune donnée disponible pour identifier des situations à risque.")
        st.stop()
    guard_columns(df, REQ_COLS, f"dataset ({DATA_PATH})")
except Exception as e:
    st.error(str(e))
    st.stop()

df = prep_types(df)

# =========================
# Cadre & limites (F7)
# =========================
st.caption(
    "Cadre : cet écran **signale des situations à surveiller** par catégories factuelles. "
    "Il n’évalue pas un risque, ne compare pas les catégories entre elles et ne recommande aucune action."
)

# =========================
# Filtre unique autorisé (plan)
# =========================
st.subheader("Filtre (borné)")
plans = ["(tous)"] + sorted(df["plan"].dropna().unique().tolist())
plan_sel = st.selectbox("Plan", plans)

df_f = df.copy()
if plan_sel != "(tous)":
    df_f = df_f[df_f["plan"] == plan_sel]

st.divider()

# =========================
# Catégorie 1 — Inactivité prolongée
# =========================
st.subheader("Situations — Inactivité prolongée")

inactive = (
    df_f.loc[df_f["days_since_last_paid"].notna()]
    .sort_values(["days_since_last_paid", "client_id"], ascending=[False, True])
    .head(TOP_N_PER_CATEGORY)
)

if inactive.empty:
    st.info(
        "Aucune situation d’inactivité prolongée pour ce plan (dans le cadre actuel)."
    )
else:
    st.dataframe(
        inactive[
            ["client_id", "plan", "ville", "days_since_last_paid", "last_paid_before_T"]
        ],
        use_container_width=True,
        hide_index=True,
    )

st.divider()

# =========================
# Catégorie 2 — Historique de paiement faible
# =========================
st.subheader("Situations — Historique de paiement faible")

low_history = (
    df_f.loc[df_f["paid_count_before_T"].notna() & df_f["paid_sum_before_T"].notna()]
    .sort_values(
        ["paid_count_before_T", "paid_sum_before_T", "client_id"],
        ascending=[True, True, True],
    )
    .head(TOP_N_PER_CATEGORY)
)

if low_history.empty:
    st.info(
        "Aucune situation d’historique de paiement faible pour ce plan (dans le cadre actuel)."
    )
else:
    st.dataframe(
        low_history[
            ["client_id", "plan", "ville", "paid_count_before_T", "paid_sum_before_T"]
        ],
        use_container_width=True,
        hide_index=True,
    )

st.divider()

# =========================
# Interaction autorisée
# =========================
st.subheader("Sélection (préparer une exploration guidée)")
choices = sorted(
    set(inactive["client_id"].tolist() + low_history["client_id"].tolist())
)

if not choices:
    st.caption("Aucune entité à sélectionner pour une exploration guidée.")
else:
    client_id = st.selectbox("Choisir un client_id", choices)
    st.caption(
        "Action autorisée : sélectionner un client pour préparer une **exploration guidée (E5)**. "
        "Aucune action métier n’est déclenchée ici."
    )

st.divider()

# =========================
# Fiche minimale (factuelle)
# =========================
if choices:
    st.subheader("Fiche minimale (factuelle)")
    row = df_f.loc[df_f["client_id"] == client_id].iloc[0]

    facts = {
        "client_id": row["client_id"],
        "plan": row["plan"],
        "ville": row["ville"],
        "days_since_last_paid": (
            int(row["days_since_last_paid"])
            if pd.notna(row["days_since_last_paid"])
            else None
        ),
        "paid_count_before_T": (
            int(row["paid_count_before_T"])
            if pd.notna(row["paid_count_before_T"])
            else None
        ),
        "paid_sum_before_T": (
            round(float(row["paid_sum_before_T"]), 2)
            if pd.notna(row["paid_sum_before_T"])
            else None
        ),
        "last_paid_before_T": str(row["last_paid_before_T"]),
    }
    st.write(facts)

    st.info(
        "Étape suivante (plus tard) : basculer vers **E5 — Exploration ciblée** pour comprendre le contexte."
    )
