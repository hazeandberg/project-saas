# src/dashboards/app_e1_watchlist.py
import pandas as pd
import streamlit as st

st.header("E1 — À surveiller")
st.caption(
    "Vue prioritaire : situations à regarder en premier (sans score, sans décision automatique)."
)

# === Source unique (Dataset V2) ===
DATA_PATH = "data/ml_ready/df_ml_churn_ready.csv"

# Colonnes autorisées (strict)
REQ_COLS = [
    "client_id",
    "paid_count_before_T",
    "paid_sum_before_T",
    "last_paid_before_T",
    "days_since_last_paid",
    "plan",
    "ville",
]

# Top K fixe (pas de réglage libre sur E1)
TOP_K = 30


@st.cache_data
def load_df(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def guard_columns(df: pd.DataFrame, required: list[str], where: str) -> None:
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise KeyError(f"Colonnes manquantes dans {where}: {missing}")


def build_watchlist(df: pd.DataFrame) -> pd.DataFrame:
    """
    Priorisation factuelle, explicable, sans score ML :
    1) days_since_last_paid DESC (plus ancien en premier)
    2) paid_count_before_T ASC (moins d'historique en premier)
    3) paid_sum_before_T ASC
    """
    out = df.copy()

    # Types robustes (sans suppositions)
    out["days_since_last_paid"] = pd.to_numeric(
        out["days_since_last_paid"], errors="coerce"
    )
    out["paid_count_before_T"] = pd.to_numeric(
        out["paid_count_before_T"], errors="coerce"
    )
    out["paid_sum_before_T"] = pd.to_numeric(out["paid_sum_before_T"], errors="coerce")

    # Date lisible (sans effet de bord)
    out["last_paid_before_T"] = pd.to_datetime(
        out["last_paid_before_T"], errors="coerce"
    )

    ranked = out.sort_values(
        [
            "days_since_last_paid",
            "paid_count_before_T",
            "paid_sum_before_T",
            "client_id",
        ],
        ascending=[False, True, True, True],
        na_position="last",
    )

    visible = ranked[
        [
            "client_id",
            "plan",
            "ville",
            "days_since_last_paid",
            "paid_count_before_T",
            "paid_sum_before_T",
            "last_paid_before_T",
        ]
    ].head(TOP_K)

    visible = visible.copy()
    visible["paid_sum_before_T"] = visible["paid_sum_before_T"].round(2)
    return visible


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
        st.warning("Aucun signal critique : le dataset est vide.")
        st.stop()
    guard_columns(df, REQ_COLS, f"dataset ({DATA_PATH})")
except Exception as e:
    st.error(str(e))
    st.stop()

# =========================
# Cadre (F7) — visible
# =========================
st.caption(
    "Cadre : cette vue met en évidence des situations à regarder en priorité, "
    "sans score, sans prédiction, sans recommandation d’action."
)
with st.expander("Voir la règle de priorisation (factuelle)"):
    st.write(
        "- Tri 1 : **jours depuis dernier paiement** (plus élevé = en premier)\n"
        "- Tri 2 : **nombre de paiements avant T** (plus faible = en premier)\n"
        "- Tri 3 : **montant total payé avant T** (plus faible = en premier)\n"
        "- Aucun modèle, aucun seuil, aucune probabilité."
    )

st.divider()

# =========================
# Filtre unique autorisé (plan)
# =========================
st.subheader("Filtre (borné)")
plans = ["(tous)"] + sorted(df["plan"].dropna().unique().tolist())
plan_sel = st.selectbox("Plan", plans)

df_f = df.copy()
if plan_sel != "(tous)":
    df_f = df_f[df_f["plan"] == plan_sel].reset_index(drop=True)

if df_f.empty:
    st.info("Aucun client pour ce plan dans le périmètre observé.")
    st.stop()

st.divider()

# =========================
# Watchlist
# =========================
watchlist = build_watchlist(df_f)

if watchlist.empty:
    st.warning("Aucun élément prioritaire détecté dans le cadre actuel.")
    st.info("Tu peux consulter E2 (santé globale) si disponible.")
    st.stop()

left, right = st.columns([2, 1], gap="large")

with left:
    st.subheader("Liste priorisée « À surveiller »")
    st.dataframe(watchlist, use_container_width=True, hide_index=True)

with right:
    st.subheader("Sélection")
    choices = watchlist["client_id"].tolist()
    client_id = st.selectbox("Choisir un client_id", choices)

    st.caption(
        "Action autorisée : sélectionner un client pour préparer une exploration guidée (E5). "
        "Aucune action métier n’est déclenchée ici."
    )

st.divider()

# =========================
# Fiche minimale (factuelle, non-exploratoire)
# =========================
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
