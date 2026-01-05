# src/dashboards/app_e4_decision_journal.py
import os
import csv
from datetime import datetime
import pandas as pd
import streamlit as st

st.header("E4 — Revue décision & journal")
st.caption("E4 — Revue décision & journal (validation humaine)")

# === Sources ===
DATASET_PATH = "data/ml_ready/df_ml_churn_ready.csv"
JOURNAL_DIR = "data/derived"
JOURNAL_PATH = os.path.join(JOURNAL_DIR, "decision_log.csv")

# Colonnes minimales du dataset (affichage factuel)
REQ_DATASET_COLS = ["client_id", "plan", "ville"]

# Schéma du journal (strict)
JOURNAL_FIELDS = [
    "timestamp_iso",
    "scope_type",
    "scope_value",
    "decision_type",
    "rationale_short",
    "author",
]

# Décisions autorisées (bornées)
DECISION_TYPES = [
    "Surveiller",
    "À analyser (E5)",
    "Rien à faire",
    "Hors périmètre",
]


# =========================
# Utilitaires
# =========================
@st.cache_data
def load_dataset(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def ensure_journal(path: str, fields: list[str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()


def append_journal(path: str, row: dict) -> None:
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=JOURNAL_FIELDS)
        writer.writerow(row)


@st.cache_data
def load_journal(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame(columns=JOURNAL_FIELDS)
    return pd.read_csv(path)


def guard_columns(df: pd.DataFrame, required: list[str], where: str) -> None:
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise KeyError(f"Colonnes manquantes dans {where}: {missing}")


# =========================
# Chargement + garde-fous
# =========================
try:
    df = load_dataset(DATASET_PATH)
except FileNotFoundError:
    st.error(f"Dataset introuvable : {DATASET_PATH}")
    st.stop()
except Exception as e:
    st.error(f"Erreur de lecture dataset : {e}")
    st.stop()

try:
    guard_columns(df, REQ_DATASET_COLS, "dataset")
except Exception as e:
    st.error(str(e))
    st.stop()

ensure_journal(JOURNAL_PATH, JOURNAL_FIELDS)

# =========================
# Cadre & limites (F7)
# =========================
st.caption(
    "Cadre : cet écran sert uniquement à **consigner une décision humaine** et à en garder une trace. "
    "Aucune action métier n’est déclenchée."
)

st.divider()

# =========================
# B2 — Périmètre (scope)
# =========================
st.subheader("Périmètre de la décision")

scope_type = st.radio(
    "Type de périmètre", ["Client", "Segment (plan)"], horizontal=True
)

if scope_type == "Client":
    client_ids = sorted(df["client_id"].dropna().unique().tolist())
    scope_value = st.selectbox("client_id", client_ids)
    scope_type_val = "client"
else:
    plans = sorted(df["plan"].dropna().unique().tolist())
    scope_value = st.selectbox("plan", plans)
    scope_type_val = "plan"

# =========================
# B3 — Décision (bornée)
# =========================
st.subheader("Décision")
decision_type = st.selectbox("Type de décision", DECISION_TYPES)

# =========================
# B4 — Justification courte (obligatoire)
# =========================
st.subheader("Justification (courte, obligatoire)")
rationale = st.text_area(
    "Pourquoi cette décision ? (1–2 phrases)",
    max_chars=200,
    placeholder="Ex : activité récente faible mais contexte connu, à surveiller.",
)

author = st.text_input("Auteur (optionnel)", value="")

# =========================
# Enregistrement
# =========================
can_save = bool(rationale and rationale.strip())

if st.button("Enregistrer la décision", disabled=not can_save):
    row = {
        "timestamp_iso": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "scope_type": scope_type_val,
        "scope_value": scope_value,
        "decision_type": decision_type,
        "rationale_short": rationale.strip(),
        "author": author.strip(),
    }
    append_journal(JOURNAL_PATH, row)
    st.success("Décision enregistrée dans le journal.")
    st.cache_data.clear()

st.divider()

# =========================
# B5 — Journal des décisions
# =========================
st.subheader("Journal des décisions")

journal = load_journal(JOURNAL_PATH)

if journal.empty:
    st.info("Aucune décision enregistrée pour le moment.")
else:
    # Tri du plus récent au plus ancien
    journal_sorted = journal.sort_values("timestamp_iso", ascending=False)

    # Filtre simple (optionnel)
    fcol1, fcol2 = st.columns(2)
    with fcol1:
        filter_scope = st.selectbox(
            "Filtrer par périmètre", ["(tous)", "client", "plan"]
        )
    with fcol2:
        filter_value = st.text_input("Valeur (optionnel)", value="")

    jf = journal_sorted.copy()
    if filter_scope != "(tous)":
        jf = jf[jf["scope_type"] == filter_scope]
    if filter_value:
        jf = jf[jf["scope_value"].astype(str) == filter_value]

    st.dataframe(jf, use_container_width=True, hide_index=True)
