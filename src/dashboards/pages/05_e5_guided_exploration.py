# src/dashboards/app_e5_guided_exploration.py
import os
import re

from typing import List, Tuple, Dict, Any


import numpy as np
import pandas as pd
import streamlit as st

st.header("E5 — Exploration guidée")
st.caption("E5 — Exploration ciblée (guidée) — Contexte + options (non automatisées)")

# === Source unique (Dataset V2) ===
DATA_PATH = "data/ml_ready/df_ml_churn_ready.csv"

# === Corpus (RAG explicatif) ===
CORPUS_DIR = "docs_corpus"
CORPUS_FILES = [
    "10_metrics.md",
    "20_rules.yaml",
    "30_playbooks_retention.md",
    "31_playbooks_one_shot.md",
    "40_response_format.md",
    "00_context.md",
]

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

# Indicateurs utilisés en E5 (guidé)
NUM_COLS = ["days_since_last_paid", "paid_count_before_T", "paid_sum_before_T"]


# =========================
# Utilitaires data
# =========================
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


def safe_median(s: pd.Series) -> float:
    s2 = pd.to_numeric(s, errors="coerce").dropna()
    return float(s2.median()) if not s2.empty else float("nan")


def percentile_in_group(value: float, group: pd.Series) -> float:
    """
    Percentile descriptif : part des valeurs <= value (0..100).
    Purement descriptif, sans seuil ni décision.
    """
    g = pd.to_numeric(group, errors="coerce").dropna()
    if g.empty or pd.isna(value):
        return float("nan")
    return float((g.le(value).mean()) * 100.0)


# =========================
# Utilitaires RAG (retrieval simple, explicatif)
# =========================


def read_text_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def chunk_text(text: str, max_len: int = 900) -> List[str]:
    """
    Chunking simple et robuste :
    - split par paragraphes
    - regroupe jusqu'à max_len
    """
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks = []
    buf = ""
    for p in paras:
        if len(buf) + len(p) + 2 <= max_len:
            buf = (buf + "\n\n" + p).strip()
        else:
            if buf:
                chunks.append(buf)
            buf = p
    if buf:
        chunks.append(buf)
    return chunks


def tokenize(s: str) -> List[str]:
    s = s.lower()
    s = re.sub(r"[^a-z0-9àâçéèêëîïôûùüÿñæœ_\-\s]", " ", s)
    toks = [t for t in s.split() if len(t) >= 3]
    return toks


def score_chunk(query_toks: List[str], chunk_toks: List[str]) -> float:
    # Score overlap (simple, explicatif, déterministe)
    if not query_toks or not chunk_toks:
        return 0.0
    q = set(query_toks)
    c = set(chunk_toks)
    inter = len(q & c)
    # bonus léger si mots-clés répétés dans le chunk
    freq_bonus = 0.0
    for t in q:
        freq_bonus += min(chunk_toks.count(t), 3) * 0.05
    return float(inter) + freq_bonus


@st.cache_data
def load_corpus(corpus_dir: str, files: List[str]) -> List[Dict[str, Any]]:
    chunks: List[Dict[str, Any]] = []
    for fn in files:
        p = os.path.join(corpus_dir, fn)
        if not os.path.exists(p):
            continue
        raw = read_text_file(p)
        for ch in chunk_text(raw):
            chunks.append({"source": fn, "text": ch})
    return chunks


def retrieve(
    chunks: List[Dict[str, Any]], query: str, k: int = 5
) -> List[Tuple[float, Dict[str, Any]]]:
    qt = tokenize(query)
    scored: List[Tuple[float, Dict[str, Any]]] = []

    for ch in chunks:
        text = str(ch.get("text", ""))
        ct = tokenize(text)
        s = score_chunk(qt, ct)
        if s > 0:
            scored.append((s, ch))

    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:k]


def build_query_from_signals(mode: str, plan: str, ville: str, signals: dict) -> str:
    """
    Construit une requête explicative (pas décisionnelle) à partir des signaux factuels.
    """
    parts = ["contexte", "métriques", "rétention", "pmE saas"]
    if mode == "client":
        parts += ["client", plan, ville]
    else:
        parts += ["segment", "plan", plan]

    # signaux
    if signals.get("inactivity_high"):
        parts += ["inactivité", "dernier", "paiement", "retard", "days_since_last_paid"]
    if signals.get("history_low"):
        parts += [
            "historique",
            "paiement",
            "faible",
            "paid_count_before_T",
            "paid_sum_before_T",
        ]
    if plan == "free":
        parts += ["acquisition", "conversion", "free"]
    else:
        parts += ["rétention", "payant", "basic", "pro"]

    return " ".join([p for p in parts if p])


# =========================
# Chargement + garde-fous
# =========================
try:
    df_raw = load_df(DATA_PATH)
except FileNotFoundError:
    st.error(f"Fichier introuvable : {DATA_PATH}")
    st.stop()
except Exception as e:
    st.error(f"Erreur de lecture : {e}")
    st.stop()

try:
    if df_raw.empty:
        st.warning("Aucune donnée disponible pour l’exploration guidée.")
        st.stop()
    guard_columns(df_raw, REQ_COLS, f"dataset ({DATA_PATH})")
except Exception as e:
    st.error(str(e))
    st.stop()

df = prep_types(df_raw)

# Corpus
corpus_chunks = load_corpus(CORPUS_DIR, CORPUS_FILES)

# =========================
# Cadre (F7) — visible
# =========================
st.caption(
    "Cadre : E5 fournit une **lecture guidée** (factuelle + comparaisons descriptives) et un **contexte RAG** "
    "à partir des documents du projet. Aucune probabilité, aucun score ML, aucune recommandation obligatoire, "
    "et aucune automatisation."
)

st.divider()

# =========================
# B1 — Entrée : Client OU Plan (pas d’exploration libre)
# =========================
mode = st.radio(
    "Mode (choisir un seul axe)", ["Client", "Segment (plan)"], horizontal=True
)

if mode == "Client":
    mode_key = "client"
    client_ids = sorted(df["client_id"].dropna().unique().tolist())
    client_id = st.selectbox("client_id", client_ids)
    row = df.loc[df["client_id"] == client_id].iloc[0]

    plan = str(row["plan"])
    ville = str(row["ville"])
    scope_label = f"client_id={client_id}"
else:
    mode_key = "plan"
    plans = sorted(df["plan"].dropna().unique().tolist())
    plan = st.selectbox("plan", plans)
    ville = ""
    scope_label = f"plan={plan}"

st.divider()

# =========================
# B2 — Fiche factuelle (client / segment)
# =========================
st.subheader("Faits observés (périmètre)")

if mode_key == "client":
    facts = {
        "client_id": str(row["client_id"]),
        "plan": str(row["plan"]),
        "ville": str(row["ville"]),
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
else:
    seg = df[df["plan"] == plan].copy()
    st.write(
        {
            "plan": plan,
            "clients (segment)": int(len(seg)),
            "villes (distinct)": int(seg["ville"].nunique(dropna=True)),
        }
    )

st.divider()

# =========================
# B3 — Comparaison guidée (médianes + percentile) — descriptif
# =========================
st.subheader("Comparaisons descriptives (guidées)")

global_meds = {c: safe_median(df[c]) for c in NUM_COLS}

if mode_key == "client":
    seg = df[df["plan"] == plan]
    plan_meds = {c: safe_median(seg[c]) for c in NUM_COLS}

    client_vals = {
        c: float(row[c]) if pd.notna(row[c]) else float("nan") for c in NUM_COLS
    }
    client_pcts = {c: percentile_in_group(client_vals[c], seg[c]) for c in NUM_COLS}

    comp_rows = []
    for c in NUM_COLS:
        comp_rows.append(
            {
                "indicateur": c,
                "valeur_client": client_vals[c],
                "médiane_plan": plan_meds[c],
                "médiane_globale": global_meds[c],
                "percentile_dans_plan_(<=)": client_pcts[c],
            }
        )
    comp = pd.DataFrame(comp_rows)

    # Lisibilité : arrondis
    for col in [
        "valeur_client",
        "médiane_plan",
        "médiane_globale",
        "percentile_dans_plan_(<=)",
    ]:
        comp[col] = pd.to_numeric(comp[col], errors="coerce")
    comp["valeur_client"] = comp["valeur_client"].round(2)
    comp["médiane_plan"] = comp["médiane_plan"].round(2)
    comp["médiane_globale"] = comp["médiane_globale"].round(2)
    comp["percentile_dans_plan_(<=)"] = comp["percentile_dans_plan_(<=)"].round(1)

    st.dataframe(comp, use_container_width=True, hide_index=True)
else:
    seg = df[df["plan"] == plan]
    plan_meds = {c: safe_median(seg[c]) for c in NUM_COLS}

    comp_rows = []
    for c in NUM_COLS:
        comp_rows.append(
            {
                "indicateur": c,
                "médiane_plan": plan_meds[c],
                "médiane_globale": global_meds[c],
            }
        )
    comp = pd.DataFrame(comp_rows)
    for col in ["médiane_plan", "médiane_globale"]:
        comp[col] = pd.to_numeric(comp[col], errors="coerce").round(2)

    st.dataframe(comp, use_container_width=True, hide_index=True)

st.divider()

# =========================
# B4 — Lecture guidée (texte neutre, non prescriptif)
# =========================
st.subheader("Lecture guidée (descriptive, non prescriptive)")

# signaux factuels (sans seuil fixe : on compare à la médiane plan/global)
signals = {"inactivity_high": False, "history_low": False}

if mode_key == "client":
    seg = df[df["plan"] == plan]
    plan_med_days = safe_median(seg["days_since_last_paid"])
    plan_med_count = safe_median(seg["paid_count_before_T"])
    plan_med_sum = safe_median(seg["paid_sum_before_T"])

    v_days = (
        float(row["days_since_last_paid"])
        if pd.notna(row["days_since_last_paid"])
        else float("nan")
    )
    v_count = (
        float(row["paid_count_before_T"])
        if pd.notna(row["paid_count_before_T"])
        else float("nan")
    )
    v_sum = (
        float(row["paid_sum_before_T"])
        if pd.notna(row["paid_sum_before_T"])
        else float("nan")
    )

    if not pd.isna(v_days) and not pd.isna(plan_med_days) and v_days > plan_med_days:
        signals["inactivity_high"] = True
    if (
        not pd.isna(v_count)
        and not pd.isna(plan_med_count)
        and v_count < plan_med_count
    ) or (not pd.isna(v_sum) and not pd.isna(plan_med_sum) and v_sum < plan_med_sum):
        signals["history_low"] = True

    lines = []
    lines.append(f"- Périmètre : **{scope_label}** (plan={plan}, ville={ville}).")
    if not pd.isna(v_days) and not pd.isna(plan_med_days):
        lines.append(
            f"- `days_since_last_paid` = **{int(v_days)}** vs médiane plan **{plan_med_days:.0f}** (médiane globale **{global_meds['days_since_last_paid']:.0f}**)."
        )
    if not pd.isna(v_count) and not pd.isna(plan_med_count):
        lines.append(
            f"- `paid_count_before_T` = **{v_count:.0f}** vs médiane plan **{plan_med_count:.0f}** (médiane globale **{global_meds['paid_count_before_T']:.0f}**)."
        )
    if not pd.isna(v_sum) and not pd.isna(plan_med_sum):
        lines.append(
            f"- `paid_sum_before_T` = **{v_sum:.2f}** vs médiane plan **{plan_med_sum:.2f}** (médiane globale **{global_meds['paid_sum_before_T']:.2f}**)."
        )
    lines.append(
        "- Ces comparaisons sont **descriptives** (pas de score, pas de décision automatique)."
    )
    st.write("\n".join(lines))
else:
    # Segment mode
    signals = {"inactivity_high": False, "history_low": False}
    plan_med_days = safe_median(seg["days_since_last_paid"])
    plan_med_count = safe_median(seg["paid_count_before_T"])
    plan_med_sum = safe_median(seg["paid_sum_before_T"])

    lines = []
    lines.append(f"- Périmètre : **{scope_label}**.")
    lines.append(
        f"- Médiane `days_since_last_paid` (plan) = **{plan_med_days:.0f}** vs globale **{global_meds['days_since_last_paid']:.0f}**."
    )
    lines.append(
        f"- Médiane `paid_count_before_T` (plan) = **{plan_med_count:.0f}** vs globale **{global_meds['paid_count_before_T']:.0f}**."
    )
    lines.append(
        f"- Médiane `paid_sum_before_T` (plan) = **{plan_med_sum:.2f}** vs globale **{global_meds['paid_sum_before_T']:.2f}**."
    )
    lines.append(
        "- Cette lecture est **synthétique** et **descriptive** (pas de décision automatique)."
    )
    st.write("\n".join(lines))

st.divider()

# =========================
# RAG explicatif — contexte depuis docs_corpus
# =========================
st.subheader("Contexte (RAG explicatif depuis les documents)")
if not corpus_chunks:
    st.warning(
        "Corpus introuvable ou vide (docs_corpus). E5 fonctionne, mais sans contexte RAG."
    )
else:
    query = build_query_from_signals(mode_key, plan, ville, signals)
    results = retrieve(corpus_chunks, query, k=5)

    st.caption(f"Requête de contexte (interne) : {query}")

    if not results:
        st.info("Aucun extrait pertinent trouvé dans le corpus pour ce contexte.")
    else:
        for score, ch in results:
            src = str(ch.get("source", "unknown"))
            txt = str(ch.get("text", ""))
            with st.expander(f"{src} — extrait (score={score:.2f})"):
                st.write(txt)


st.divider()

# =========================
# Suggestions d’actions (non automatisées, non prescriptives)
# =========================
st.subheader("Options possibles (non automatisées)")
st.caption(
    "Ces options sont proposées à titre **indicatif**. "
    "Elles ne sont pas exécutées ici et ne constituent pas une recommandation obligatoire."
)

# Options génériques, bornées (pas de déclenchement)
options = [
    "Contacter (email) : message de relance neutre + demande de retour (non automatisé).",
    "Contacter (appel) : prise de nouvelles / vérification de besoin (non automatisé).",
    "Proposer une offre de rétention (si plan payant) : geste commercial ponctuel (non automatisé).",
    "Demander une analyse complémentaire : passer en revue le contexte (E5) et consigner une décision (E4).",
    "Ne rien faire pour l’instant : consigner “Rien à faire” dans E4 (traçabilité).",
]
for o in options:
    st.write(f"- {o}")

st.info(
    "Si tu prends une décision, consigne-la dans **E4 — Journal** (décision humaine + trace)."
)
