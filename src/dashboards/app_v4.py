import streamlit as st
import pandas as pd
import joblib


st.title("Dashboard V4 — ML (client prediction, plus metrics, plus features)")

METRICS_PATH = "data/ml_ready/metrics_v1.csv"
MODEL_PATH = "src/ml/models/model_current_v1.joblib"
REPORT_PATH = "data/processed/report_oop.csv"

DATE_COL = "last_activity_date"
ID_COL = "client_id"
TARGET_COL = "ca_total"

FEATURE_COLS = [
    "nb_paiements",
    "actions_total",
    "sessions_total",
    "anciennete_jours",
    "plan",
    "ville",
]


def guard_columns(df: pd.DataFrame, required: list[str], where: str) -> None:
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise KeyError(f"Colonnes manquantes dans {where}: {missing}")


@st.cache_data
def load_report() -> pd.DataFrame:
    df = pd.read_csv(REPORT_PATH)

    # Guards minimal input
    guard_columns(df, [DATE_COL], f"report ({REPORT_PATH})")

    # Parse dates robust
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    ref_date = df[DATE_COL].max()
    if pd.isna(ref_date):
        raise ValueError(
            f"Impossible de calculer ref_date: toutes les valeurs de {DATE_COL} sont NaT après parsing."
        )

    df["anciennete_jours"] = (ref_date - df[DATE_COL]).dt.days
    return df


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


# --- Chargements ---
try:
    report = load_report()
    model = load_model()
except FileNotFoundError as e:
    st.error(f"Fichier introuvable: {e}")
    st.stop()
except Exception as e:
    st.error(f"Erreur chargement: {e}")
    st.stop()


# --- Trace UI minimale ---
st.caption("Sources")
st.write(
    {
        "REPORT_PATH": REPORT_PATH,
        "MODEL_PATH": MODEL_PATH,
        "METRICS_PATH": METRICS_PATH,
        "shape_report": report.shape,
        "cols_report": report.columns.tolist(),
    }
)

# Guard report columns used in KPIs + prediction
try:
    guard_columns(report, [ID_COL, TARGET_COL], "report (KPI)")
    guard_columns(report, FEATURE_COLS, "report (features prediction)")
except Exception as e:
    st.error(str(e))
    st.stop()


# --- Metrics (runs) ---
st.divider()
st.header("Modèle — métriques (runs)")

try:
    metrics = pd.read_csv(METRICS_PATH)
    st.dataframe(metrics, use_container_width=True)

    if not metrics.empty:
        last = metrics.iloc[-1].to_dict()
        st.subheader("Dernier run")
        st.write(last)
    else:
        st.info("metrics_v1.csv est vide.")
except FileNotFoundError:
    st.warning(f"Fichier introuvable: {METRICS_PATH}")
except Exception as e:
    st.warning(f"Impossible de lire metrics: {e}")


# --- Feature importance (si RandomForest) ---
st.divider()
st.header("Modèle — importance des features (RandomForest)")

try:
    preprocess = model.named_steps["preprocess"]
    model_core = model.named_steps["model"]
    feature_names = preprocess.get_feature_names_out()

    if not hasattr(model_core, "feature_importances_"):
        st.info(
            "Le modèle courant n'expose pas feature_importances_ (pas un RandomForest)."
        )
    else:
        importances = model_core.feature_importances_
        fi = pd.DataFrame(
            {"feature": feature_names, "importance": importances}
        ).sort_values("importance", ascending=False)
        st.dataframe(fi.head(15), use_container_width=True)
        st.bar_chart(fi.set_index("feature")["importance"].head(15))
except Exception as e:
    st.warning(f"Impossible d'afficher feature importance : {e}")


# --- KPI Top clients ---
st.divider()
st.header("KPI — Top clients")

top_n = st.slider("Top N clients", min_value=3, max_value=20, value=10)

top_clients = report.sort_values(TARGET_COL, ascending=False).head(top_n)[
    [ID_COL, "plan", "ville", TARGET_COL, "actions_total", "sessions_total"]
]
st.dataframe(top_clients, use_container_width=True)


# --- KPI CA par plan ---
st.divider()
st.header("KPI — CA par plan")

ca_par_plan = (
    report.groupby("plan", as_index=False)
    .agg(ca_total=(TARGET_COL, "sum"))
    .sort_values("ca_total", ascending=False)
)

st.dataframe(ca_par_plan, use_container_width=True)
st.subheader("Graphique — CA total par plan")
st.bar_chart(ca_par_plan.set_index("plan")["ca_total"])


# --- Prediction by client_id ---
st.divider()
st.header("Prédiction — par client_id")

client_id = st.text_input("client_id", value="")

if not client_id:
    st.info("Entre un client_id (ex: C008) pour voir la prédiction.")
else:
    row = report.loc[report[ID_COL] == client_id]
    if row.empty:
        st.error(f"client_id introuvable: {client_id}")
    else:
        X_one = row.iloc[[0]][FEATURE_COLS]
        pred = float(model.predict(X_one)[0])
        actual = float(row[TARGET_COL].iloc[0])

        st.write("### Résultat")
        st.write(
            {
                "client_id": client_id,
                "prediction": round(pred, 2),
                "actual": round(actual, 2),
            }
        )
