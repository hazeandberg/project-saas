import streamlit as st
import pandas as pd
import joblib

st.title("Dashboard V3 — ML (client prediction, plus metrics)")

METRICS_PATH = "data/ml_ready/metrics_v1.csv"

st.divider()
st.header("Modèle — métriques (runs)")

try:
    metrics = pd.read_csv(METRICS_PATH)
    st.dataframe(metrics, use_container_width=True)

    last = metrics.iloc[-1].to_dict()
    st.subheader("Dernier run")
    st.write(last)

except FileNotFoundError:
    st.warning(f"Fichier introuvable: {METRICS_PATH}")


MODEL_PATH = "src/ml/models/model_current_v1.joblib"
REPORT_PATH = "data/processed/report_oop.csv"


@st.cache_data
def load_report():
    df = pd.read_csv(REPORT_PATH)
    df["last_activity_date"] = pd.to_datetime(df["last_activity_date"])
    ref_date = df["last_activity_date"].max()
    df["anciennete_jours"] = (ref_date - df["last_activity_date"]).dt.days
    return df


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


report = load_report()
st.divider()
st.header("KPI — Top clients")

top_n = st.slider("Top N clients", min_value=3, max_value=20, value=10)

top_clients = report.sort_values("ca_total", ascending=False).head(top_n)[
    ["client_id", "plan", "ville", "ca_total", "actions_total", "sessions_total"]
]

st.dataframe(top_clients, use_container_width=True)

st.divider()
st.header("KPI — CA par plan")

ca_par_plan = (
    report.groupby("plan", as_index=False)
    .agg(ca_total=("ca_total", "sum"))
    .sort_values("ca_total", ascending=False)
)

st.dataframe(ca_par_plan, use_container_width=True)

st.subheader("Graphique — CA total par plan")
st.bar_chart(ca_par_plan.set_index("plan")["ca_total"])

model = load_model()

FEATURE_COLS = [
    "nb_paiements",
    "actions_total",
    "sessions_total",
    "anciennete_jours",
    "plan",
    "ville",
]

client_id = st.text_input("client_id", value="")

if not client_id:
    st.info("Entre un client_id (ex: C008) pour voir la prédiction.")
else:

    row = report.loc[report["client_id"] == client_id]
    if row.empty:
        st.error(f"client_id introuvable: {client_id}")
    else:
        X_one = row.iloc[[0]][FEATURE_COLS]
        pred = float(model.predict(X_one)[0])
        actual = float(row["ca_total"].iloc[0])

        st.write("### Résultat")
        st.write(
            {
                "client_id": client_id,
                "prediction": round(pred, 2),
                "actual": round(actual, 2),
            }
        )
