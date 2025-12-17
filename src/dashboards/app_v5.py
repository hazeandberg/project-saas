import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

st.title(
    "Dashboard V5 — ML (client prediction, metrics, features, qualité des prédictions (régression))"
)

METRICS_PATH = "data/ml_ready/metrics_v1.csv"
MODEL_PATH = "src/ml/models/model_current_v1.joblib"
REPORT_PATH = "data/processed/report_oop.csv"

FEATURE_COLS = [
    "nb_paiements",
    "actions_total",
    "sessions_total",
    "anciennete_jours",
    "plan",
    "ville",
]


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


# --- Chargements ---
report = load_report()
model = load_model()

# --- Metrics (runs) ---
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

# --- Feature importance (si RandomForest) ---
st.divider()
st.header("Modèle — importance des features (RandomForest)")

try:
    preprocess = model.named_steps["preprocess"]
    model_rf = model.named_steps["model"]

    feature_names = preprocess.get_feature_names_out()

    if not hasattr(model_rf, "feature_importances_"):
        st.info(
            "Le modèle courant n'expose pas feature_importances_ (pas un RandomForest)."
        )
    else:
        importances = model_rf.feature_importances_
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

top_clients = report.sort_values("ca_total", ascending=False).head(top_n)[
    ["client_id", "plan", "ville", "ca_total", "actions_total", "sessions_total"]
]

st.dataframe(top_clients, use_container_width=True)

# --- KPI CA par plan + bar chart ---
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

# --- Qualité des prédictions (régression) ---
st.divider()
st.header("Modèle — qualité des prédictions (régression)")

X_all = report[FEATURE_COLS]
y_true = report["ca_total"]
y_pred = model.predict(X_all)

mae = mean_absolute_error(y_true, y_pred)
mse = mean_squared_error(y_true, y_pred)
rmse = mse**0.5

st.write({"MAE": round(float(mae), 2), "RMSE": round(float(rmse), 2)})

pred_df = report[["client_id", "ca_total"]].copy()
pred_df["pred_ca_total"] = y_pred
pred_df["residual"] = pred_df["ca_total"] - pred_df["pred_ca_total"]

st.subheader("Table — actual / pred / residual")
st.dataframe(pred_df.sort_values("residual", ascending=False), use_container_width=True)

st.subheader("Graphique — Actual vs Pred")
fig, ax = plt.subplots()
ax.scatter(pred_df["ca_total"], pred_df["pred_ca_total"])
ax.set_xlabel("Actual (ca_total)")
ax.set_ylabel("Predicted (pred_ca_total)")
st.pyplot(fig)

# --- Prédiction par client_id ---
st.divider()
st.header("Prédiction — par client_id")

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

        st.write(
            {
                "client_id": client_id,
                "prediction": round(pred, 2),
                "actual": round(actual, 2),
            }
        )
