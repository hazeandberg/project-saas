import json
import joblib
import pandas as pd
import streamlit as st

st.title("Dashboard V6 — ML (Churn 7–30j) — Décisionnel")

DATA_PATH = "data/ml_ready/df_ml_churn_ready.csv"
METRICS_PATH = "data/ml_ready/churn_metrics_v1.json"
MODEL_PATH = "src/ml/models/churn_model_v1.joblib"

TARGET = "churn_7_30j"
FEATURES_NUM = ["paid_count_before_T", "paid_sum_before_T", "days_since_last_paid"]
FEATURES_CAT = ["plan", "ville"]

# ==========
# Load
# ==========
df = pd.read_csv(DATA_PATH)
model = joblib.load(MODEL_PATH)

metrics = None
try:
    with open(METRICS_PATH, "r", encoding="utf-8") as f:
        metrics = json.load(f)
except FileNotFoundError:
    st.warning(f"Métriques introuvables: {METRICS_PATH}")

st.subheader("Aperçu dataset")
st.write(df.head(10))
st.caption(
    f"shape={df.shape} | churn distribution: {df[TARGET].value_counts().to_dict()}"
)

# ==========
# Qualité du modèle
# ==========
st.subheader("Qualité du modèle (classification)")

if metrics:
    st.write("**Confusion matrix** (format [[TN, FP],[FN, TP]]):")
    st.write(metrics.get("confusion_matrix"))

    c1, c2, c3 = st.columns(3)
    c1.metric("Precision", f"{metrics.get('precision', 0):.3f}")
    c2.metric("Recall", f"{metrics.get('recall', 0):.3f}")
    c3.metric("Test size", str(metrics.get("n_test", "NA")))

    st.text("Classification report:")
    st.text(metrics.get("classification_report", ""))

# ==========
# Feature importance (simple)
# ==========
st.subheader("Feature importance (simple)")

# Pour LogisticRegression: coefficients sur les features one-hot
try:
    pre = model.named_steps["pre"]
    clf = model.named_steps["clf"]

    # Récupérer les noms des features après preprocessing
    ohe = pre.named_transformers_["cat"]
    cat_feature_names = list(ohe.get_feature_names_out(FEATURES_CAT))
    feature_names = FEATURES_NUM + cat_feature_names

    coefs = clf.coef_[0]
    imp = pd.DataFrame({"feature": feature_names, "weight": coefs}).sort_values(
        "weight", ascending=False
    )

    st.write("Poids positifs = augmentent le risque churn (classe 1).")
    st.dataframe(imp.head(15), use_container_width=True)
    st.write("Poids négatifs = diminuent le risque churn.")
    st.dataframe(imp.tail(15).sort_values("weight"), use_container_width=True)

except Exception as e:
    st.warning(f"Impossible d'afficher la feature importance: {e}")

# ==========
# Prédiction client
# ==========
st.subheader("Prédiction client spécifique")

client_id = st.selectbox("Choisir un client_id", sorted(df["client_id"].unique()))
row = df[df["client_id"] == client_id].iloc[0]

X = pd.DataFrame([row[FEATURES_NUM + FEATURES_CAT].to_dict()])

proba = float(model.predict_proba(X)[0][1])
pred = int(model.predict(X)[0])

st.write(f"**client_id**: {client_id}")
st.write(f"**churn_probability**: {proba:.3f}")
st.write(f"**churn_pred**: {pred}")

st.caption(
    "Note: dataset très petit et déséquilibré — dashboard = validation architecture."
)
