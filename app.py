import pandas as pd
import streamlit as st

st.set_page_config(page_title="Dashboard V1", layout="wide")

st.title("Dashboard V1 — Report OOP")

PATH = "data/processed/report_oop.csv"

df = pd.read_csv(PATH)

# parsing dates (robuste)
df["last_payment_date"] = pd.to_datetime(df["last_payment_date"], errors="coerce")
df["last_activity_date"] = pd.to_datetime(df["last_activity_date"], errors="coerce")

st.subheader("Aperçu du dataset")
st.write(f"Lignes: {len(df)} — Colonnes: {len(df.columns)}")
st.dataframe(df, use_container_width=True)

st.subheader("Top clients")

top_n = st.slider(
    "Nombre de clients à afficher", min_value=5, max_value=50, value=10, step=5
)

top_clients = (
    df.sort_values(["ca_total", "actions_total"], ascending=[False, False])
    .loc[
        :,
        [
            "client_id",
            "plan",
            "ville",
            "ca_total",
            "actions_total",
            "sessions_total",
            "nb_paiements",
            "last_payment_date",
            "last_activity_date",
        ],
    ]
    .head(top_n)
)

st.dataframe(top_clients, use_container_width=True)

st.subheader("Chiffre d'affaires par plan")

ca_par_plan = (
    df.groupby("plan")
    .agg(
        ca_total=("ca_total", "sum"),
        nb_clients=("client_id", "count"),
        ca_moyen_client=("ca_total", "mean"),
    )
    .reset_index()
    .sort_values("ca_total", ascending=False)
)

st.dataframe(ca_par_plan, use_container_width=True)

st.subheader("Churn — règle heuristique (X = 30 jours)")

X_DAYS = 30

# date de référence = dernière activité observée dans le dataset
date_ref = df["last_activity_date"].max()

# calcul de l'ancienneté en jours
df["anciennete_jours"] = (date_ref - df["last_activity_date"]).dt.days

# règle de churn
df["is_churn"] = (df["nb_paiements"] == 0) & (df["anciennete_jours"] > X_DAYS)

# résumé global
nb_clients = len(df)
nb_churn = df["is_churn"].sum()
taux_churn = nb_churn / nb_clients if nb_clients > 0 else 0

st.write(f"Date de référence : {date_ref.date()}")
st.write(f"Clients churn : {nb_churn} / {nb_clients} ({taux_churn:.1%})")

# tableau des clients churn
df_churn = (
    df[df["is_churn"]]
    .loc[
        :,
        [
            "client_id",
            "plan",
            "ville",
            "nb_paiements",
            "anciennete_jours",
            "last_activity_date",
        ],
    ]
    .sort_values("anciennete_jours", ascending=False)
)

st.dataframe(df_churn, use_container_width=True)

st.subheader("CA par plan — graphique")
st.bar_chart(ca_par_plan.set_index("plan")[["ca_total"]])
