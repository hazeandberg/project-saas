from __future__ import annotations

from pathlib import Path
import pandas as pd


def load_raw(base: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    clients = pd.read_csv(base / "clients.csv")
    subs = pd.read_csv(base / "subscriptions.csv")
    usage = pd.read_csv(base / "usage.csv")
    return clients, subs, usage


def clean(
    clients: pd.DataFrame, subs: pd.DataFrame, usage: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # --- clients ---
    clients = clients.copy()
    clients["client_id"] = clients["client_id"].astype(str).str.strip()
    clients["ville"] = clients["ville"].astype(str).str.strip()
    clients["plan"] = clients["plan"].astype(str).str.strip()
    clients["date_inscription"] = pd.to_datetime(
        clients["date_inscription"], format="%Y-%m-%d", errors="coerce"
    )

    clients = clients.dropna(subset=["client_id", "ville", "plan", "date_inscription"])
    clients = clients[clients["plan"].isin(["free", "basic", "pro"])]
    clients = clients.drop_duplicates(subset=["client_id"], keep="first")

    # --- subscriptions ---
    subs = subs.copy()
    subs["client_id"] = subs["client_id"].astype(str).str.strip()
    subs["montant"] = pd.to_numeric(subs["montant"], errors="coerce")
    subs["date_paiement"] = pd.to_datetime(
        subs["date_paiement"], format="%Y-%m-%d", errors="coerce"
    )
    subs["statut"] = subs["statut"].astype(str).str.strip()

    subs = subs.dropna(subset=["client_id", "montant", "date_paiement", "statut"])
    subs = subs[
        (subs["montant"] >= 0) & (subs["statut"].isin(["paid", "failed", "cancelled"]))
    ]

    # --- usage ---
    usage = usage.copy()
    usage["client_id"] = usage["client_id"].astype(str).str.strip()
    usage["actions"] = pd.to_numeric(usage["actions"], errors="coerce")
    usage["sessions"] = pd.to_numeric(usage["sessions"], errors="coerce")
    usage["timestamp"] = pd.to_datetime(
        usage["timestamp"], format="%Y-%m-%d", errors="coerce"
    )

    usage = usage.dropna(subset=["client_id", "actions", "sessions", "timestamp"])
    usage = usage[(usage["actions"] >= 0) & (usage["sessions"] >= 0)]

    return clients, subs, usage


def enrich_and_aggregate(
    clients: pd.DataFrame, subs: pd.DataFrame, usage: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    # --- Subscriptions KPIs (paid only for revenue KPIs) ---
    subs_paid = subs[subs["statut"] == "paid"].copy()

    kpi_subs = subs_paid.groupby("client_id", as_index=False).agg(
        ca_total=("montant", "sum"),
        nb_paiements=("montant", "count"),
        last_payment_date=("date_paiement", "max"),
    )

    # --- Usage KPIs ---
    kpi_usage = usage.groupby("client_id", as_index=False).agg(
        actions_total=("actions", "sum"),
        sessions_total=("sessions", "sum"),
        last_activity_date=("timestamp", "max"),
        days_active=("timestamp", "nunique"),
    )

    # fréquence simple (actions / jour actif)
    kpi_usage["freq_actions"] = (
        kpi_usage["actions_total"] / kpi_usage["days_active"]
    ).fillna(0)

    # --- Merge all ---
    kpi = clients.merge(kpi_subs, on="client_id", how="left").merge(
        kpi_usage, on="client_id", how="left"
    )

    # Fill missing for free users (no paid subs) or sparse usage
    for col in [
        "ca_total",
        "nb_paiements",
        "actions_total",
        "sessions_total",
        "days_active",
        "freq_actions",
    ]:
        kpi[col] = kpi[col].fillna(0)

    # --- ML-ready v1 features ---
    # ancienneté en jours (par rapport au max timestamp observé dans usage)
    ref_date = usage["timestamp"].max()
    kpi["tenure_days"] = (ref_date - pd.to_datetime(kpi["date_inscription"])).dt.days

    # churn heuristique simple v1 (sera affiné en S2)
    # Si pas d’activité depuis 14 jours (par rapport à ref_date) => churn_heuristic = 1
    kpi["days_since_last_activity"] = (
        ref_date - pd.to_datetime(kpi["last_activity_date"])
    ).dt.days
    kpi["churn_heuristic"] = (kpi["days_since_last_activity"].fillna(9999) > 14).astype(
        int
    )

    # df_ml_ready_v1 = colonnes utiles, propres, numériques + catégorielles
    df_ml_ready = kpi[
        [
            "client_id",
            "plan",
            "ville",
            "tenure_days",
            "ca_total",
            "nb_paiements",
            "actions_total",
            "sessions_total",
            "days_active",
            "freq_actions",
            "days_since_last_activity",
            "churn_heuristic",
        ]
    ].copy()

    # Rapport KPI (pour dashboard)
    kpi_report = kpi.sort_values(
        ["ca_total", "actions_total", "client_id"], ascending=[False, False, True]
    ).reset_index(drop=True)

    return kpi_report, df_ml_ready


def export_outputs(
    project_root: Path, kpi_report: pd.DataFrame, df_ml_ready: pd.DataFrame
) -> None:
    processed_dir = project_root / "data" / "processed"
    ml_ready_dir = project_root / "data" / "ml_ready"
    processed_dir.mkdir(parents=True, exist_ok=True)
    ml_ready_dir.mkdir(parents=True, exist_ok=True)

    kpi_report.to_csv(processed_dir / "kpi_by_client.csv", index=False)
    df_ml_ready.to_csv(ml_ready_dir / "df_ml_ready_v1.csv", index=False)


if __name__ == "__main__":
    project_root = Path(".")
    raw_dir = project_root / "data" / "raw"

    clients, subs, usage = load_raw(raw_dir)
    clients, subs, usage = clean(clients, subs, usage)

    kpi_report, df_ml_ready = enrich_and_aggregate(clients, subs, usage)
    export_outputs(project_root, kpi_report, df_ml_ready)

    print("OK pandas_v31")
    print("kpi_by_client:", len(kpi_report))
    print("df_ml_ready_v1:", len(df_ml_ready))
    print(df_ml_ready.head(5))
