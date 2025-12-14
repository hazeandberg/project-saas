from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from pathlib import Path

from src.pipeline.parse_clean import parse_clients, parse_subscriptions, parse_usage


def aggregate_by_client(
    clients: list[dict],
    subscriptions: list[dict],
    usage: list[dict],
) -> dict[str, dict]:

    agg: dict[str, dict] = defaultdict(
        lambda: {
            "client_id": None,
            "plan": None,
            "ville": None,
            "date_inscription": None,
            "ca_total": 0.0,
            "nb_paiements": 0,
            "actions_total": 0,
            "sessions_total": 0,
            "last_payment_date": None,
            "last_activity_date": None,
        }
    )

    # --- base client ---
    for c in clients:
        a = agg[c["client_id"]]
        a["client_id"] = c["client_id"]
        a["plan"] = c["plan"]
        a["ville"] = c["ville"]
        a["date_inscription"] = c["date_inscription"]

    # --- subscriptions ---
    for s in subscriptions:
        if s["statut"] != "paid":
            continue

        a = agg[s["client_id"]]
        a["ca_total"] += s["montant"]
        a["nb_paiements"] += 1

        if (
            a["last_payment_date"] is None
            or s["date_paiement"] > a["last_payment_date"]
        ):
            a["last_payment_date"] = s["date_paiement"]

    # --- usage ---
    for u in usage:
        a = agg[u["client_id"]]
        a["actions_total"] += u["actions"]
        a["sessions_total"] += u["sessions"]

        if a["last_activity_date"] is None or u["timestamp"] > a["last_activity_date"]:
            a["last_activity_date"] = u["timestamp"]

    return dict(agg)


# ---------- test manuel ----------
if __name__ == "__main__":
    base = Path("data/raw")

    clients = parse_clients(base / "clients.csv")
    subs = parse_subscriptions(base / "subscriptions.csv")
    usage = parse_usage(base / "usage.csv")

    agg = aggregate_by_client(clients, subs, usage)

    print("clients agrégés:", len(agg))
    sample = list(agg.values())[:3]
    for r in sample:
        print(r)
