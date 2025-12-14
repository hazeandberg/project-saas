from __future__ import annotations

from datetime import datetime
from pathlib import Path

from src.pipeline.parse_clean import parse_clients, parse_subscriptions, parse_usage
from src.pipeline.group_aggregate import aggregate_by_client


def build_report(agg_by_client: dict[str, dict]) -> list[dict]:
    report: list[dict] = []

    for client_id, a in agg_by_client.items():
        report.append(
            {
                "client_id": client_id,
                "plan": a["plan"],
                "ville": a["ville"],
                "ca_total": round(a["ca_total"], 2),
                "nb_paiements": a["nb_paiements"],
                "actions_total": a["actions_total"],
                "sessions_total": a["sessions_total"],
                "last_payment_date": a["last_payment_date"],
                "last_activity_date": a["last_activity_date"],
            }
        )

    report.sort(key=lambda r: (-r["ca_total"], -r["actions_total"], r["client_id"]))
    return report


# ---------- test manuel ----------
if __name__ == "__main__":
    base = Path("data/raw")

    clients = parse_clients(base / "clients.csv")
    subs = parse_subscriptions(base / "subscriptions.csv")
    usage = parse_usage(base / "usage.csv")

    agg = aggregate_by_client(clients, subs, usage)
    report = build_report(agg)

    print("rapport final:", len(report))
    for r in report[:5]:
        print(r)
