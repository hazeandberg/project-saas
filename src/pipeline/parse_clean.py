from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from typing import Iterable


# ---------- Helpers ----------


def parse_date(value: str) -> datetime | None:
    try:
        return datetime.strptime(value.strip(), "%Y-%m-%d")
    except Exception:
        return None


def parse_int(value: str) -> int | None:
    try:
        return int(value)
    except Exception:
        return None


def parse_float(value: str) -> float | None:
    try:
        return float(value)
    except Exception:
        return None


# ---------- Parsers ----------


def parse_clients(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            client_id = r.get("client_id", "").strip()
            ville = r.get("ville", "").strip()
            plan = r.get("plan", "").strip()
            date_inscription = parse_date(r.get("date_inscription", ""))

            if not client_id or not ville or plan not in {"free", "basic", "pro"}:
                continue
            if date_inscription is None:
                continue

            rows.append(
                {
                    "client_id": client_id,
                    "ville": ville,
                    "plan": plan,
                    "date_inscription": date_inscription,
                }
            )
    return rows


def parse_subscriptions(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            client_id = r.get("client_id", "").strip()
            montant = parse_float(r.get("montant", ""))
            date_paiement = parse_date(r.get("date_paiement", ""))
            statut = r.get("statut", "").strip()

            if not client_id or statut not in {"paid", "failed", "cancelled"}:
                continue
            if montant is None or montant < 0:
                continue
            if date_paiement is None:
                continue

            rows.append(
                {
                    "client_id": client_id,
                    "montant": montant,
                    "date_paiement": date_paiement,
                    "statut": statut,
                }
            )
    return rows


def parse_usage(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            client_id = r.get("client_id", "").strip()
            actions = parse_int(r.get("actions", ""))
            sessions = parse_int(r.get("sessions", ""))
            timestamp = parse_date(r.get("timestamp", ""))

            if not client_id:
                continue
            if actions is None or actions < 0:
                continue
            if sessions is None or sessions < 0:
                continue
            if timestamp is None:
                continue

            rows.append(
                {
                    "client_id": client_id,
                    "actions": actions,
                    "sessions": sessions,
                    "timestamp": timestamp,
                }
            )
    return rows


# ---------- Entry point (test manuel) ----------

if __name__ == "__main__":
    base = Path("data/raw")

    clients = parse_clients(base / "clients.csv")
    subs = parse_subscriptions(base / "subscriptions.csv")
    usage = parse_usage(base / "usage.csv")

    print(f"clients: {len(clients)}")
    print(f"subscriptions: {len(subs)}")
    print(f"usage: {len(usage)}")
