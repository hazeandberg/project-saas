from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


REVENUE_EVENTS_PATH = Path("data/derived/revenue_events.csv")


@dataclass(frozen=True)
class RevenueSummary:
    client_id: str
    total_amount: float
    events_count: int
    types: Dict[str, int]
    most_recent_date: Optional[str]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "client_id": self.client_id,
            "total_amount": self.total_amount,
            "events_count": self.events_count,
            "types": self.types,
            "most_recent_date": self.most_recent_date,
        }


def get_revenue_events_summary(client_id: str) -> RevenueSummary:
    cid = (client_id or "").strip()
    if not cid:
        raise ValueError("client_id is empty")

    if not REVENUE_EVENTS_PATH.exists():
        # dataset enrichi pas encore créé : on renvoie un résumé vide (propre)
        return RevenueSummary(
            client_id=cid,
            total_amount=0.0,
            events_count=0,
            types={},
            most_recent_date=None,
        )

    df = pd.read_csv(REVENUE_EVENTS_PATH)

    required = {"client_id", "event_type", "amount", "event_date"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"revenue_events.csv missing required columns: {sorted(missing)}"
        )

    d = df.loc[df["client_id"].astype(str) == cid]
    if d.empty:
        return RevenueSummary(
            client_id=cid,
            total_amount=0.0,
            events_count=0,
            types={},
            most_recent_date=None,
        )

    # nettoyage léger
    d = d.copy()
    d["amount"] = pd.to_numeric(d["amount"], errors="coerce").fillna(0.0)
    d["event_date"] = pd.to_datetime(d["event_date"], errors="coerce")

    total_amount = float(d["amount"].sum())
    events_count = int(len(d))
    types = d["event_type"].astype(str).value_counts().to_dict()

    most_recent = d["event_date"].max()
    most_recent_date = None if pd.isna(most_recent) else str(most_recent.date())

    return RevenueSummary(
        client_id=cid,
        total_amount=total_amount,
        events_count=events_count,
        types={str(k): int(v) for k, v in types.items()},
        most_recent_date=most_recent_date,
    )
