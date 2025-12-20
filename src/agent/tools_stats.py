from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd


REPORT_PATH = Path("data/processed/report_oop.csv")


@dataclass(frozen=True)
class ClientStats:
    client_id: str
    plan: Optional[str]
    ville: Optional[str]
    ca_total: float
    nb_paiements: int
    actions_total: int
    sessions_total: int
    last_activity_date: Optional[str]
    last_payment_date: Optional[str]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "client_id": self.client_id,
            "plan": self.plan,
            "ville": self.ville,
            "ca_total": self.ca_total,
            "nb_paiements": self.nb_paiements,
            "actions_total": self.actions_total,
            "sessions_total": self.sessions_total,
            "last_activity_date": self.last_activity_date,
            "last_payment_date": self.last_payment_date,
        }


def stats_client(client_id: str) -> ClientStats:
    cid = (client_id or "").strip()
    if not cid:
        raise ValueError("client_id is empty")

    if not REPORT_PATH.exists():
        raise FileNotFoundError(f"Missing file: {REPORT_PATH}")

    df = pd.read_csv(REPORT_PATH)

    if "client_id" not in df.columns:
        raise ValueError("report_oop.csv missing required column: client_id")

    row = df.loc[df["client_id"].astype(str) == cid]
    if row.empty:
        raise ValueError(f"client_id not found in report_oop.csv: {cid}")

    r = row.iloc[0].to_dict()

    def _to_int(x: Any, default: int = 0) -> int:
        try:
            return int(x)
        except Exception:
            return default

    def _to_float(x: Any, default: float = 0.0) -> float:
        try:
            return float(x)
        except Exception:
            return default

    def _to_str_or_none(x: Any) -> Optional[str]:
        if x is None:
            return None
        s = str(x).strip()
        return s if s else None

    return ClientStats(
        client_id=cid,
        plan=_to_str_or_none(r.get("plan")),
        ville=_to_str_or_none(r.get("ville")),
        ca_total=_to_float(r.get("ca_total"), 0.0),
        nb_paiements=_to_int(r.get("nb_paiements"), 0),
        actions_total=_to_int(r.get("actions_total"), 0),
        sessions_total=_to_int(r.get("sessions_total"), 0),
        last_activity_date=_to_str_or_none(
            r.get("last_activity_date") or r.get("last_activity")
        ),
        last_payment_date=_to_str_or_none(r.get("last_payment_date")),
    )
