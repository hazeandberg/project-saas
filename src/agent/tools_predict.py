from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any, Dict, Optional

import pandas as pd
import requests


@dataclass(frozen=True)
class ChurnPrediction:
    client_id: str
    churn_probability: Optional[float]
    churn: Optional[bool]
    churn_risk: str  # "low" | "medium" | "high"
    raw: Dict[str, Any]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "client_id": self.client_id,
            "churn_probability": self.churn_probability,
            "churn": self.churn,
            "churn_risk": self.churn_risk,
            "raw": self.raw,
        }


def _compute_days_since_last_paid(
    last_payment_date: Optional[str], reference_date: str
) -> int:
    """
    last_payment_date: 'YYYY-MM-DD' or None
    reference_date: 'YYYY-MM-DD' (T)
    """
    if not last_payment_date:
        return 999

    lp = pd.to_datetime(last_payment_date, errors="coerce")
    ref = pd.to_datetime(reference_date, errors="coerce")
    if pd.isna(lp) or pd.isna(ref):
        return 999

    days = (ref - lp).days
    return int(days) if days >= 0 else 0


def _load_churn_thresholds() -> tuple[float, float]:
    """
    Reads docs_corpus/20_rules.yaml thresholds:
      policy.thresholds.churn_risk.low_max
      policy.thresholds.churn_risk.medium_max
    Fallback defaults:
      low_max=0.20, medium_max=0.50
    """
    low_max = 0.20
    medium_max = 0.50

    try:
        import yaml  # type: ignore
    except Exception:
        return low_max, medium_max

    try:
        with open("docs_corpus/20_rules.yaml", "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        cr = cfg.get("policy", {}).get("thresholds", {}).get("churn_risk", {})
        lm = cr.get("low_max", low_max)
        mm = cr.get("medium_max", medium_max)
        low_max = float(lm)
        medium_max = float(mm)
    except Exception:
        return low_max, medium_max

    # Guardrails
    if not (0.0 <= low_max <= 1.0):
        low_max = 0.20
    if not (0.0 <= medium_max <= 1.0):
        medium_max = 0.50
    if low_max >= medium_max:
        low_max, medium_max = 0.20, 0.50

    return low_max, medium_max


def _bucket_churn_risk(p: Optional[float], churn_flag: Optional[bool]) -> str:
    """
    Produces a stable churn_risk in {"low","medium","high"}.
    Priority:
    - If churn_flag is True -> "high"
    - Else bucketize by probability using YAML thresholds.
    """
    if churn_flag is True:
        return "high"

    low_max, medium_max = _load_churn_thresholds()

    if p is None:
        # Unknown -> conservative but not alarmist
        return "medium"

    if p < low_max:
        return "low"
    if p < medium_max:
        return "medium"
    return "high"


def predict_churn(client_id: str) -> ChurnPrediction:
    cid = (client_id or "").strip()
    if not cid:
        raise ValueError("client_id is empty")

    url = (os.getenv("PREDICT_API_URL") or "http://127.0.0.1:8000/predict").strip()
    if not url.startswith("http"):
        raise RuntimeError("Invalid PREDICT_API_URL (must start with http/https)")

    # Local dependency (keeps import graph clean)
    from src.agent.tools_stats import stats_client

    st = stats_client(cid)

    # Reference date T = max observed last_activity_date (robust for frozen datasets)
    df = pd.read_csv("data/processed/report_oop.csv")
    T = pd.to_datetime(df.get("last_activity_date"), errors="coerce").max()
    if pd.isna(T):
        T = pd.to_datetime(df.get("last_payment_date"), errors="coerce").max()
    if pd.isna(T):
        T = pd.Timestamp.today()
    T_str = str(T.date())

    payload = {
        "paid_count_before_T": st.nb_paiements,
        "paid_sum_before_T": st.ca_total,
        "days_since_last_paid": _compute_days_since_last_paid(
            st.last_payment_date, T_str
        ),
        "plan": st.plan,
        "ville": st.ville,
    }

    resp = requests.post(url, json=payload, timeout=15)
    resp.raise_for_status()
    data: Dict[str, Any] = resp.json()

    churn_probability: Optional[float] = None
    churn_flag: Optional[bool] = None

    if "churn_probability" in data:
        try:
            churn_probability = float(data["churn_probability"])
        except Exception:
            churn_probability = None

    if "churn" in data:
        try:
            churn_flag = bool(data["churn"])
        except Exception:
            churn_flag = None

    churn_risk = _bucket_churn_risk(churn_probability, churn_flag)

    return ChurnPrediction(
        client_id=cid,
        churn_probability=churn_probability,
        churn=churn_flag,
        churn_risk=churn_risk,
        raw=data,
    )
