from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import pandas as pd

from src.agent.rules_loader import Policy


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _days_between(date_a: Optional[str], date_b: Optional[str]) -> Optional[int]:
    if not date_a or not date_b:
        return None
    a = pd.to_datetime(date_a, errors="coerce")
    b = pd.to_datetime(date_b, errors="coerce")
    if pd.isna(a) or pd.isna(b):
        return None
    return int((b - a).days)


def compute_usage_level(sessions_total: int, policy: Policy) -> str:
    usage_cfg = (policy.thresholds or {}).get("usage_level", {}) or {}
    high_sessions = _safe_int(usage_cfg.get("high_sessions", 30), 30)
    low_sessions = _safe_int(usage_cfg.get("low_sessions", 8), 8)

    if sessions_total >= high_sessions:
        return "high"
    if sessions_total <= low_sessions:
        return "low"
    return "medium"


def compute_ca_total_high(
    ca_total: float, policy: Policy, fallback_quantile: float = 0.80
) -> bool:
    """
    Pro rule: ca_total_high should be derived from data distribution.
    MVP rule (robust): compute threshold as the fallback_quantile of ca_total in report_oop.csv.
    If file read fails, fallback to a simple absolute threshold (e.g., 300).
    """
    try:
        df = pd.read_csv("data/processed/report_oop.csv")
        thr = float(
            pd.to_numeric(df["ca_total"], errors="coerce").quantile(fallback_quantile)
        )
        if pd.isna(thr) or thr <= 0:
            thr = 300.0
    except Exception:
        thr = 300.0
    return ca_total >= thr


def compute_recent_one_shot(
    most_recent_date: Optional[str],
    window_days: int = 30,
    reference_date: Optional[str] = None,
) -> bool:
    """
    recent_one_shot = True if the most recent one-shot revenue event happened within window_days.
    If we don't have data, returns False (safe).
    """
    if not most_recent_date:
        return False

    mr = pd.to_datetime(most_recent_date, errors="coerce")
    if pd.isna(mr):
        return False

    if reference_date:
        ref = pd.to_datetime(reference_date, errors="coerce")
    else:
        ref = pd.Timestamp.today()

    if pd.isna(ref):
        ref = pd.Timestamp.today()

    delta = (ref - mr).days
    return 0 <= int(delta) <= int(window_days)


def build_features(
    policy: Policy,
    *,
    client_stats: Any,
    churn_pred: Any,
    revenue_summary: Any,
    reference_date: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Returns a dict used by the rules engine.
    Expected keys (from YAML conditions):
      - churn_risk
      - ca_total_high
      - recent_one_shot
      - usage_level
    """
    sessions_total = _safe_int(getattr(client_stats, "sessions_total", 0), 0)
    ca_total = _safe_float(getattr(client_stats, "ca_total", 0.0), 0.0)

    churn_risk = getattr(churn_pred, "churn_risk", None) or "medium"

    # revenue_summary expected fields:
    # - most_recent_date (or None)
    most_recent_date = getattr(revenue_summary, "most_recent_date", None)

    feats: Dict[str, Any] = {
        "churn_risk": churn_risk,
        "usage_level": compute_usage_level(sessions_total, policy),
        "ca_total_high": compute_ca_total_high(ca_total, policy),
        "recent_one_shot": compute_recent_one_shot(
            most_recent_date, window_days=30, reference_date=reference_date
        ),
    }
    return feats
