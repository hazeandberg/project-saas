from __future__ import annotations

from typing import Any, Dict

from src.agent.rules_engine import apply_policy


def decide_action(features: Dict[str, Any], policy) -> Dict[str, Any]:
    rule = apply_policy(policy, features)

    if rule is None:
        return {
            "action": "no_action",
            "priority": "low",
            "rationale": ["Aucune règle métier déclenchée"],
            "confidence": "low",
            "missing_data": ["signal client insuffisant"],
        }

    # Rationale : rester non-tech, basé sur ce que la règle fournit
    rationale = []
    why = getattr(rule, "why", None)
    if isinstance(why, list) and why:
        rationale = [str(x) for x in why]
    else:
        r = getattr(rule, "rationale", None)
        if isinstance(r, str) and r.strip():
            rationale = [r.strip()]
        else:
            rationale = ["Règle métier déclenchée"]

    return {
        "action": getattr(rule, "action", "no_action"),
        "priority": getattr(rule, "priority", "low"),
        "rationale": rationale,
        "confidence": getattr(rule, "confidence", "medium"),
        "missing_data": getattr(rule, "missing_data", []),
    }
