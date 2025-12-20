from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import yaml


@dataclass(frozen=True)
class Rule:
    name: str
    expr: str
    priority: int
    action: str
    rationale: str


@dataclass(frozen=True)
class Policy:
    version: int
    rules: List[Rule]
    thresholds: Dict[str, Any]


def load_policy(path: str | Path = "docs_corpus/20_rules.yaml") -> Policy:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Missing policy file: {p}")

    data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}

    version = int(data.get("version", 1))
    policy = data.get("policy", {})
    prio = policy.get("prioritization", [])
    thresholds = policy.get("thresholds", {}) or {}

    rules: List[Rule] = []
    for i, r in enumerate(prio):
        try:
            rules.append(
                Rule(
                    name=str(r["name"]),
                    expr=str(r["if"]),
                    priority=int(r["priority"]),
                    action=str(r["action"]),
                    rationale=str(r.get("rationale", "")),
                )
            )
        except Exception as e:
            raise ValueError(f"Invalid rule at index {i}: {r}") from e

    rules = sorted(rules, key=lambda x: x.priority)

    return Policy(version=version, rules=rules, thresholds=thresholds)
