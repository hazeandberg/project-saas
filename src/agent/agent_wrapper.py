# src/agent/agent_wrapper.py

from __future__ import annotations

from typing import Any, Dict

from src.agent.decision import decide_action
from src.agent.features import build_features
from src.agent.rules_loader import load_policy
from src.agent.tools_predict import predict_churn
from src.agent.tools_revenue import get_revenue_events_summary
from src.agent.tools_stats import stats_client
from src.rag.generator import rag_generate_checklist, rag_generate_email


def prepare_action(
    action: str, question: str, client_features: Dict[str, Any]
) -> Dict[str, Any]:
    if action == "send_retention_email":
        out = rag_generate_email(
            question=question,
            client_features=client_features,
            context_docs="docs_corpus/retention",
        )
        return {
            "type": "email",
            "payload": {"email": out["email"]},
            "internal": {"context": out["internal_context"], "sources": out["sources"]},
        }

    if action == "schedule_call":
        out = rag_generate_checklist(
            question=question,
            client_features=client_features,
            context_docs="docs_corpus/sales",
        )
        return {
            "type": "call_plan",
            "payload": {"checklist": out["checklist"]},
            "internal": {"context": out["internal_context"], "sources": out["sources"]},
        }

    if action == "proposer_upsell":
        out = rag_generate_checklist(
            question=question,
            client_features=client_features,
            context_docs="docs_corpus/sales",
        )
        return {
            "type": "call_plan",
            "payload": {
                "checklist": [
                    "Valider l’objectif business (ce que le client cherche à obtenir).",
                    "Confirmer les usages forts (quelles features apportent le plus de valeur).",
                    "Proposer une montée en gamme alignée (1 bénéfice clair, 1 objection anticipée).",
                    "Offrir un essai / une option simple (si applicable).",
                    "Fixer la prochaine étape (devis / upgrade / date de décision).",
                    *out["checklist"],
                ]
            },
            "internal": {"context": out["internal_context"], "sources": out["sources"]},
        }

    return {
        "type": "none",
        "payload": {"message": "Aucune action préparée"},
        "internal": {"context": "", "sources": []},
    }


def run_agent(question: str, client_id: str) -> Dict[str, Any]:
    """
    Agent décisionnel final.
    Entrée : question + client_id
    Sortie : décision (UNE action) + action préparée (RAG)
    """

    cid = (client_id or "").strip()
    if not cid:
        raise ValueError("client_id is empty")

    # 1) Policy (seuils + règles)
    policy = load_policy()

    # 2) Data tools (réelles)
    st = stats_client(cid)
    rev = get_revenue_events_summary(cid)

    # 3) ML churn (réel, via API /predict)
    churn_pred = predict_churn(cid)

    # 4) Features (strictement celles attendues par les règles)
    features = build_features(
        policy,
        client_stats=st,
        churn_pred=churn_pred,
        revenue_summary=rev,
        reference_date=None,
    )

    # 5) Décision (UNE action)
    decision = decide_action(features, policy)

    # 6) Préparation action (RAG subordonné)
    prepared_action = prepare_action(
        action=decision["action"],
        question=question,
        client_features={
            # On garde les features "règles" + quelques champs utiles non-tech
            **features,
            "client_id": cid,
            "plan": getattr(st, "plan", None),
            "ville": getattr(st, "ville", None),
            "ca_total": getattr(st, "ca_total", None),
            "sessions_total": getattr(st, "sessions_total", None),
            "nb_paiements": getattr(st, "nb_paiements", None),
            "churn_probability": getattr(churn_pred, "churn_probability", None),
            "churn": getattr(churn_pred, "churn", None),
            "churn_risk": getattr(churn_pred, "churn_risk", None),
        },
    )

    return {
        "client_id": cid,
        "decision": {
            "action": decision["action"],
            "priority": decision["priority"],
            "why": decision["rationale"],
            "confidence": decision["confidence"],
            "missing_data": decision["missing_data"],
        },
        "prepared_action": prepared_action,
        # Debug optionnel (tu peux supprimer si tu ne veux rien exposer)
        "debug": {
            "features_rules": features,
            "churn_pred": churn_pred.as_dict(),
        },
    }
