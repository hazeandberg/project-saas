# src/rag/generator.py

from __future__ import annotations

from typing import Any, Dict, List

from src.rag.rag_query import rag_query


def _format_hits_as_context(
    hits: List[Dict[str, Any]], *, max_hits: int = 4, max_chars: int = 900
) -> str:
    """
    Construit un bloc "Contexte interne" à partir des hits Chroma.
    Déterministe, compact, prêt à être collé dans un email / doc.
    """
    if not hits:
        return ""

    lines: List[str] = ["", "Contexte interne (extraits du corpus):"]
    used = 0
    for i, h in enumerate(hits[:max_hits], start=1):
        text = (h.get("text") or "").strip()
        if not text:
            continue

        source = h.get("source")
        chunk = h.get("chunk")
        dist = h.get("distance")

        header_parts = []
        if source:
            header_parts.append(str(source))
        if chunk is not None:
            header_parts.append(f"chunk={chunk}")
        if dist is not None:
            try:
                header_parts.append(f"dist={float(dist):.4f}")
            except Exception:
                header_parts.append(f"dist={dist}")

        header = " | ".join(header_parts) if header_parts else "source inconnue"

        # normalisation espaces
        text_one_line = " ".join(text.split())

        remaining = max_chars - used
        if remaining <= 0:
            break

        snippet = text_one_line[: min(len(text_one_line), remaining)]
        used += len(snippet)

        lines.append(f"- ({i}) {header}: {snippet}")

    return "\n".join(lines)


def rag_generate_email(
    *,
    question: str,
    client_features: Dict[str, Any],
    context_docs: str = "docs_corpus",
    top_k: int = 4,
) -> Dict[str, Any]:
    cid = str(client_features.get("client_id", "") or "").strip()
    plan = str(client_features.get("plan", "") or "").strip()
    ville = str(client_features.get("ville", "") or "").strip()

    churn_risk = str(client_features.get("churn_risk", "medium") or "medium")
    usage_level = str(client_features.get("usage_level", "medium") or "medium")

    rq = rag_query(question=f"[{context_docs}] {question}", top_k=top_k)
    hits = rq.get("hits") or []
    context_block = _format_hits_as_context(hits, max_hits=top_k)

    if churn_risk == "high":
        subject = "On peut vous aider à relancer l’usage"
    elif churn_risk == "low" and usage_level == "high":
        subject = "Optimiser encore votre usage"
    else:
        subject = "Point rapide sur votre usage"

    email = (
        f"Objet : {subject}\n\n"
        "Bonjour,\n\n"
        "Je vous contacte car nous avons observé des signaux d’usage qui méritent un point rapide.\n"
        f"- Compte : {cid}\n"
        f"- Plan : {plan}\n"
        f"- Ville : {ville}\n"
        f"- Niveau d’usage : {usage_level}\n"
        f"- Risque churn : {churn_risk}\n\n"
        "Proposition : un échange de 10 minutes pour identifier 1 action simple à appliquer cette semaine.\n\n"
        "Souhaitez-vous que je vous propose deux créneaux ?\n\n"
        "Cordialement,\n"
    )

    return {
        "email": email,
        "internal_context": context_block.strip(),
        "sources": hits,
    }


def rag_generate_checklist(
    *,
    question: str,
    client_features: Dict[str, Any],
    context_docs: str = "docs_corpus",
    top_k: int = 4,
) -> Dict[str, Any]:
    """
    Checklist actionnable (déterministe) + contexte interne séparé.
    Retour:
      {
        "checklist": [str, ...],
        "internal_context": str,
        "sources": hits
      }
    """
    churn_risk = str(client_features.get("churn_risk", "medium") or "medium")
    usage_level = str(client_features.get("usage_level", "medium") or "medium")

    rq = rag_query(question=f"[{context_docs}] {question}", top_k=top_k)
    hits = rq.get("hits") or []
    context_block = _format_hits_as_context(hits, max_hits=top_k)

    checklist: List[str] = []

    if churn_risk == "high":
        checklist.append("Priorité élevée : traiter sous 24–48h.")

    checklist.extend(
        [
            "Confirmer le contexte (plan, usage, objectif principal).",
            "Qualifier le problème principal (1 phrase, mesurable).",
            "Identifier la cause probable (onboarding, valeur perçue, timing, friction).",
            "Proposer 1 action immédiate (simple, testable en 7 jours).",
            "Fixer un suivi (date + critère de succès).",
        ]
    )

    if usage_level == "low":
        checklist.insert(
            3,
            "Vérifier l’activation : quelles fonctionnalités sont réellement utilisées ?",
        )

    return {
        "checklist": checklist,
        "internal_context": context_block.strip(),
        "sources": hits,
    }
