from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


# --- Types stables (interfaces "tool") ---


@dataclass(frozen=True)
class RagHit:
    source: Optional[str]
    chunk: Optional[int]
    distance: float
    text: str


@dataclass(frozen=True)
class RagResult:
    question: str
    hits: List[RagHit]


# --- Tool 1: RAG query (wrapper stable autour de src.rag.rag_query) ---


def tool_rag_query(question: str, top_k: int = 4) -> RagResult:
    """
    Tool stable pour l'Agent :
    - récupère des extraits (chunks) + sources depuis Chroma
    - ne formate pas encore une réponse finale
    """
    from src.rag.rag_query import (
        rag_query,
    )  # import local pour éviter soucis de packaging

    raw: Dict[str, Any] = rag_query(question, top_k=top_k)

    hits: List[RagHit] = []
    for h in raw.get("hits", []):
        hits.append(
            RagHit(
                source=h.get("source"),
                chunk=h.get("chunk"),
                distance=float(h.get("distance", 0.0)),
                text=str(h.get("text", "")),
            )
        )

    return RagResult(question=str(raw.get("question", question)), hits=hits)


# --- Tool 2: Formateur non-tech (MVP) selon docs_corpus/40_response_format.md ---


def format_non_tech_answer(
    question: str,
    rag: RagResult,
    *,
    confidence: str = "moyen",
) -> str:
    """
    MVP "copilote" basé uniquement sur RAG :
    - Résumé
    - Pourquoi (3 bullets max)
    - Action recommandée
    - Action préparée
    - Confiance
    """
    # Sélection du meilleur extrait (hit #1)
    best = rag.hits[0].text.strip() if rag.hits else ""
    src = rag.hits[0].source if rag.hits else None

    # Heuristique MVP: on extrait quelques lignes utiles sans NLP lourd
    why_bullets = []
    if "Quand proposer" in best:
        # cas playbook one-shot
        why_bullets = [
            "Client actif (sessions/actions élevés).",
            "Besoin clair (support, intégration, formation).",
            "Ou opportunité détectée via l’historique one-shot (revenue_events).",
        ]
        action = "Proposer une prestation one-shot packagée (durée + livrable + bénéfice immédiat)."
        prepared = (
            "Email prêt (à personnaliser) :\n"
            "Objet: Proposition rapide pour gagner du temps dès cette semaine\n\n"
            "Bonjour,\n"
            "En regardant votre usage, on voit une opportunité simple pour réduire vos frictions.\n"
            "Je vous propose une prestation one-shot : [nom], durée [X], livrable [Y].\n"
            "Si vous êtes d’accord, je vous envoie un créneau.\n"
        )
        summary = "Vous pouvez proposer une prestation one-shot quand l’usage est fort et que le bénéfice est immédiat."
    else:
        # fallback générique
        summary = (
            "Je retrouve des éléments internes pour répondre et préparer une action."
        )
        why_bullets = [
            "Réponse basée sur le corpus interne (règles + playbooks).",
            "Objectif : action simple, rapide, mesurable.",
            "Format non-tech, directement utilisable.",
        ]
        action = (
            "Appliquer le playbook le plus pertinent et préparer un message de contact."
        )
        prepared = (
            "Action préparée : brouillon email / checklist selon le playbook trouvé."
        )

    # Garde-fous
    confidence = confidence.strip().lower()
    if confidence not in {"faible", "moyen", "fort"}:
        confidence = "moyen"

    # 40_response_format.md
    lines = []
    lines.append("## 1) Résumé")
    lines.append(summary)
    lines.append("")
    lines.append("## 2) Pourquoi")
    for b in why_bullets[:3]:
        lines.append(f"- {b}")
    lines.append("")
    lines.append("## 3) Action recommandée")
    lines.append(action)
    lines.append("")
    lines.append("## 4) Action préparée")
    lines.append(prepared)
    lines.append("")
    lines.append("## 5) Confiance")
    lines.append(confidence)
    if src:
        lines.append(f"\n_Source: {src}_")

    return "\n".join(lines)


# --- Mini test manuel (optionnel) ---
if __name__ == "__main__":
    q = "Quand proposer une prestation one-shot ?"
    rag = tool_rag_query(q, top_k=4)
    print(format_non_tech_answer(q, rag, confidence="fort"))
