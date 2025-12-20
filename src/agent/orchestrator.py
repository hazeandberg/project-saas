from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from src.agent.tools import tool_rag_query, format_non_tech_answer
from src.agent.tools_revenue import get_revenue_events_summary
from src.agent.tools_stats import stats_client
from src.agent.tools_predict import predict_churn


@dataclass(frozen=True)
class AgentResponse:
    question: str
    client_id: Optional[str]
    answer_md: str
    debug: Dict[str, Any]


class CopilotAgent:
    """
    MVP Agent :
    - RAG pour règles/playbooks
    - stats client (report_oop.csv) si client_id fourni
    - revenue_events si dispo
    - churn via API si configurée
    """

    def ask(self, question: str, client_id: Optional[str] = None) -> AgentResponse:
        q = (question or "").strip()
        if not q:
            raise ValueError("question is empty")

        cid = (client_id or "").strip() or None

        debug: Dict[str, Any] = {}

        # 1) RAG : toujours
        rag = tool_rag_query(q, top_k=4)
        debug["rag_hits"] = [
            {"source": h.source, "chunk": h.chunk, "distance": h.distance}
            for h in rag.hits
        ]

        # 2) Si client_id : enrichir avec données
        confidence = "moyen"
        if cid:
            # stats
            try:
                st = stats_client(cid)
                debug["stats_client"] = st.as_dict()
            except Exception as e:
                debug["stats_client_error"] = str(e)

            # revenue events
            try:
                rev = get_revenue_events_summary(cid)
                debug["revenue_summary"] = rev.as_dict()
            except Exception as e:
                debug["revenue_summary_error"] = str(e)

            # churn prediction (optionnel)
            try:
                pred = predict_churn(cid)
                debug["predict_churn"] = pred.as_dict()
                if pred.churn_risk:
                    # si on a une info de risque, confiance monte (un peu)
                    confidence = "fort"
            except Exception as e:
                debug["predict_churn_error"] = str(e)

        # 3) Synthèse non-tech (MVP basé RAG, puis on enrichira)
        answer_md = format_non_tech_answer(q, rag, confidence=confidence)

        return AgentResponse(
            question=q,
            client_id=cid,
            answer_md=answer_md,
            debug=debug,
        )


if __name__ == "__main__":
    agent = CopilotAgent()
    r = agent.ask("Quand proposer une prestation one-shot ?", client_id=None)
    print(r.answer_md)
