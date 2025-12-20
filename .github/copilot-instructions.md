# Copilot Instructions for SaaS PME Data × IA

## Project Overview
- **Architecture:** Data pipelines → ML model training → Dashboards → FastAPI API → Dockerized deployment → (Next: RAG, Agent)
- **Goal:** Demonstrate a reproducible, end-to-end SaaS data/ML/AI stack, focusing on architecture and correctness over business-grade ML.

## Key Components & Data Flow
- **Data:**
  - Raw CSVs in `data/raw/`
  - Processed outputs in `data/processed/`, ML-ready in `data/ml_ready/`
- **ML:**
  - Training scripts in `src/ml/` (e.g., `train_churn_model_v1.py`)
  - Model artifacts in `src/ml/models/`
- **API:**
  - FastAPI app in `src/api/main.py` (serves `/predict` endpoint)
- **Dashboards:**
  - Streamlit apps in `src/dashboards/` (notably `app_v5.py`, `app_v6_churn.py`)
- **RAG & Agent:**
  - RAG logic in `src/rag/` (see `rag_query.py`, `build_index.py`)
  - Agent orchestration in `src/agent/orchestrator.py` (uses tools in `src/agent/`)

## Developer Workflows
- **Train Model:**
  - `python -m src.ml.train_churn_model_v1`
- **Run API:**
  - `python -m uvicorn src.api.main:app --reload`
- **Run Dashboards:**
  - `streamlit run src/dashboards/app_v5.py`
  - `streamlit run src/dashboards/app_v6_churn.py`
- **Docker Compose (full stack):**
  - `docker compose up --build`

## Project-Specific Patterns
- **Agent Tools:**
  - Agent uses modular tools: `tool_rag_query`, `stats_client`, `get_revenue_events_summary`, `predict_churn` (see `src/agent/`)
  - RAG queries use ChromaDB (local vector DB, see `src/rag/rag_query.py`)
  - Answers are formatted for non-technical users (see `format_non_tech_answer` in `src/agent/tools.py`)
- **Data/ML:**
  - Data flows: raw → processed → ML-ready → model → API/dashboard
  - ML artifacts and metrics are versioned in `src/ml/models/` and `data/ml_ready/`
- **Configuration:**
  - RAG index must be built before querying: `python -m src.rag.build_index`
  - ChromaDB files in `chroma_db/`

## Integration & Conventions
- **Imports:** Use relative imports within `src/` modules
- **Entrypoints:** Scripts are run as modules (e.g., `python -m src.ml.train_churn_model_v1`)
- **Artifacts:** All persistent outputs (models, metrics, processed data) are stored in versioned subfolders
- **Docs:**
  - See `docs_corpus/` for internal rules, playbooks, and response formats (notably `40_response_format.md`)

## Examples
- To answer a business question with the agent: see `CopilotAgent.ask()` in `src/agent/orchestrator.py`
- To extend RAG: update `src/rag/build_index.py` and `src/rag/rag_query.py`

---
For further details, see the [README.md](../README.md) and code comments in each module.
