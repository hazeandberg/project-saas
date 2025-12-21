# SaaS PME Data × IA — Project 1 (Portfolio Architecture)

This repository is part of a larger macro-project: **“SaaS PME Data × IA”**.
The goal is to demonstrate a complete, reproducible architecture covering:

**Python pipelines → ML → Dashboard → API → Docker → (next) RAG → (next) Agent**

> Note: the datasets are intentionally simplified to focus on **architecture, correctness, and reproducibility** (not business-grade ML performance yet).

---

## Current status — Project 1 (Week 2) ✅
**Delivered components:**
- **Data pipeline outputs** (processed SaaS-like datasets)  
  - `data/processed/report_oop.csv`
  - `data/processed/kpi_by_client.csv`
- **ML-ready dataset (churn)**  
  - `data/ml_ready/df_ml_ready.csv`
  - metrics artifact: `data/ml_ready/churn_metrics_v1.json`
- **Churn model (baseline)**  
  - model artifact: `src/ml/models/churn_model_v1.joblib`
- **FastAPI prediction service**  
  - `src/api/main.py`
  - endpoint: `POST /predict`
- **Streamlit dashboards (iterative builds)**  
  - `src/dashboards/app_v5.py`
  - `src/dashboards/app_v6_churn.py`
- **Docker / Compose (v1)**  
  - `Dockerfile`
  - `docker-compose.yml`
  - `.dockerignore`

---

## Definition — churn_7_30j
Classification target: **`churn_7_30j`** (binary)

Objective:
- detect clients at risk of churn in a **7–30 days** window

Important:
- the dataset is small and imbalanced; the goal is to validate the **end-to-end ML product pipeline**.

---

## Architecture overview
High-level flow:

data/raw (SaaS-like CSV)
→ Python/Pandas/OOP processing (report_oop, KPIs)
→ ML-ready dataset (df_ml_ready)
→ train churn model (joblib)
→ FastAPI /predict
→ Streamlit dashboards (v5 / v6)
→ Docker / Compose

yaml
Copier le code

---

## How to run (local)

### 1) Train churn model (build used by churn dashboards / API)
```bash
python -m src.ml.train_churn_model
This generates:

src/ml/models/churn_model_v1.joblib

data/ml_ready/churn_metrics_v1.json

2) Run the API (FastAPI)
bash
Copier le code
python -m uvicorn src.api.main:app --reload
Healthcheck: GET /health

Docs: /docs

Predict: POST /predict

Example request:

json
Copier le code
{
  "paid_count_before_T": 3,
  "paid_sum_before_T": 237,
  "days_since_last_paid": 31,
  "plan": "pro",
  "ville": "Paris"
}
Example response:

json
Copier le code
{
  "churn_probability": 0.27,
  "churn": false
}
3) Run dashboards (Streamlit)
Dashboards are iterative builds aligned with different ML script variants.
Both v5 and v6 are available and functional:

bash
Copier le code
streamlit run src/dashboards/app_v5.py
streamlit run src/dashboards/app_v6_churn.py
How to run (Docker / Compose v1)
From the repo root:

bash
Copier le code
docker compose up --build
Then test:

http://127.0.0.1:8000/health

http://127.0.0.1:8000/docs

Repository map (active entrypoints)
Data / ML
ML-ready builder: src/ml/build_ml_churn_ready.py

Churn trainer (active): src/ml/train_churn_model.py

Model artifact: src/ml/models/churn_model_v1.joblib

Dataset: data/ml_ready/df_ml_ready.csv

Metrics: data/ml_ready/churn_metrics_v1.json

API
src/api/main.py

Dashboards
src/dashboards/app_v5.py

src/dashboards/app_v6_churn.py

Deploy
Dockerfile, docker-compose.yml, .dockerignore

Planned — Project 1 (Week 3) 🚧
Next deliverables (per official plan):

RAG

internal SaaS documentation corpus

embeddings + Chroma index

retrieve → generate → answer

Agent
Tools:

predict_churn(client_id)

rag_query(question)

stats_client(client_id)
Loop:

input → decision → tool → synthesis

Docker Compose v2

api

vectorstore

(optional) agent worker

Packaging final

README (final)

architecture.md

dashboard screenshots

deployment instructions

Global plan — 3 projects (Official, version 11/12/2025)
Project 1 — Main SaaS (showcase) — 11/12/2025 → 02/01/2026
Data → ML → Dashboard → RAG → Agent

structured Python → Pandas pipeline

ML-ready datasets

ML models (Logistic / RF)

dashboard(s)

RAG on internal SaaS docs

light agent (tools + LLM decision)

Docker / Compose

professional documentation (README + architecture)

Project 2 — Additional ML pipeline — 02/01 → 14/01
Goal: reproduce a complete ML pipeline on another dataset

Pandas pipeline

feature engineering

model comparison

mini dashboard

clean README

Docker minimal (if possible)

Project 3 — Mini AI project (RAG or Agent) — 14/01 → 24/01
Goal: demonstrate modern AI (LLM, RAG, agents)

minimal RAG or autonomous agent

FastAPI or CLI

functional tests

documentation + examples

complete Dockerfile

Global objective — 31/01/2026

3 coherent projects

1 complete main SaaS showcase

1 additional ML pipeline

1 standalone AI project

GitHub portfolio ready for recruiters / senior devs / freelance missions