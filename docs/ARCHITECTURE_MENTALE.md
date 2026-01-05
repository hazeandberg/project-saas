# 🗺️ Carte Mentale Architecture — SaaS PME Data × IA

> **Documentation système complète** pour la stack Data → ML → API → RAG → Agent

---

## ⚡ Quick Start

### Local Environment

```powershell
# 1️⃣ Build RAG index (local vector DB)
python -m src.rag.build_index

# 2️⃣ Train churn model
python -m src.ml.train_churn_model

# 3️⃣ Run API locally
python -m uvicorn src.api.main:app --reload

# 4️⃣ Test health (new terminal)
Invoke-RestMethod -Uri 'http://127.0.0.1:8000/health'

# 5️⃣ Example prediction
$body = @{
  paid_count_before_T = 3; paid_sum_before_T = 237
  days_since_last_paid = 31; plan = 'pro'; ville = 'Paris'
}
Invoke-RestMethod -Method Post -Uri 'http://127.0.0.1:8000/predict' `
  -Body ($body | ConvertTo-Json) -ContentType 'application/json'
```

### Docker Environment

```powershell
# 1️⃣ Build & run full stack
docker compose up --build

# 2️⃣ Test API health
Invoke-RestMethod -Uri 'http://127.0.0.1:8000/health'

# 3️⃣ Example prediction (Docker)
$body = @{
  paid_count_before_T = 2; paid_sum_before_T = 150
  days_since_last_paid = 21; plan = 'basic'; ville = 'Lyon'
}
Invoke-RestMethod -Method Post -Uri 'http://127.0.0.1:8000/predict' `
  -Body ($body | ConvertTo-Json) -ContentType 'application/json'

# Note: Agent intra-network: set PREDICT_API_URL=http://api:8000/predict
```

---

## 1️⃣ Architecture Globale

### Component Diagram

```
🏗️  PROJECT-SAAS
    │
    ├─📁 DATA BRUTES (data/raw/)
    │  ├─ clients.csv, subscriptions.csv, usage.csv
    │
    ├─🔄 PIPELINE TRANSFORMATION (src/pipeline/)
    │  └─ OOP Processing → report_oop.csv + KPIs
    │
    ├─🤖 ML WORKFLOW (src/ml/)
    │  ├─ Build ML-ready: df_ml_churn_ready.csv
    │  ├─ Train: churn_model_v1.joblib
    │  └─ Metrics: churn_metrics_v1.json
    │
    ├─📊 DASHBOARDS (src/dashboards/)
    │  ├─ app_v5.py (générale)
    │  └─ app_v6_churn.py (churn-focused)
    │
    ├─🔌 API SERVING (src/api/)
    │  └─ FastAPI /predict endpoint
    │
    ├─🧠 RAG SYSTEM (src/rag/)
    │  ├─ build_index.py: ChromaDB vectors
    │  └─ rag_query.py: semantic search
    │
    ├─🤖 AGENT DÉCISIONNEL (src/agent/)
    │  ├─ orchestrator.py: main entrypoint
    │  ├─ rules_engine.py: AST-safe policy evaluation
    │  ├─ tools_*.py: stats, churn, revenue, RAG
    │  └─ generator.py: email/checklist output
    │
    └─🐳 DEPLOYMENT (Docker)
       ├─ Dockerfile + docker-compose.yml
       └─ .dockerignore (excludes chroma_db/, preserves docs_corpus/)
```

---

## 2️⃣ Data Flow Pipeline

### Step-by-Step

```
┌─────────────────────────────────────────────────────────────┐
│ ÉTAPE 1: RAW DATA INTAKE                                    │
├─────────────────────────────────────────────────────────────┤
│ Input:  data/raw/{clients, subscriptions, usage}.csv        │
│ Action: Load into Pandas DataFrames                         │
└─────────────────────────────────────────────────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────────┐
│ ÉTAPE 2: PARSE & CLEAN (src/pipeline/parse_clean.py)       │
├─────────────────────────────────────────────────────────────┤
│ • Fix dates, handle nulls, type conversion                  │
│ • Minimal filtering, output: list[dict]                     │
└─────────────────────────────────────────────────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────────┐
│ ÉTAPE 3: OOP PIPELINE (src/pipeline/pipeline_oop.py)        │
├─────────────────────────────────────────────────────────────┤
│ • Group by client_id                                        │
│ • Aggregate: payments, usage, KPIs                          │
│ • Merge: plan, ville from raw data                          │
│ Output: report_oop.csv, kpi_by_client.csv                   │
└─────────────────────────────────────────────────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────────┐
│ ÉTAPE 4: ML-READY DATASET (src/ml/build_ml_churn_ready.py) │
├─────────────────────────────────────────────────────────────┤
│ Input: subscriptions.csv (PRE-T), report_oop.csv            │
│ • Fixed reference date (T)                                  │
│ • Features: paid_count_before_T, days_since_last_paid       │
│ • Label: churn_7_30j (via [T+7, T+30] payment window)       │
│ • Merge: plan, ville                                        │
│ Output: df_ml_churn_ready.csv                               │
└─────────────────────────────────────────────────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────────┐
│ ÉTAPE 5: MODEL TRAINING (src/ml/train_churn_model.py)       │
├─────────────────────────────────────────────────────────────┤
│ Input: df_ml_churn_ready.csv                                │
│ • Temporal split (train/test)                               │
│ • OneHotEncoder(plan, ville) + LogisticRegression           │
│ • Evaluate: precision, recall, confusion matrix             │
│ Outputs:                                                     │
│   - src/ml/models/churn_model_v1.joblib                     │
│   - data/ml_ready/churn_metrics_v1.json                     │
└─────────────────────────────────────────────────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────────┐
│ ÉTAPE 6: SERVING & CONSUMPTION                              │
├─────────────────────────────────────────────────────────────┤
│ • FastAPI loads model → /predict endpoint                   │
│ • Dashboards: load report_oop + metrics → visualize         │
│ • Agent: uses all components for decision-making            │
└─────────────────────────────────────────────────────────────┘
```

---

## 3️⃣ Main Entry Points

### Scripts & Services

| # | Command | Purpose | Inputs | Outputs |
|:-:|----------|---------|--------|---------|
| 1️⃣ | `python -m src.rag.build_index` | 🧠 Build vector DB | `docs_corpus/` | `chroma_db/` |
| 2️⃣ | `python -m src.ml.build_ml_churn_ready` | 📊 Prepare ML dataset | `subscriptions.csv`<br/>`report_oop.csv` | `df_ml_churn_ready.csv` |
| 3️⃣ | `python -m src.ml.train_churn_model` | 🤖 Train churn model | `df_ml_churn_ready.csv` | `churn_model_v1.joblib`<br/>`churn_metrics_v1.json` |
| 4️⃣ | `python -m uvicorn src.api.main:app` | 🔌 Run API server | `churn_model_v1.joblib` | `GET /health`<br/>`POST /predict` |
| 5️⃣ | `streamlit run src/dashboards/app_v6_churn.py` | 📈 Visualize dashboards | `report_oop.csv`, metrics | Interactive web UI |
| 6️⃣ | `from src.agent.orchestrator import CopilotAgent` | 🎯 Run agent | Policy, stats, RAG, API | Decision + action |

---

## 4️⃣ Agent Logic (RAG + Decision)

```
INPUT: question + client_id
      │
      ▼
1) Collecte data client
   - stats_client(client_id) via report_oop.csv
   - revenue summary (si fichier/outil branché)
   - churn_pred via API /predict (si PREDICT_API_URL défini)

2) Feature engineering (features_rules)
   - usage_level
   - churn_risk
   - ca_total_high
   - recent_one_shot

3) Décision métier (rules engine)
   - apply_policy(policy, features_rules)
   - UNE action déterministe + priorité + rationale

4) RAG = préparation de l’action
   - rag_query(...) → extrait playbooks / format réponse
   - génère un email OU une checklist / plan d’appel
   - le RAG n’a pas le droit de changer l’action

OUTPUT (non-tech):
   - decision: action, priorité, pourquoi, confiance
   - prepared_action: email/checklist + contexte interne séparé
   - debug: traces (features, churn_pred, etc.)

```

---

## 5️⃣ User Scenarios

### 1. Train a Churn Model
```powershell
$ python -m src.ml.train_churn_model
```
✅ **Creates:**
- `src/ml/models/churn_model_v1.joblib`
- `data/ml_ready/churn_metrics_v1.json`

---

### 2. Real-Time Predictions (API)
```powershell
$ python -m uvicorn src.api.main:app --reload
```
✅ **Access:**
- `GET  http://localhost:8000/health`
- `GET  http://localhost:8000/docs` (Swagger)
- `POST http://localhost:8000/predict`

---

### 3. Visualize Metrics (Dashboard)
```powershell
$ streamlit run src/dashboards/app_v6_churn.py
```
✅ **Displays:**
- KPIs by client
- Churn distribution
- Model performance comparisons

---

### 4. Ask a Business Question (Agent)
```python
from src.agent.orchestrator import CopilotAgent

agent = CopilotAgent()
response = agent.ask("How to reduce churn?", client_id="C123")

print(response.answer_md)          # User-friendly summary
print(response.debug)              # Internal details (RAG hits, etc.)
```
✅ **Returns:**
- Structured answer (rules + RAG-enhanced)
- Confidence level
- Actionable recommendations

---

## ⚡ SYNTHÈSE POINTS CLÉS

| Aspect | Description |
|--------|-------------|
| **Données** | CSV → Pandas → Agrégation OOP → Report structuré |
| **ML** | Features + Label → Logistic Regression → Model joblib |
| **Serving** | FastAPI charge model → /predict endpoint |
| **Dashboards** | Streamlit lit report_oop + metrics → affiche KPIs |
| **Intelligence** | Agent utilise RAG (docs) + Stats (report) + Prédictions (API) |
| **Déploiement** | Docker Compose lance tout ensemble |

---

## 🔗 RÉFÉRENCES FICHIERS CLÉS

- **Pipeline Data** → `src/pipeline/pipeline_oop.py`
- **ML Training** → `src/ml/train_churn_model.py`
- **API** → `src/api/main.py`
- **Dashboards** → `src/dashboards/app_v6_churn.py`
- **RAG** → `src/rag/rag_query.py` + `src/rag/build_index.py`
- **Agent** → `src/agent/orchestrator.py`
- **Corpus** → `docs_corpus/` (40_response_format.md, 30_playbooks_retention.md, etc.)

---

## 6️⃣ NOTES RUNTIME

6️⃣ NOTES RUNTIME (corrigé)

API endpoints: GET /health, POST /predict

Model load: src/api/main.py charge src/ml/models/churn_model_v1.joblib et lève une erreur si absent.

PREDICT_API_URL (obligatoire pour predict_churn)
Ton tools_predict.py lève une erreur si la variable n’est pas définie.

Local :

PREDICT_API_URL=http://127.0.0.1:8000/predict

Docker intra-compose :

PREDICT_API_URL=http://api:8000/predict

RAG index: requis avant requêtes RAG

commande: python -m src.rag.build_index

stockage local: chroma_db/ (exclu de l’image via .dockerignore)

erreur explicite si collection absente dans rag_query.py

Docker compose actuel: 1 service api (pas de stack multi-services).

```bash
python -m src.rag.build_index
```

   - Stockage local: `chroma_db/` (exclu de l'image via `.dockerignore`).
   - Erreur claire si collection absente: voir [src/rag/rag_query.py](src/rag/rag_query.py).
- **Déploiement:** `.dockerignore` exclut `chroma_db/` et conserve `docs_corpus/` pour le RAG runtime.
