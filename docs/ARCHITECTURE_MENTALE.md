# 🗺️ Carte Mentale Architecture - SaaS PME Data × IA

## 1️⃣ CARTE MENTALE GLOBALE

```
PROJECT-SAAS
│
├─ 📁 DATA BRUTES (data/raw/)
│  ├─ clients.csv
│  ├─ subscriptions.csv
│  └─ usage.csv
│
├─ 🔄 PIPELINE TRANSFORMATION (src/pipeline/)
│  └─ OOP Processing → report_oop.csv + kpi_by_client.csv
│
├─ 🤖 ML WORKFLOW (src/ml/)
│  ├─ Build ML-ready dataset
│  ├─ Train churn_model_v1
│  └─ Generate metrics + artifacts
│
├─ 📊 DASHBOARDS (src/dashboards/)
│  ├─ app_v5.py (générale)
│  └─ app_v6_churn.py (churn-focused)
│
├─ 🔌 API (src/api/)
│  └─ FastAPI /predict endpoint
│
├─ 🧠 RAG SYSTEM (src/rag/)
│  ├─ Build index (ChromaDB)
│  └─ Query corpus
│
├─ 🤖 AGENT (src/agent/)
│  ├─ orchestrator.py (décisions)
│  ├─ RAG tool
│  ├─ Stats tool
│  ├─ Churn prediction tool
│  └─ Revenue tool
│
└─ 🐳 DOCKER DEPLOYMENT
   └─ docker-compose.yml
```

---

## 2️⃣ FLUX DE LA DATA

```
ÉTAPE 1: RAW DATA READING
  data/raw/*.csv → pandas DataFrames

ÉTAPE 2: CLEANING & PARSING (parse_clean.py)
  ├─ Fix dates
  ├─ Handle nulls
  └─ Type conversion

ÉTAPE 3: OOP PROCESSING (pipeline_oop.py)
  ├─ Group by client
  ├─ Aggregate payments/usage
  └─ Calculate KPIs
  
  📤 OUTPUT: report_oop.csv
           kpi_by_client.csv

ÉTAPE 4: ML-READY BUILDING (build_ml_churn_ready_v1.py)
  ├─ Merge report + KPIs
  ├─ Create features (paid_count, days_since_paid, etc.)
  ├─ Label target (churn_7_30j)
  └─ Train/test split
  
  📤 OUTPUT: df_ml_ready.csv

ÉTAPE 5: MODEL TRAINING (train_churn_model_v1.py)
  ├─ Load df_ml_ready.csv
  ├─ Train Logistic Regression
  ├─ Evaluate metrics
  └─ Save joblib artifact
  
  📤 OUTPUT: churn_model_v1.joblib
           churn_metrics_v1.json

ÉTAPE 6: SERVING
  ├─ API loads model → /predict endpoint
  ├─ Dashboards load report_oop + metrics
  └─ Agent uses all for decision-making
```

---

## 3️⃣ INTERACTIONS FICHIERS/SCRIPTS

```
MAIN ENTRY POINTS:

1️⃣ python -m src.ml.train_churn_model_v1
   │
   ├─ reads: data/ml_ready/df_ml_ready.csv
   ├─ imports: src.pipeline.* (processing logic)
   ├─ trains: Logistic Regression
   │
   └─ outputs:
      ├─ src/ml/models/churn_model_v1.joblib
      └─ data/ml_ready/churn_metrics_v1.json

2️⃣ python -m uvicorn src.api.main:app
   │
   ├─ imports: src.ml.models/churn_model_v1.joblib
   ├─ exposes: POST /predict
   │
   └─ request format:
      {paid_count_before_T, paid_sum_before_T, 
       days_since_last_paid, plan, ville}

3️⃣ streamlit run src/dashboards/app_v6_churn.py
   │
   ├─ reads: data/processed/report_oop.csv
   ├─ reads: data/ml_ready/churn_metrics_v1.json
   ├─ loads: src/ml/models/churn_model_v1.joblib
   │
   └─ displays: KPIs + churn predictions

4️⃣ python -m src.agent.orchestrator
   │
   ├─ imports: src.agent.tools_*.py
   ├─ reads: data/processed/report_oop.csv (for stats)
   ├─ queries: ChromaDB (docs_corpus → RAG)
   ├─ calls: http://api:8000/predict (churn predictions)
   │
   └─ returns: AgentResponse(question, answer_md, debug)
```

---

## 4️⃣ LOGIQUE RAG + AGENT

```
USER QUESTION
      │
      ▼
AGENT.ask(question, client_id)
      │
      ├─ 🔍 TOOL 1: RAG_QUERY
      │  └─ Embed question → search ChromaDB (docs_corpus)
      │     → return top-4 chunks (règles/playbooks)
      │
      ├─ 👤 TOOL 2: STATS_CLIENT (si client_id fourni)
      │  └─ Read report_oop.csv
      │     → filter by client_id
      │     → return {nb_paiements, ca_total, plan, ville, etc.}
      │
      ├─ 📈 TOOL 3: REVENUE_EVENTS (si client_id fourni)
      │  └─ Lookup client events
      │     → return revenue summary
      │
      ├─ ⚠️ TOOL 4: PREDICT_CHURN (si client_id fourni)
      │  ├─ Get stats via tool_stats
      │  ├─ POST to /predict API
      │  └─ return {churn_probability, churn_risk}
      │
      └─ 🎯 FORMAT ANSWER (non-tech format)
         └─ return AgentResponse:
            • Résumé
            • Pourquoi (3 bullets)
            • Action recommandée
            • Action préparée
            • Confiance
```

---

## 5️⃣ POINT DE VUE UTILISATEUR

```
SCENARIO 1: Je veux entraîner un modèle de churn
  $ python -m src.ml.train_churn_model_v1
  
  ✅ LE CODE FAIT:
     1. Lit les données brutes
     2. Crée features ML
     3. Entraîne un modèle
     4. Sauvegarde le modèle (artifact)
     5. Évalue performance (metrics)
  
  📦 FICHIERS CRÉÉS:
     - src/ml/models/churn_model_v1.joblib
     - data/ml_ready/churn_metrics_v1.json


SCENARIO 2: Je veux voir les prédictions en temps réel
  $ python -m uvicorn src.api.main:app --reload
  
  ✅ LE CODE FAIT:
     1. Démarre un serveur web
     2. Charge le modèle en mémoire
     3. Accepte requêtes JSON
     4. Retourne prédictions
  
  🔗 URL:
     POST http://localhost:8000/predict
     GET  http://localhost:8000/docs (interactive docs)


SCENARIO 3: Je veux visualiser les métriques
  $ streamlit run src/dashboards/app_v6_churn.py
  
  ✅ LE CODE FAIT:
     1. Lit le rapport de clients
     2. Lit les métriques du modèle
     3. Affiche graphiques interactifs
  
  📊 AFFICHAGE:
     - KPIs par client
     - Distribution churn
     - Comparaisons


SCENARIO 4: Je veux poser une question commerciale
  >>> agent = CopilotAgent()
  >>> agent.ask("Comment éviter la fuite client ?")
  
  ✅ LE CODE FAIT:
     1. Cherche dans la base de connaissances (RAG)
     2. Retourne règles + playbooks
     3. Formatte réponse pour non-tech
  
  💡 RÉSULTAT:
     Réponse structurée avec:
     - Explication simple
     - Actions concrètes
     - Confiance (faible/moyen/fort)
```

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
- **ML Training** → `src/ml/train_churn_model_v1.py`
- **API** → `src/api/main.py`
- **Dashboards** → `src/dashboards/app_v6_churn.py`
- **RAG** → `src/rag/rag_query.py` + `src/rag/build_index.py`
- **Agent** → `src/agent/orchestrator.py`
- **Corpus** → `docs_corpus/` (40_response_format.md, 30_playbooks_retention.md, etc.)
