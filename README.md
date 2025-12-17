# SaaS ML Project — Churn Prediction

## Overview
This project demonstrates a complete SaaS-oriented ML pipeline:
dataset preparation, churn modeling, API deployment, and dashboarding.

## Churn Prediction API
🔮 Churn Prediction API (FastAPI)
🎯 Objectif

Cette API expose un modèle de classification churn permettant d’estimer le risque de churn entre 7 et 30 jours pour un client SaaS, à partir de données d’abonnement.

⚠️ Le dataset est volontairement limité : l’objectif est de démontrer une architecture ML complète et déployable, pas une performance business optimale.

🧠 Modèle

Type : Classification binaire

Cible : churn_7_30j

Algorithme : Logistic Regression (baseline explicable)

Sortie :

probabilité de churn

décision binaire (churn / non churn)

🚀 Lancer l’API

Depuis la racine du projet :

python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload


Healthcheck : http://127.0.0.1:8000/health

Documentation interactive : http://127.0.0.1:8000/docs

📥 Endpoint /predict
Requête (JSON)
{
  "paid_count_before_T": 3,
  "paid_sum_before_T": 237,
  "days_since_last_paid": 31,
  "plan": "pro",
  "ville": "Paris"
}

Réponse (JSON)
{
  "churn_probability": 0.27,
  "churn": false
}

🧩 Cas d’usage

priorisation des clients à risque

support / rétention proactive

intégration CRM ou dashboard

brique décisionnelle pour agent IA

🧭 Positionnement du projet

Ce projet fait partie d’un pipeline SaaS Data → ML → API → Dashboard, avec un accent sur :

rigueur temporelle (pas de fuite de données)

clarté des hypothèses métier

déploiement réaliste (API + Docker)