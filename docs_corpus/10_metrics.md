# Définitions & interprétations — métriques SaaS (version non-tech)

## Churn (désabonnement / perte client)
Le churn correspond à la probabilité qu’un client arrête d’utiliser ou de payer.

Signaux fréquents :
- baisse des sessions
- baisse des actions
- inactivité prolongée (last_activity_date ancienne)
- absence de paiements (nb_paiements faible / stagnation)

## Engagement
L’engagement se lit via :
- sessions_total (fréquence)
- actions_total (intensité)
Un client peut avoir beaucoup de sessions mais peu d’actions utiles : attention à la “fausse activité”.

## CA (ca_total)
CA observé sur la période couverte par le dataset.
À utiliser avec :
- nb_paiements (régularité)
- plan (prix / promesse)
- événements revenue_events (one_shot / upsell)

## One-shot
Vente ponctuelle (prestation, service, add-on) non récurrente.
Bon pour augmenter le CA court terme, mais ne prouve pas la fidélité.

## Upsell
Montée en gamme (plan supérieur) ou extension de contrat.
Souvent à proposer quand :
- usage élevé
- churn faible à moyen
- satisfaction implicite (activité stable)

## Extra service
Service additionnel (formation, intégration, migration, support premium).
