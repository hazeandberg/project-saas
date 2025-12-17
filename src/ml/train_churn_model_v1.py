from __future__ import annotations

import os
import json
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    classification_report,
)


PATH_IN = "data/ml_ready/df_ml_ready.csv"
MODEL_OUT = "src/ml/models/churn_model_v1.joblib"
METRICS_OUT = "data/ml_ready/churn_metrics_v1.json"

TARGET = "churn_7_30j"

NUM_COLS = ["paid_count_before_T", "paid_sum_before_T", "days_since_last_paid"]
CAT_COLS = ["plan", "ville"]


def temporal_split(
    df: pd.DataFrame, test_size: float = 0.3
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split temporel simple + garde-fou:
    - on trie par days_since_last_paid
    - on prend les derniers X% en test
    - MAIS on force la présence des 2 classes dans le train si possible
    """
    df_sorted = df.sort_values("days_since_last_paid").reset_index(drop=True)
    n = len(df_sorted)
    n_test = max(1, int(round(n * test_size)))

    train = df_sorted.iloc[: n - n_test].copy()
    test = df_sorted.iloc[n - n_test :].copy()

    # Garde-fou classes: train doit contenir 0 et 1
    classes_train = set(train["churn_7_30j"].astype(int).unique())
    classes_all = set(df_sorted["churn_7_30j"].astype(int).unique())

    if classes_all == {0, 1} and classes_train != {0, 1}:
        # Si le seul "1" est dans test, on le déplace dans train (split minimal)
        ones_in_test = test[test["churn_7_30j"].astype(int) == 1]
        if len(ones_in_test) > 0:
            row = ones_in_test.iloc[[0]]
            test = test.drop(row.index)
            train = pd.concat([train, row], ignore_index=True)

    return train, test


def main() -> None:
    df = pd.read_csv(PATH_IN)

    # Sécurité minimale
    expected = set(NUM_COLS + CAT_COLS + [TARGET])
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"Colonnes manquantes dans {PATH_IN}: {sorted(missing)}")

    # Split temporel (pas de random)
    train_df, test_df = temporal_split(df, test_size=0.3)

    X_train = train_df[NUM_COLS + CAT_COLS]
    y_train = train_df[TARGET].astype(int)

    X_test = test_df[NUM_COLS + CAT_COLS]
    y_test = test_df[TARGET].astype(int)

    # Pipeline
    pre = ColumnTransformer(
        transformers=[
            ("num", "passthrough", NUM_COLS),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_COLS),
        ]
    )

    clf = LogisticRegression(max_iter=1000)

    pipe = Pipeline(steps=[("pre", pre), ("clf", clf)])

    pipe.fit(X_train, y_train)

    # Évaluation
    y_pred = pipe.predict(X_test)

    cm = confusion_matrix(y_test, y_pred).tolist()
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)

    report_txt = classification_report(y_test, y_pred, zero_division=0)

    metrics = {
        "n_rows": int(len(df)),
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "target": TARGET,
        "confusion_matrix": cm,
        "precision": float(precision),
        "recall": float(recall),
        "classification_report": report_txt,
    }

    os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)
    joblib.dump(pipe, MODEL_OUT)

    with open(METRICS_OUT, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print(f"Model saved   : {MODEL_OUT}")
    print(f"Metrics saved : {METRICS_OUT}")
    print("Confusion matrix:", cm)
    print(f"Precision: {precision:.3f} | Recall: {recall:.3f}")
    print(report_txt)


if __name__ == "__main__":
    main()
