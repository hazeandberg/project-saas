import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import joblib
from pathlib import Path


PATH = "data/ml_ready/df_ml_ready.csv"
df = pd.read_csv(PATH)

print(df.shape)
print(df.head())

X = df.drop(columns=["ca_total"])
y = df["ca_total"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("X_train:", X_train.shape)
print("X_test:", X_test.shape)
print("y_train:", y_train.shape)
print("y_test:", y_test.shape)

### preprocess

num_features = ["nb_paiements", "actions_total", "sessions_total", "anciennete_jours"]
cat_features = ["plan", "ville"]

print("Num features:", num_features)
print("Cat features:", cat_features)
print("X columns:", list(X.columns))

preprocess = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features),
    ]
)

preprocess.fit(X_train)

X_train_p = preprocess.transform(X_train)
X_test_p = preprocess.transform(X_test)

print("X_train_p shape:", X_train_p.shape)
print("X_test_p shape:", X_test_p.shape)

###construire le pipe

model = LinearRegression()

pipeline = Pipeline(
    steps=[
        ("preprocess", preprocess),
        ("model", model),
    ]
)
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

print("Prédictions:", y_pred)
print("Valeurs réelles:", y_test.values)


### Evaluer un modèle

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"MAE  : {mae:.2f}")
print(f"RMSE : {rmse:.2f}")

###exporter le résultat

Path("src/ml/models").mkdir(parents=True, exist_ok=True)

joblib.dump(pipeline, "src/ml/models/model_baseline_v1.joblib")

### export por comparaison modèles dans csv communs.

Path("data/ml_ready").mkdir(parents=True, exist_ok=True)

row = pd.DataFrame([{"model": "LinearRegression", "MAE": mae, "RMSE": rmse}])
out = Path("data/ml_ready/metrics_v1.csv")

row.to_csv(out, mode="a", header=not out.exists(), index=False)
print(f"Metrics saved to: {out}")
