import pandas as pd
import joblib

MODEL_PATH = "src/ml/models/model_current_v1.joblib"
DATA_PATH = "data/ml_ready/df_ml_ready.csv"

df = pd.read_csv(DATA_PATH)

X = df.drop(columns=["ca_total"])
y = df["ca_total"]

model = joblib.load(MODEL_PATH)

pred = model.predict(X)

print("Pred shape:", pred.shape)
print("First 5 predictions:", pred[:5])
print("First 5 actual:", y.values[:5])

client_id = "C008"
row = df[df["client_id"] == client_id]

if row.empty:
    raise ValueError(f"client_id not found: {client_id}")

X_one = row.drop(columns=["ca_total"])
pred_one = model.predict(X_one)[0]
actual_one = row["ca_total"].values[0]

print(f"\nClient {client_id} -> pred: {pred_one:.2f} | actual: {actual_one:.2f}")
