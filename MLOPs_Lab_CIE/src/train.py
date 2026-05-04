import os, json
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH   = os.path.join(BASE_DIR, "data", "training_data.csv")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
MODELS_DIR  = os.path.join(BASE_DIR, "models")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR,  exist_ok=True)

MLFLOW_TRACKING_URI = "file:///" + os.path.join(BASE_DIR, "mlruns").replace("\\", "/")
EXPERIMENT_NAME     = "promptlab-response-quality-score"

def compute_metrics(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mae  = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2   = float(r2_score(y_true, y_pred))
    mask = y_true != 0
    mape = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)
    return {"mae": mae, "rmse": rmse, "r2": r2, "mape": mape}

def main():
    df = pd.read_csv(DATA_PATH)
    X  = df[["prompt_token_count", "system_prompt_length", "temperature", "is_few_shot"]]
    y  = df["response_quality_score"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    candidates = {
        "Ridge":        Ridge(alpha=1.0),
        "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
    }

    results = []
    for name, model in candidates.items():
        with mlflow.start_run(run_name=name):
            mlflow.set_tag("experiment_type", "baseline_comparison")
            mlflow.log_params(model.get_params())
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            m = compute_metrics(y_test.values, y_pred)
            mlflow.log_metrics(m)
            joblib.dump(model, os.path.join(MODELS_DIR, f"{name.lower()}_model.pkl"))
            mlflow.sklearn.log_model(sk_model=model, artifact_path=name)
            results.append({"name": name, **m})
            print(f"{name}: MAE={m['mae']:.4f}  RMSE={m['rmse']:.4f}  R2={m['r2']:.4f}  MAPE={m['mape']:.4f}")

    best = min(results, key=lambda x: x["rmse"])
    output = {
        "experiment_name": EXPERIMENT_NAME,
        "models": [
            {
                "name": r["name"],
                "mae":  round(r["mae"],  4),
                "rmse": round(r["rmse"], 4),
                "r2":   round(r["r2"],   4),
                "mape": round(r["mape"], 4),
            }
            for r in results
        ],
        "best_model":        best["name"],
        "best_metric_name":  "rmse",
        "best_metric_value": round(best["rmse"], 4),
    }
    out_path = os.path.join(RESULTS_DIR, "step1_s1.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nBest model: {best['name']} (RMSE={best['rmse']:.4f})")
    print(json.dumps(output, indent=2))
    print(f"\nSaved results to: {out_path}")

if __name__ == "__main__":
    main()