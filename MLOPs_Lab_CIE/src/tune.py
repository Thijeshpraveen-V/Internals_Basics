import os
import json
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "training_data.csv")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

MLFLOW_TRACKING_URI = "file:///" + os.path.join(BASE_DIR, "mlruns").replace("\\", "/")
EXPERIMENT_NAME = "promptlab-response-quality-score"

def main():
    # 1. Load data
    df = pd.read_csv(DATA_PATH)
    X = df[["prompt_token_count", "system_prompt_length", "temperature", "is_few_shot"]]
    y = df["response_quality_score"]
    
    # 2. Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    # 3. Setup Hyperparameter Grid
    # The assignment instructs to "Tune the best model from Task 1" but provides a grid
    # specific to tree-based models. We will tune RandomForestRegressor.
    param_grid = {
        "n_estimators": [50, 100, 200],
        "max_depth": [5, 10, None],
        "min_samples_split": [2, 5]
    }

    rf = RandomForestRegressor(random_state=42)

    # We need to select the best config by RMSE and also log MAE
    search = RandomizedSearchCV(
        estimator=rf,
        param_distributions=param_grid,
        n_iter=10,
        scoring=['neg_root_mean_squared_error', 'neg_mean_absolute_error'],
        refit='neg_root_mean_squared_error',
        cv=5,
        random_state=42,
        n_jobs=-1
    )

    # 4. Run tuning with MLflow tracking
    with mlflow.start_run(run_name="tuning-promptlab") as parent_run:
        search.fit(X_train, y_train)
        
        cv_results = search.cv_results_
        total_trials = len(cv_results['params'])
        
        # Log each trial as a nested run
        for i in range(total_trials):
            with mlflow.start_run(run_name=f"trial_{i}", nested=True):
                mlflow.log_params(cv_results['params'][i])
                mlflow.log_metric("rmse", -cv_results['mean_test_neg_root_mean_squared_error'][i])
                mlflow.log_metric("mae", -cv_results['mean_test_neg_mean_absolute_error'][i])
                
        # 5. Evaluate best model
        best_model = search.best_estimator_
        y_pred = best_model.predict(X_test)
        
        test_mae = float(mean_absolute_error(y_test, y_pred))
        
        best_index = search.best_index_
        best_cv_mae = float(-cv_results['mean_test_neg_mean_absolute_error'][best_index])
        
        mlflow.sklearn.log_model(sk_model=best_model, artifact_path="tuned_model")
        joblib.dump(best_model, os.path.join(MODELS_DIR, "tuned_rf_model.pkl"))
        
        # 6. Save results
        output = {
            "search_type": "random",
            "n_folds": 5,
            "total_trials": total_trials,
            "best_params": search.best_params_,
            "best_mae": round(test_mae, 4),
            "best_cv_mae": round(best_cv_mae, 4),
            "parent_run_name": "tuning-promptlab"
        }
        
        out_path = os.path.join(RESULTS_DIR, "step2_s2.json")
        with open(out_path, "w") as f:
            json.dump(output, f, indent=2)
            
        print(f"Best params: {search.best_params_}")
        print(f"Saved results to: {out_path}")
        print(json.dumps(output, indent=2))

if __name__ == "__main__":
    main()