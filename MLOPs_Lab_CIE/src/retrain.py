import os
import json
import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
MODELS_DIR = os.path.join(BASE_DIR, "models")

def main():
    # 1. Load original data to recreate the EXACT test set
    orig_data_path = os.path.join(DATA_DIR, "training_data.csv")
    df_orig = pd.read_csv(orig_data_path)
    
    X_orig = df_orig[["prompt_token_count", "system_prompt_length", "temperature", "is_few_shot"]]
    y_orig = df_orig["response_quality_score"]
    
    X_train_orig, X_test, y_train_orig, y_test = train_test_split(
        X_orig, y_orig, test_size=0.2, random_state=42
    )

    # 2. Load new data
    new_data_path = os.path.join(DATA_DIR, "new_data.csv")
    df_new = pd.read_csv(new_data_path)
    X_new = df_new[["prompt_token_count", "system_prompt_length", "temperature", "is_few_shot"]]
    y_new = df_new["response_quality_score"]

    # 3. Combine training sets (Original Train + New Data)
    # The instruction says "Combine data/training_data.csv + data/new_data.csv"
    # But since we must evaluate on the "same test set", we append new data to the original training data.
    X_train_comb = pd.concat([X_train_orig, X_new], ignore_index=True)
    y_train_comb = pd.concat([y_train_orig, y_new], ignore_index=True)

    # 4. Load Champion Model (Ridge) and compute MAE on the test set
    champion_path = os.path.join(MODELS_DIR, "ridge_model.pkl")
    champion = joblib.load(champion_path)
    champion_preds = champion.predict(X_test)
    champion_mae = mean_absolute_error(y_test, champion_preds)

    # 5. Retrain the SAME model type (Ridge) on the combined dataset
    retrained = Ridge(alpha=1.0)
    retrained.fit(X_train_comb, y_train_comb)
    retrained_preds = retrained.predict(X_test)
    retrained_mae = mean_absolute_error(y_test, retrained_preds)

    # 6. Compare and Decide
    # MAE is an error metric, so lower is better. Improvement is champion - retrained.
    improvement = champion_mae - retrained_mae
    action = "promoted" if improvement > 0 else "kept_champion"

    if action == "promoted":
        joblib.dump(retrained, os.path.join(MODELS_DIR, "retrained_ridge_model.pkl"))

    # 7. Output Results
    output = {
        "original_data_rows": len(df_orig),
        "new_data_rows": len(df_new),
        "combined_data_rows": len(df_orig) + len(df_new),
        "champion_mae": round(float(champion_mae), 4),
        "retrained_mae": round(float(retrained_mae), 4),
        "improvement": round(float(improvement), 4),
        "min_improvement_threshold": 0,
        "action": action,
        "comparison_metric": "mae"
    }

    out_path = os.path.join(RESULTS_DIR, "step4_s8.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(json.dumps(output, indent=2))
    print(f"\nSaved results to: {out_path}")

if __name__ == "__main__":
    main()
