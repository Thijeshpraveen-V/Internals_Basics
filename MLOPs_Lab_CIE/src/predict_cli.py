import argparse, json, os
import pandas as pd
import joblib

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "ridge_model.pkl")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt_token_count",   type=float, required=True)
    parser.add_argument("--system_prompt_length", type=float, required=True)
    parser.add_argument("--temperature",          type=float, required=True)
    parser.add_argument("--is_few_shot",          type=int,   required=True)
    args = parser.parse_args()

    model = joblib.load(MODEL_PATH)
    features = pd.DataFrame([{
        "prompt_token_count":   args.prompt_token_count,
        "system_prompt_length": args.system_prompt_length,
        "temperature":          args.temperature,
        "is_few_shot":          args.is_few_shot,
    }])
    prediction = round(float(model.predict(features)[0]), 4)

    result = {
        "image_name": "promptlab-predictor",
        "image_tag":  "v1",
        "base_image": "python:3.10-slim",
        "test_input": {
            "prompt_token_count":   args.prompt_token_count,
            "system_prompt_length": args.system_prompt_length,
            "temperature":          args.temperature,
            "is_few_shot":          args.is_few_shot,
        },
        "prediction": prediction,
    }

    results_dir = os.path.join(BASE_DIR, "results")
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, "step3_s3.json"), "w") as f:
        json.dump(result, f, indent=2)

    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
