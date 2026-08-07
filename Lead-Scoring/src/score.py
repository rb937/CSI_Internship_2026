# Usage: python src/score.py --input path_to/new_leads.csv --output path_to/scored_leads.csv

import argparse
import joblib
import pandas as pd

from preprocessing import clean_raw_leads, encode_features, align_columns

MODEL_DIR = "./data"


def load_artifacts():
    model = joblib.load(f"{MODEL_DIR}/logistic_model.pkl")
    scaler = joblib.load(f"{MODEL_DIR}/scaler.pkl")
    feature_columns = joblib.load(f"{MODEL_DIR}/feature_columns.pkl")
    numeric_columns = joblib.load(f"{MODEL_DIR}/numeric_columns.pkl")
    return model, scaler, feature_columns, numeric_columns


def score_leads(df_raw: pd.DataFrame) -> pd.DataFrame:
    model, scaler, feature_columns, numeric_columns = load_artifacts()

    df_clean = clean_raw_leads(df_raw)
    df_encoded = encode_features(df_clean)

    df_encoded = df_encoded.drop(columns=["Converted"], errors="ignore")

    df_aligned = align_columns(df_encoded, feature_columns)
    df_aligned[numeric_columns] = scaler.transform(df_aligned[numeric_columns])

    probabilities = model.predict_proba(df_aligned)[:, 1]

    result = df_raw.copy()
    result["lead_score"] = (probabilities * 100).round(1)
    return result.sort_values("lead_score", ascending=False).reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description="Score new leads for conversion likelihood.")
    parser.add_argument("--input", required=True, help="Path to CSV of new leads")
    parser.add_argument("--output", required=True, help="Path to write scored CSV")
    args = parser.parse_args()

    df_raw = pd.read_csv(args.input)
    scored = score_leads(df_raw)
    scored.to_csv(args.output, index=False)
    print(f"Scored {len(scored)} leads -> {args.output}")


if __name__ == "__main__":
    main()
