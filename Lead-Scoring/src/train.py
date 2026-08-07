import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from xgboost import XGBClassifier

from preprocessing import clean_raw_leads, encode_features

RANDOM_STATE = 42
DATA_PATH = "./dataset/LeadScoring.csv"
OUTPUT_DIR = "./data"


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading raw data...")
    df = pd.read_csv(DATA_PATH)

    print("Cleaning...")
    df_clean = clean_raw_leads(df)

    print("Encoding...")
    df_encoded = encode_features(df_clean)

    X = df_encoded.drop(columns=["Converted"])
    y = df_encoded["Converted"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    num_cols = X.select_dtypes(include=["number"]).columns.tolist()
    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    X_train_scaled[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_test_scaled[num_cols] = scaler.transform(X_test[num_cols])

    print("Training Logistic Regression...")
    log_reg = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE)
    log_reg.fit(X_train_scaled, y_train)
    lr_proba = log_reg.predict_proba(X_test_scaled)[:, 1]
    print(f"  Logistic Regression — Accuracy: {accuracy_score(y_test, log_reg.predict(X_test_scaled)):.3f}, "
          f"ROC-AUC: {roc_auc_score(y_test, lr_proba):.3f}")

    print("Training XGBoost...")
    xgb_model = XGBClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.1,
        scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
        eval_metric="logloss", random_state=RANDOM_STATE
    )
    xgb_model.fit(X_train, y_train)
    xgb_proba = xgb_model.predict_proba(X_test)[:, 1]
    print(f"  XGBoost — Accuracy: {accuracy_score(y_test, xgb_model.predict(X_test)):.3f}, "
          f"ROC-AUC: {roc_auc_score(y_test, xgb_proba):.3f}")

    print("Saving artifacts...")
    joblib.dump(log_reg, f"{OUTPUT_DIR}/logistic_model.pkl")
    joblib.dump(xgb_model, f"{OUTPUT_DIR}/xgboost_model.pkl")
    joblib.dump(scaler, f"{OUTPUT_DIR}/scaler.pkl")
    joblib.dump(list(X_train.columns), f"{OUTPUT_DIR}/feature_columns.pkl")
    joblib.dump(num_cols, f"{OUTPUT_DIR}/numeric_columns.pkl")
    df_encoded.to_csv(f"{OUTPUT_DIR}/processed_leads.csv", index=False)

    print("Done.")


if __name__ == "__main__":
    main()
