import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import classification_report, average_precision_score, roc_auc_score
import xgboost as xgb
import joblib

def run_pipeline(data_path="creditcard.csv"):
    print("[*] Loading transaction dataset...")
    df = pd.read_csv(data_path)

    # Preprocessing: Scale Amount and Time, preserve PCA components
    scaler = RobustScaler()
    df["scaled_amount"] = scaler.fit_transform(df[["Amount"]])
    df["scaled_time"] = scaler.fit_transform(df[["Time"]])
    df.drop(["Time", "Amount"], axis=1, inplace=True)

    X = df.drop("Class", axis=1)
    y = df["Class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Native gradient scaling
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    print(f"[*] scale_pos_weight calculated: {scale_pos_weight:.2f}")

    # Champion XGBoost Model
    model = xgb.XGBClassifier(
        n_estimators=150,
        max_depth=5,
        learning_rate=0.08,
        scale_pos_weight=scale_pos_weight,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="aucpr",
        n_jobs=-1
    )

    print("[*] Training XGBoost fraud engine...")
    model.fit(X_train, y_train)

    # Cost-tuned probability threshold (tau = 0.28)
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= 0.28).astype(int)

    print("\n=== Test Set Performance (Threshold tau = 0.28) ===")
    print(f"PR-AUC (Average Precision): {average_precision_score(y_test, y_proba):.4f}")
    print(f"ROC-AUC:                  {roc_auc_score(y_test, y_proba):.4f}\n")
    print(classification_report(y_test, y_pred, target_names=["Legitimate", "Fraud"], digits=4))

if __name__ == "__main__":
    run_pipeline()