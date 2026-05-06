import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
import pickle
import os

# ── Load & prepare data ──────────────────────────────────────────
df = pd.read_csv("data/creditcard.csv")

fraud = df[df["Class"] == 1]
legit = df[df["Class"] == 0].sample(n=len(fraud), random_state=42)
df_balanced = pd.concat([fraud, legit]).sample(frac=1, random_state=42)

X = df_balanced.drop("Class", axis=1)
y = df_balanced["Class"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ── Define models ────────────────────────────────────────────────
models = {
    "DecisionTree":       DecisionTreeClassifier(random_state=42),
    "RandomForest":       RandomForestClassifier(n_estimators=100, random_state=42),
    "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
    "NaiveBayes":         GaussianNB(),
    "KNN":                KNeighborsClassifier(),
    "XGBoost":            XGBClassifier(eval_metric="logloss", random_state=42),
    "AdaBoost":           AdaBoostClassifier(random_state=42),
}

# ── Set tracking ─────────────────────────────────────────────────
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("fraud-detection")

best_f1 = 0
best_model = None
best_model_name = ""

for name, model in models.items():
    with mlflow.start_run(run_name=name):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        f1        = f1_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall    = recall_score(y_test, y_pred)
        roc_auc   = roc_auc_score(y_test, y_pred)

        mlflow.log_param("model", name)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("roc_auc", roc_auc)
        mlflow.sklearn.log_model(model, name=name)

        print(f"{name:20s} | F1: {f1:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            best_model = model
            best_model_name = name

# ── Save best model ──────────────────────────────────────────────
os.makedirs("models", exist_ok=True)
with open("models/best_model.pkl", "wb") as f:
    pickle.dump(best_model, f)

print(f"\nBest model: {best_model_name} with F1: {best_f1:.4f}")
print("Saved to models/best_model.pkl")