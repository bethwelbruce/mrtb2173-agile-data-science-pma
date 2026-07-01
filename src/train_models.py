"""
Sprint 2 & 3 -- Feature Engineering, Baseline Model, and Improved Model.
Q2(a), Q2(b), Q2(c)
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                             roc_auc_score, f1_score)
import joblib, os, json

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA  = os.path.join(BASE_DIR, "data", "telco_churn_clean.csv")
MDIR  = os.path.join(BASE_DIR, "models")
OUT   = os.path.join(BASE_DIR, "outputs")
os.makedirs(MDIR, exist_ok=True)
os.makedirs(OUT, exist_ok=True)

df = pd.read_csv(DATA)

# Fix Churn FIRST before anything else
df["Churn"] = df["Churn"].apply(lambda x: 1 if str(x).strip() == "Yes" else 0)

print("Churn dtype:", df["Churn"].dtype, "| unique:", df["Churn"].unique())

# Q2(a) Feature Engineering
print("=" * 70)
print("Q2(a) FEATURE ENGINEERING")
print("=" * 70)

cat_cols = [c for c in df.select_dtypes(include=["object","string"]).columns if c != "Churn"]
le = LabelEncoder()
for col in cat_cols:
    df[col] = le.fit_transform(df[col].astype(str))
print(f"Step 1 -- Label Encoding: {len(cat_cols)} columns encoded")

df["TenureGroup"] = pd.cut(df["tenure"], bins=[0,12,24,48,72], labels=[0,1,2,3], include_lowest=True).astype(int)
print("Step 2 -- TenureGroup binning created")

TARGET = "Churn"
X = df.drop(columns=[TARGET]).fillna(0)
y = df[TARGET]

scaler = StandardScaler()
num_cols = ["MonthlyCharges","TotalCharges","tenure"]
X[num_cols] = scaler.fit_transform(X[num_cols])
print("Step 3 -- StandardScaler applied to numeric features")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Split: Train {X_train.shape}, Test {X_test.shape}")

# Q2(b) Baseline
print("\n" + "=" * 70)
print("Q2(b) BASELINE -- Logistic Regression")
print("=" * 70)
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
y_prob_lr = lr.predict_proba(X_test)[:,1]
acc_lr = accuracy_score(y_test, y_pred_lr)
auc_lr = roc_auc_score(y_test, y_prob_lr)
f1_lr  = f1_score(y_test, y_pred_lr, pos_label=1)
print(f"Accuracy: {acc_lr:.4f}  ROC-AUC: {auc_lr:.4f}  F1: {f1_lr:.4f}")
print(classification_report(y_test, y_pred_lr, target_names=["No Churn","Churn"]))
joblib.dump(lr, f"{MDIR}/baseline_logistic.pkl")
joblib.dump(scaler, f"{MDIR}/scaler.pkl")
joblib.dump(list(X.columns), f"{MDIR}/feature_names.pkl")

# Q2(c) Improved
print("\n" + "=" * 70)
print("Q2(c) IMPROVED -- Gradient Boosting + GridSearchCV")
print("=" * 70)
param_grid = {"n_estimators":[100,200],"learning_rate":[0.05,0.10],"max_depth":[3,5]}
gb = GradientBoostingClassifier(random_state=42)
grid = GridSearchCV(gb, param_grid, cv=3, scoring="roc_auc", n_jobs=-1)
grid.fit(X_train, y_train)
best_gb = grid.best_estimator_
y_pred_gb = best_gb.predict(X_test)
y_prob_gb = best_gb.predict_proba(X_test)[:,1]
acc_gb = accuracy_score(y_test, y_pred_gb)
auc_gb = roc_auc_score(y_test, y_prob_gb)
f1_gb  = f1_score(y_test, y_pred_gb, pos_label=1)
print(f"Best params: {grid.best_params_}")
print(f"Accuracy: {acc_gb:.4f}  ROC-AUC: {auc_gb:.4f}  F1: {f1_gb:.4f}")
print(classification_report(y_test, y_pred_gb, target_names=["No Churn","Churn"]))
joblib.dump(best_gb, f"{MDIR}/improved_gb.pkl")

# Save metrics
metrics = {
    "baseline": {"accuracy": round(acc_lr,4), "roc_auc": round(auc_lr,4), "f1": round(f1_lr,4)},
    "improved": {"accuracy": round(acc_gb,4), "roc_auc": round(auc_gb,4), "f1": round(f1_gb,4)},
    "best_params": grid.best_params_
}
with open(f"{OUT}/model_metrics.json","w") as f:
    json.dump(metrics, f, indent=2)

# Feature importance
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fi = pd.Series(best_gb.feature_importances_, index=X.columns).sort_values(ascending=False)[:15]
fig, ax = plt.subplots(figsize=(9,5))
fi.plot(kind="barh", ax=ax, color="steelblue")
ax.set_title("Top 15 Feature Importances - Gradient Boosting")
ax.set_xlabel("Importance"); ax.invert_yaxis()
plt.tight_layout(); plt.savefig(f"{OUT}/fig5_feature_importance.png", dpi=150); plt.close()

print("\n[DONE] Models and metrics saved.")
print(f"Baseline: Acc={acc_lr:.4f}, AUC={auc_lr:.4f}")
print(f"Improved: Acc={acc_gb:.4f}, AUC={auc_gb:.4f}")
