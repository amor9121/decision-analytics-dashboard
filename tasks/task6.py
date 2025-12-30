# Task 6 — Predict 4-hour breach (classification)
# Creates: (1) Confusion Matrix Heatmap, (2) ROC Curve, (3) Logistic Coefficient Bar Chart

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    roc_auc_score,
)

try:
    import seaborn as sns
except ModuleNotFoundError:
    sns = None

# ===== DEBUG ONLY (REMOVE BEFORE SUBMISSION) =====
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
# =================================================

# ===== 1) Load data =====
df = pd.read_csv("data/AED4weeks.csv")  # <- change path if needed

# ===== 2) Define target and features =====
# Target: Breach (True/False). If your file has only 'Breachornot', convert it.
if "Breach" not in df.columns and "Breachornot" in df.columns:
    df["Breach"] = df["Breachornot"].str.lower().eq("breach")

y = df["Breach"].astype(int)

# IMPORTANT (avoid data leakage): Drop LoS (length of stay) because it determines breach after the fact.
drop_cols = [c for c in ["ID", "Breachornot", "Breach", "LoS"] if c in df.columns]
X = df.drop(columns=drop_cols)

# ===== 3) Split =====
SEED = 20251229
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y
)

# ===== 4) Preprocess =====
# Separate numeric vs categorical automatically (safe default).
num_cols = X.select_dtypes(
    include=["int64", "float64", "int32", "float32"]
).columns.tolist()
cat_cols = [c for c in X.columns if c not in num_cols]

preprocess = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ],
    remainder="drop",
)

# ===== 5) Model =====
# class_weight="balanced" helps if breaches are rarer than non-breaches.
model = LogisticRegression(max_iter=2000, class_weight="balanced")

clf = Pipeline(steps=[("prep", preprocess), ("model", model)])

clf.fit(X_train, y_train)

# ===== 6) Predict + metrics =====
y_pred = clf.predict(X_test)
y_prob = clf.predict_proba(X_test)[:, 1]  # probability of breach

print("\n=== Classification report ===")
print(classification_report(y_test, y_pred, target_names=["Non-breach", "Breach"]))

auc = roc_auc_score(y_test, y_prob)
print(f"ROC-AUC: {auc:.3f}")

# ===== 7) Plot 1: Confusion Matrix Heatmap =====
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(5, 4))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Non-breach", "Breach"],
    yticklabels=["Non-breach", "Breach"],
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix (Logistic Regression)")
plt.tight_layout()
plt.show()

# ===== 8) Plot 2: ROC Curve =====
fpr, tpr, thresholds = roc_curve(y_test, y_prob)

plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, label=f"Logistic Regression (AUC={auc:.3f})")
plt.plot([0, 1], [0, 1], linestyle="--", label="Random")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.tight_layout()
plt.show()

# ===== 9) Plot 3: Coefficient bar chart (top factors) =====
# Get feature names after preprocessing
prep = clf.named_steps["prep"]
ohe = prep.named_transformers_["cat"]

# Numeric feature names stay the same; categorical expand via one-hot
cat_feature_names = []
if len(cat_cols) > 0:
    cat_feature_names = ohe.get_feature_names_out(cat_cols).tolist()

feature_names = num_cols + cat_feature_names

coefs = clf.named_steps["model"].coef_.ravel()
coef_df = pd.DataFrame({"feature": feature_names, "coef": coefs})
coef_df["abs_coef"] = coef_df["coef"].abs()

top = coef_df.sort_values("abs_coef", ascending=False).head(15).sort_values("coef")

plt.figure(figsize=(8, 5))
plt.barh(top["feature"], top["coef"])
plt.axvline(0, linewidth=1)
plt.title("Top 15 Logistic Regression Coefficients (Breach = 1)")
plt.xlabel("Coefficient (positive = higher breach risk)")
plt.tight_layout()
plt.show()
