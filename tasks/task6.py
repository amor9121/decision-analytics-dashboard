import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_score,
    learning_curve,
)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFECV

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    accuracy_score,
)

# ------------------------------------------------------------
# Plotting style (kept simple & academic)
# ------------------------------------------------------------
sns.set_style("whitegrid")

# ------------------------------------------------------------
# Data utilities (kept local to avoid extra dependencies)
# ------------------------------------------------------------


def get_data(filepath: str):
    df = pd.read_csv(filepath)
    return df


def clean_target_variable(x):
    """
    Robust cleaning: convert target variable to binary (0/1).
    """
    s = str(x).lower().strip()
    if "non" in s or "not" in s or "on time" in s or s in {"0", "0.0"}:
        return 0
    if "breach" in s or s in {"1", "1.0"}:
        return 1
    return 0


# ------------------------------------------------------------
# Task 6 main solver
# ------------------------------------------------------------
def solve_task6(filepath: str = "data/AED4weeks.csv"):
    """
    Final classification model for Task 6:
    - Logistic Regression
    - RFECV feature selection
    - Robust validation & diagnostics
    """

    # === Figure collector (for Streamlit / dashboard integration) ===
    figures = []

    def _collect_fig():
        fig = plt.gcf()
        figures.append(fig)
        plt.close(fig)

    # ------------------------------------------------------------
    # 1. Data preparation
    # ------------------------------------------------------------
    df = get_data(filepath)

    # Clean target
    df["Breachornot"] = df["Breachornot"].apply(clean_target_variable)

    potential_features = [
        "Age",
        "Period",
        "DayofWeek",
        "noofinvestigations",
        "noofinvestigation",
        "nooftreatments",
        "nooftreatment",
        "noofpatients",
        "noofpatient",
    ]

    feature_cols = [c for c in potential_features if c in df.columns]

    # Defensive check against leakage
    if "LoS" in feature_cols:
        feature_cols.remove("LoS")
        print("[Info] 'LoS' removed to prevent data leakage.")

    X = df[feature_cols]
    y = df["Breachornot"]

    # One-hot encoding (safe for categorical inputs)
    X = pd.get_dummies(X, drop_first=True)
    feature_names = X.columns.tolist()

    # Train / test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=123, stratify=y
    )

    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    X_train_df = pd.DataFrame(X_train_scaled, columns=feature_names)
    X_test_df = pd.DataFrame(X_test_scaled, columns=feature_names)

    # ------------------------------------------------------------
    # 2. Feature selection (RFECV)
    # ------------------------------------------------------------
    base_model = LogisticRegression(
        class_weight="balanced", solver="liblinear", random_state=123
    )

    rfecv = RFECV(
        estimator=base_model,
        step=1,
        cv=StratifiedKFold(5),
        scoring="roc_auc",
        min_features_to_select=1,
    )

    rfecv.fit(X_train_df, y_train)

    selected_features = X_train_df.columns[rfecv.support_].tolist()

    # --- Plot 1: RFECV performance ---
    plt.figure(figsize=(8, 6))
    plt.plot(
        range(1, len(rfecv.cv_results_["mean_test_score"]) + 1),
        rfecv.cv_results_["mean_test_score"],
        marker="o",
        color="purple",
    )
    plt.title("RFECV Feature Selection Process")
    plt.xlabel("Number of Features Selected")
    plt.ylabel("CV Score (ROC-AUC)")
    plt.grid(True)
    plt.tight_layout()
    _collect_fig()

    # ------------------------------------------------------------
    # 3. Final model training
    # ------------------------------------------------------------
    X_train_final = X_train_df[selected_features]
    X_test_final = X_test_df[selected_features]

    final_model = LogisticRegression(
        class_weight="balanced", solver="liblinear", random_state=123
    )
    final_model.fit(X_train_final, y_train)

    y_pred = final_model.predict(X_test_final)
    y_prob = final_model.predict_proba(X_test_final)[:, 1]

    # ------------------------------------------------------------
    # 4. Performance metrics
    # ------------------------------------------------------------
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report_txt = classification_report(
        y_test, y_pred, target_names=["On Time", "Breach"]
    )

    # --- Plot 2: Confusion matrix ---
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Pred: On Time", "Pred: Breach"],
        yticklabels=["Actual: On Time", "Actual: Breach"],
    )
    plt.title("Confusion Matrix (Balanced Logistic Regression)")
    plt.tight_layout()
    _collect_fig()

    # --- Plot 3: ROC curve ---
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc_val = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"AUC = {roc_auc_val:.2f}")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.title("ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.tight_layout()
    _collect_fig()

    # --- Plot 4: Odds ratios ---
    odds_df = pd.DataFrame(
        {
            "Feature": selected_features,
            "Odds_Ratio": np.exp(final_model.coef_[0]),
        }
    ).sort_values("Odds_Ratio", ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(
        x="Odds_Ratio",
        y="Feature",
        data=odds_df,
        hue="Feature",
        legend=False,
        palette="viridis",
    )
    plt.axvline(x=1, color="red", linestyle="--", label="Neutral (OR=1)")
    plt.title("Odds Ratio Analysis (Risk Factors)")
    plt.xlabel("Odds Ratio (Values > 1 indicate higher risk)")
    plt.legend()
    plt.tight_layout()
    _collect_fig()

    # ------------------------------------------------------------
    # 5. Robustness checks
    # ------------------------------------------------------------
    cv = StratifiedKFold(5, shuffle=True, random_state=123)
    cv_recall = cross_val_score(
        final_model, X_train_final, y_train, cv=cv, scoring="recall"
    )
    cv_auc = cross_val_score(
        final_model, X_train_final, y_train, cv=cv, scoring="roc_auc"
    )

    # --- Plot 5: CV stability ---
    plt.figure(figsize=(6, 6))
    plt.boxplot(
        [cv_recall, cv_auc],
        labels=["Recall", "ROC-AUC"],
        patch_artist=True,
        boxprops=dict(facecolor="lightblue"),
    )
    plt.title("5-Fold Cross-Validation Stability")
    plt.ylabel("Score")
    plt.ylim(0, 1.05)
    plt.grid(True, axis="y")
    plt.tight_layout()
    _collect_fig()

    # --- Plot 6: Learning curve ---
    train_sizes, train_scores, test_scores = learning_curve(
        final_model,
        X_train_final,
        y_train,
        cv=5,
        scoring="recall",
        train_sizes=np.linspace(0.1, 1.0, 5),
    )

    plt.figure(figsize=(8, 6))
    plt.plot(
        train_sizes, train_scores.mean(axis=1), "o-", color="r", label="Training Recall"
    )
    plt.plot(
        test_scores.mean(axis=1),
    )
    plt.plot(
        train_sizes,
        test_scores.mean(axis=1),
        "o-",
        color="g",
        label="Validation Recall",
    )

    train_std = train_scores.std(axis=1)
    test_std = test_scores.std(axis=1)

    plt.fill_between(
        train_sizes,
        train_scores.mean(axis=1) - train_std,
        train_scores.mean(axis=1) + train_std,
        alpha=0.1,
        color="r",
    )
    plt.fill_between(
        train_sizes,
        test_scores.mean(axis=1) - test_std,
        test_scores.mean(axis=1) + test_std,
        alpha=0.1,
        color="g",
    )

    plt.title("Learning Curve (Check for Overfitting)")
    plt.xlabel("Training Examples")
    plt.ylabel("Recall Score")
    plt.legend(loc="best")
    plt.grid(True)
    plt.tight_layout()
    _collect_fig()

    # ------------------------------------------------------------
    # Return bundle (system-friendly)
    # ------------------------------------------------------------
    return {
        "name": "Task 6",
        "summary": "Logistic Regression with RFECV and balanced class weights.",
        "n": len(df),
        "accuracy": float(acc),
        "roc_auc": float(roc_auc_val),
        "classification_report": report_txt,
        "confusion_matrix": cm,
        "selected_features": selected_features,
        "odds_ratio_table": odds_df,
        "cv_recall": cv_recall,
        "cv_auc": cv_auc,
        "figures": {f"fig{i}": fig for i, fig in enumerate(figures)},
        "tables": {
            "odds_ratio": odds_df.reset_index(drop=True),
            "confusion_matrix": pd.DataFrame(
                cm,
                index=["Actual_OnTime", "Actual_Breach"],
                columns=["Pred_OnTime", "Pred_Breach"],
            ),
            "cv_recall": pd.DataFrame({"recall": cv_recall}),
            "cv_auc": pd.DataFrame({"roc_auc": cv_auc}),
            "selected_features": pd.DataFrame({"feature": selected_features}),
        },
    }
