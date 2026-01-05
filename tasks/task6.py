import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    RocCurveDisplay,
    precision_recall_curve,
    auc,
    ConfusionMatrixDisplay,
)


def solve_task6(aed: pd.DataFrame, test_size=0.2, random_state=123, threshold=0.5):
    """
    Task 6: Apply supervised machine learning to predict breach vs non-breach
    """

    df = aed.copy()

    # Define target variable (breach vs non-breach)
    if "Breach" in df.columns:
        y = df["Breach"].astype(int)
        X = df.drop(columns=["Breach"])
    else:
        y = (
            df["Breachornot"]
            .str.lower()
            .map({"breach": 1, "non-breach": 0})
            .astype(int)
        )
        X = df.drop(columns=["Breachornot"])

    # --- Drop leakage & non-deployable features
    drop_cols = [
        "ID",
        "LoS",
        "HRG",
        "noofinvestigation",
        "nooftreatment",
        "Day",
    ]

    X = X.drop(columns=[c for c in drop_cols if c in X.columns], errors="ignore")

    # 1) Train–test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # 2) Preprocessing (scale numeric + one-hot categorical)
    cat_cols = X_train.select_dtypes(include=["object"]).columns.tolist()
    num_cols = [c for c in X_train.columns if c not in cat_cols]

    # Scaled (good for LogReg / often OK for RF)
    preprocessor_scaled = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ]
    )

    # No-scale (better interpretability for DecisionTree)
    preprocessor_noscale = ColumnTransformer(
        transformers=[
            ("num", "passthrough", num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ]
    )

    # 3) Train machine learning models
    models = {
        "LogReg": LogisticRegression(max_iter=3000, class_weight="balanced"),
        "DecisionTree": DecisionTreeClassifier(
            max_depth=5, random_state=random_state, class_weight="balanced"
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=300,
            random_state=random_state,
            class_weight="balanced",
            n_jobs=-1,
        ),
    }

    rows = []
    fitted = {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    forest_importance = None
    forest_feature_names = None

    for name, clf in models.items():
        # Use no-scale preprocessing only for DecisionTree (readable thresholds)
        prep = preprocessor_noscale if name == "DecisionTree" else preprocessor_scaled

        pipe = Pipeline([("prep", prep), ("model", clf)])
        pipe.fit(X_train, y_train)
        fitted[name] = pipe

        if name == "RandomForest":
            model = pipe.named_steps["model"]
            prep = pipe.named_steps["prep"]

            forest_importance = model.feature_importances_
            forest_feature_names = prep.get_feature_names_out()

        # Add cv
        cv_score = cross_val_score(
            pipe, X_train, y_train, cv=cv, scoring="balanced_accuracy", n_jobs=-1
        ).mean()

        # 4) Score / evaluate models
        train_acc = pipe.score(X_train, y_train)
        test_acc = pipe.score(X_test, y_test)

        y_prob = pipe.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= threshold).astype(int)

        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

        pr_p, pr_r, _ = precision_recall_curve(y_test, y_prob)
        pr_auc = auc(pr_r, pr_p)

        rows.append(
            {
                "model": name,
                "train_accuracy": train_acc,
                "test_accuracy": test_acc,
                "cv_accuracy": cv_score,
                "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
                "precision_breach": precision_score(y_test, y_pred, zero_division=0),
                "recall_breach": recall_score(y_test, y_pred, zero_division=0),
                "f1_breach": f1_score(y_test, y_pred, zero_division=0),
                "roc_auc": roc_auc_score(y_test, y_prob),
                "pr_auc": pr_auc,
                "tn": tn,
                "fp": fp,
                "fn": fn,
                "tp": tp,
            }
        )

    summary = (
        pd.DataFrame(rows).set_index("model").sort_values("pr_auc", ascending=False)
    )

    best_model = summary.index[0]
    best_pipe = fitted[best_model]

    # --- Decision Tree visualisation (readable)
    if best_model == "DecisionTree":
        tree_model = best_pipe.named_steps["model"]
        prep_used = best_pipe.named_steps["prep"]

        feature_names = prep_used.get_feature_names_out()
        # shorten feature names for readability
        feature_names = [
            f.replace("num__", "").replace("cat__", "") for f in feature_names
        ]

        fig_tree, ax_tree = plt.subplots(figsize=(20, 10))
        plot_tree(
            tree_model,
            feature_names=feature_names,
            class_names=["Non-breach", "Breach"],
            filled=True,
            rounded=True,
            max_depth=2,
            fontsize=10,
            ax=ax_tree,
            impurity=False,
        )
        ax_tree.set_title("Decision Tree (Top Levels)")
        fig_tree.tight_layout()
    else:
        fig_tree = None

    # Visual evaluation (confusion matrix, ROC curves, PR curves)
    y_prob_best = best_pipe.predict_proba(X_test)[:, 1]
    y_pred_best = (y_prob_best >= threshold).astype(int)
    cm_best = confusion_matrix(y_test, y_pred_best)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    ConfusionMatrixDisplay(cm_best, display_labels=["Non-breach", "Breach"]).plot(
        ax=axes[0, 0], values_format="d", colorbar=False
    )
    axes[0, 0].set_title(f"(A) Confusion Matrix: {best_model}")

    ax = axes[0, 1]
    for name, pipe in fitted.items():
        RocCurveDisplay.from_estimator(pipe, X_test, y_test, name=name, ax=ax)
    ax.set_title("(B) ROC Curves (All Models)")

    ax = axes[1, 0]
    for name, pipe in fitted.items():
        y_prob = pipe.predict_proba(X_test)[:, 1]
        pr_p, pr_r, _ = precision_recall_curve(y_test, y_prob)
        ax.plot(pr_r, pr_p, label=name)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("(C) Precision–Recall Curves (All Models)")
    ax.legend()

    axes[1, 1].axis("off")
    fig.tight_layout()

    return {
        "name": "Task 6",
        "summary": summary,
        "forest_importance": forest_importance,
        "forest_feature_names": forest_feature_names,
        "best_model_name": best_model,
        "plots": {
            "combined": fig,
            "decision_tree": fig_tree,
        },
    }


if __name__ == "__main__":
    aed = pd.read_csv("data/AED4weeks.csv")
    out = solve_task6(aed)

    print("\n=== Task 6: Model comparison table ===")
    print(out["summary"].round(3))
    imp = pd.Series(
        out["forest_importance"], index=out["forest_feature_names"]
    ).sort_values(ascending=False)

    print(imp.head(5))

    plt.show()
