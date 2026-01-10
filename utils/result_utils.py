import pandas as pd
from core.data import days
from utils.schedule_utils import check_daily_coverage


def build_schedule_summary(results: list[dict]) -> pd.DataFrame:
    rows = []
    for r in results:
        allocation = r.get("allocation")

        # Only scheduling tasks have allocation / coverage
        if allocation is None:
            continue

        coverage_ok = check_daily_coverage(allocation, days)["ok"]
        coverage_cell = "✅" if coverage_ok else "❌"

        rows.append(
            {
                "Task": r.get("name", ""),
                "Total cost (£)": r.get("cost", ""),
                "Cost increase (%)": r.get("cost_increase_pct", ""),
                "Fairness gap": r.get("gap", ""),
                "Coverage (14h/day)": coverage_cell,
                "Skill coverage": (
                    "✅" if r.get("skill_coverage") is not None else "–"
                ),
            }
        )

    return pd.DataFrame(rows)


def build_aed_summary(results: list[dict]) -> pd.DataFrame:
    rows = []

    for r in results:
        name = r.get("name", "")
        if name not in ["Task 4", "Task 5", "Task 6"]:
            continue

        # --- common fields ---
        row = {
            "Task": name,
            "Sample size (n)": r.get("n", "–"),
        }

        # --- task-specific fields ---
        if name == "Task 4":
            row.update(
                {
                    "Analytical role": "Descriptive",
                    "Objective": "Describe AED workload & breach profile",
                    "Decision timing": "Post hoc",
                    "Feature scope": "Full record",
                    "Method": "Descriptive statistics & visualisation",
                    "Final model selected": "–",
                    "Key insight": "Summarises workload, patient-flow patterns, and breach profile.",
                    "Implication for next task": "Supports selection of variables for inferential testing.",
                }
            )

        elif name == "Task 5":
            row.update(
                {
                    "Analytical role": "Inferential",
                    "Objective": "Identify factors associated with breach",
                    "Decision timing": "Post hoc",
                    "Feature scope": "Full record",
                    "Method": "Non-parametric tests + chi-square (and logit if available)",
                    "Final model selected": "–",
                    "Key insight": "Tests associations between breach and numeric/categorical predictors.",
                    "Implication for next task": "Informs which predictors may be useful for prediction.",
                }
            )

        elif name == "Task 6":
            row.update(
                {
                    "Analytical role": "Predictive",
                    "Objective": "Predict breach at triage time",
                    "Decision timing": "At triage",
                    "Feature scope": "Arrival-time only",
                    "Method": "ML classification + stratified cross-validation",
                    "Final model selected": "Logistic Regression",
                    "Key insight": "Builds a triage-time classifier with validated performance.",
                    "Implication for next task": "Produces a deployable risk score and interpretable drivers.",
                }
            )

        rows.append(row)

    df = pd.DataFrame(rows)

    # --- column order for Summary section ---
    col_order = [
        "Task",
        "Sample size (n)",
        "Analytical role",
        "Objective",
        "Decision timing",
        "Feature scope",
        "Method",
        "Final model selected",
        "Key insight",
        "Implication for next task",
    ]

    return df[[c for c in col_order if c in df.columns]]
