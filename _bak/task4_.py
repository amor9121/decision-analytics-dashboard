import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ===== DEBUG ONLY (REMOVE BEFORE SUBMISSION) =====
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
# =================================================

# -----------------------------
# Task 4 – AED Sample Analysis
# -----------------------------


def solve_task4(csv_path="data/AED4weeks.csv", seed=123, n=400):
    """
    Task 4:
    - Draw a random sample of ED patients
    - Produce descriptive statistics
    - Generate key figures to understand workload, patient flow,
      and factors related to breaches or prolonged stays
    """

    # --------------------------------------------------
    # 1. Load data
    # --------------------------------------------------
    aed = pd.read_csv(csv_path)
    aed.columns = aed.columns.str.strip()

    # --------------------------------------------------
    # 2. Random sample (reproducible)
    # --------------------------------------------------
    sample = aed.sample(n=n, random_state=seed).copy()

    # --------------------------------------------------
    # 3. Basic data cleaning (for plotting)
    # --------------------------------------------------
    numeric_cols = [
        "Age",
        "Day",
        "Period",
        "LoS",
        "noofinvestigation",
        "nooftreatment",
        "noofpatients",
    ]

    for col in numeric_cols:
        if col in sample.columns:
            sample[col] = pd.to_numeric(sample[col], errors="coerce")

    # Breach indicator
    breach_str = sample["Breachornot"].astype(str).str.lower()
    sample["is_breach"] = breach_str.str.contains("breach") & ~breach_str.str.contains(
        "non"
    )

    # Clinical complexity proxy
    sample["clinical_complexity"] = sample["noofinvestigation"].fillna(0) + sample[
        "nooftreatment"
    ].fillna(0)

    # --------------------------------------------------
    # 4. Numerical summaries
    # --------------------------------------------------
    age_summary = sample[["Age"]].describe()
    los_summary = sample[["LoS"]].describe()
    breach_counts = sample["Breachornot"].value_counts()
    breach_rate_pct = round(sample["is_breach"].mean() * 100, 2)

    # --------------------------------------------------
    # 5. Figure 1 – Core relationships
    # --------------------------------------------------
    fig1, axes = plt.subplots(2, 2, figsize=(14, 11))

    # (a) LoS distribution
    sns.histplot(sample["LoS"].dropna(), bins=30, kde=True, ax=axes[0, 0])
    axes[0, 0].set_title("(a) Distribution of Length of Stay")
    axes[0, 0].set_xlabel("Length of Stay (minutes)")
    axes[0, 0].set_ylabel("Density")

    # (b) Breach vs LoS
    sns.boxplot(x="Breachornot", y="LoS", data=sample, ax=axes[0, 1])
    axes[0, 1].set_title("(b) Length of Stay by Breach Status")
    axes[0, 1].set_xlabel("Breach Status")
    axes[0, 1].set_ylabel("Length of Stay (minutes)")

    # (c) Congestion vs LoS
    sns.scatterplot(x="noofpatients", y="LoS", data=sample, alpha=0.6, ax=axes[1, 0])
    axes[1, 0].set_title("(c) ED Congestion and Length of Stay")
    axes[1, 0].set_xlabel("Number of Patients in ED on Arrival")
    axes[1, 0].set_ylabel("Length of Stay (minutes)")

    # (d) Period vs LoS
    sns.boxplot(x="Period", y="LoS", data=sample, showfliers=False, ax=axes[1, 1])
    axes[1, 1].set_title("(d) Length of Stay by Arrival Period")
    axes[1, 1].set_xlabel("Arrival Period (0–23)")
    axes[1, 1].set_ylabel("Length of Stay (minutes)")

    fig1.suptitle(
        "Figure 1. Key Features and Relationships in a Random Sample of ED Patients (n = 400)",
        fontsize=14,
    )
    fig1.tight_layout(rect=[0, 0, 1, 0.96])

    # --------------------------------------------------
    # 6. Figure 2 – Additional insights (optional)
    # --------------------------------------------------
    fig2, axes2 = plt.subplots(2, 2, figsize=(14, 11))

    # (a) Age distribution
    sns.histplot(sample["Age"].dropna(), bins=30, ax=axes2[0, 0])
    axes2[0, 0].set_title("(a) Age Distribution")
    axes2[0, 0].set_xlabel("Age (years)")
    axes2[0, 0].set_ylabel("Number of patients")

    # (b) Breach counts
    breach_counts.plot(kind="bar", ax=axes2[0, 1])
    axes2[0, 1].set_title("(b) Breach Status Counts")
    axes2[0, 1].set_xlabel("Breach Status")
    axes2[0, 1].set_ylabel("Number of patients")

    # (c) Breach rate by period
    breach_rate_by_period = (
        sample.groupby("Period")["is_breach"].mean().sort_index() * 100
    )
    axes2[1, 0].plot(
        breach_rate_by_period.index,
        breach_rate_by_period.values,
        marker="o",
    )
    axes2[1, 0].set_title("(c) Breach Rate by Arrival Period")
    axes2[1, 0].set_xlabel("Arrival Period (0–23)")
    axes2[1, 0].set_ylabel("Breach rate (%)")
    axes2[1, 0].set_xticks(range(0, 24, 2))

    # (d) Clinical complexity vs LoS
    sns.scatterplot(
        x="clinical_complexity",
        y="LoS",
        data=sample,
        alpha=0.6,
        ax=axes2[1, 1],
    )
    axes2[1, 1].set_title("(d) Clinical Complexity vs Length of Stay")
    axes2[1, 1].set_xlabel("Investigations + Treatments (count)")
    axes2[1, 1].set_ylabel("Length of Stay (minutes)")

    fig2.suptitle(
        "Figure 2. Additional Descriptive Insights from the ED Sample (n = 400)",
        fontsize=14,
    )
    fig2.tight_layout(rect=[0, 0, 1, 0.96])

    # --------------------------------------------------
    # 7. Short management summary
    # --------------------------------------------------
    summary_text = (
        f"The sample contains {n} patients (random seed = {seed}). "
        f"The breach rate is {breach_rate_pct}%. "
        "Long stays are right-skewed and breaches are associated with extended lengths of stay. "
        "Higher congestion, arrival timing, and greater clinical complexity are linked to longer stays."
    )

    # --------------------------------------------------
    # 8. Return everything (for app or report)
    # --------------------------------------------------
    return {
        "name": "Task 4",
        "sample": sample,
        "age_summary": age_summary,
        "los_summary": los_summary,
        "breach_counts": breach_counts,
        "breach_rate_pct": breach_rate_pct,
        "fig1": fig1,
        "fig2": fig2,
        "summary": summary_text,
    }


# ===== DEBUG ONLY (REMOVE BEFORE SUBMISSION) =====
if __name__ == "__main__":

    t4 = solve_task4()

    print("\n=== Task 4: AED Sample Analysis ===")
    print(t4["summary"])

    print("\nBreach rate (%):")
    print(t4["breach_rate_pct"])

    print("\nAge summary:")
    print(t4["age_summary"])

    print("\nLength of Stay summary:")
    print(t4["los_summary"])

    # Show figures
    plt.show()
# =================================================
