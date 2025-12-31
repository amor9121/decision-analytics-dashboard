import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# -----------------------------
# Task 4 – AED Sample Analysis
# -----------------------------
def solve_task4(csv_path="data/AED4weeks.csv", seed=123, n=400):

    aed = pd.read_csv(csv_path)
    aed.columns = aed.columns.str.strip()

    sample = aed.sample(n=n, random_state=seed).copy()

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

    breach_str = sample["Breachornot"].astype(str).str.lower()
    sample["is_breach"] = breach_str.str.contains("breach") & ~breach_str.str.contains(
        "non"
    )

    sample["clinical_complexity"] = sample["noofinvestigation"].fillna(0) + sample[
        "nooftreatment"
    ].fillna(0)

    age_summary = sample[["Age"]].describe()
    los_summary = sample[["LoS"]].describe()
    breach_counts = sample["Breachornot"].value_counts(dropna=False)
    breach_rate_pct = round(sample["is_breach"].mean() * 100, 2)

    # KPI table (optional but nice)
    kpi_table = (
        sample.groupby("is_breach")
        .agg(
            n=("ID", "count") if "ID" in sample.columns else ("LoS", "count"),
            median_LoS=("LoS", "median"),
            mean_LoS=("LoS", "mean"),
            median_patients_on_arrival=("noofpatients", "median"),
            mean_investigations=("noofinvestigation", "mean"),
            mean_treatments=("nooftreatment", "mean"),
            mean_clinical_complexity=("clinical_complexity", "mean"),
        )
        .rename(index={False: "Non-breach", True: "Breach"})
    )

    # Figure 1
    fig1, axes = plt.subplots(2, 2, figsize=(14, 11))

    sns.histplot(sample["LoS"].dropna(), bins=30, kde=True, ax=axes[0, 0])
    axes[0, 0].set_title("(a) Distribution of Length of Stay")
    axes[0, 0].set_xlabel("Length of Stay (minutes)")
    axes[0, 0].set_ylabel("Density")

    sns.boxplot(x="Breachornot", y="LoS", data=sample, ax=axes[0, 1])
    axes[0, 1].set_title("(b) Length of Stay by Breach Status")
    axes[0, 1].set_xlabel("Breach Status")
    axes[0, 1].set_ylabel("Length of Stay (minutes)")

    sns.scatterplot(x="noofpatients", y="LoS", data=sample, alpha=0.6, ax=axes[1, 0])
    axes[1, 0].set_title("(c) ED Congestion and Length of Stay")
    axes[1, 0].set_xlabel("Number of Patients in ED on Arrival")
    axes[1, 0].set_ylabel("Length of Stay (minutes)")

    sns.boxplot(x="Period", y="LoS", data=sample, showfliers=False, ax=axes[1, 1])
    axes[1, 1].set_title("(d) Length of Stay by Arrival Period")
    axes[1, 1].set_xlabel("Arrival Period (0–23)")
    axes[1, 1].set_ylabel("Length of Stay (minutes)")

    fig1.suptitle(
        f"Figure 1. Key Features and Relationships in a Random Sample (n = {n})",
        fontsize=14,
    )
    fig1.tight_layout(rect=[0, 0, 1, 0.96])

    # Figure 2
    fig2, axes2 = plt.subplots(2, 2, figsize=(14, 11))

    sns.histplot(sample["Age"].dropna(), bins=30, ax=axes2[0, 0])
    axes2[0, 0].set_title("(a) Age Distribution")
    axes2[0, 0].set_xlabel("Age (years)")
    axes2[0, 0].set_ylabel("Number of patients")

    breach_counts.plot(kind="bar", ax=axes2[0, 1])
    axes2[0, 1].set_title("(b) Breach Status Counts")
    axes2[0, 1].set_xlabel("Breach Status")
    axes2[0, 1].set_ylabel("Number of patients")

    breach_rate_by_period = (
        sample.groupby("Period")["is_breach"].mean().sort_index() * 100
    )
    axes2[1, 0].plot(
        breach_rate_by_period.index, breach_rate_by_period.values, marker="o"
    )
    axes2[1, 0].set_title("(c) Breach Rate by Arrival Period")
    axes2[1, 0].set_xlabel("Arrival Period (0–23)")
    axes2[1, 0].set_ylabel("Breach rate (%)")
    axes2[1, 0].set_xticks(range(0, 24, 2))

    # NEW: breach rate by DayofWeek (if present)
    if "DayofWeek" in sample.columns:
        breach_by_dow = (
            sample.groupby("DayofWeek")["is_breach"].mean().sort_values(ascending=False)
            * 100
        )
        breach_by_dow.plot(kind="bar", ax=axes2[1, 1])
        axes2[1, 1].set_title("(d) Breach Rate by Day of Week")
        axes2[1, 1].set_xlabel("Day of Week")
        axes2[1, 1].set_ylabel("Breach rate (%)")
    else:
        sns.scatterplot(
            x="clinical_complexity", y="LoS", data=sample, alpha=0.6, ax=axes2[1, 1]
        )
        axes2[1, 1].set_title("(d) Clinical Complexity vs Length of Stay")
        axes2[1, 1].set_xlabel("Investigations + Treatments (count)")
        axes2[1, 1].set_ylabel("Length of Stay (minutes)")

    fig2.suptitle(
        f"Figure 2. Additional Descriptive Insights (n = {n})",
        fontsize=14,
    )
    fig2.tight_layout(rect=[0, 0, 1, 0.96])

    summary_text = (
        f"The sample contains {n} patients (random seed = {seed}). "
        f"The breach rate is {breach_rate_pct}%. "
        "Length of stay is right-skewed; breaches are associated with longer stays. "
        "Congestion and arrival timing show visible differences in LoS and breach patterns."
    )

    return {
        "name": "Task 4",
        "case": "AED descriptive analysis (random sample n=400)",
        "allocation": None,
        "sample": sample,
        "age_summary": age_summary,
        "los_summary": los_summary,
        "breach_counts": breach_counts,
        "breach_rate_pct": breach_rate_pct,
        "kpi_table": kpi_table,
        "fig1": fig1,
        "fig2": fig2,
        "summary": summary_text,
    }


# ===== DEBUG ONLY (REMOVE BEFORE SUBMISSION) =====
if __name__ == "__main__":
    t4 = solve_task4()

    print("\n=== Task 4: AED Sample Analysis ===")
    print(t4["summary"])
    print("\nBreach rate (%):", t4["breach_rate_pct"])
    print("\nAge summary:\n", t4["age_summary"])
    print("\nLength of Stay summary:\n", t4["los_summary"])

    # Save figures instead of plt.show() (works in any environment)
    t4["fig1"].savefig("outputs/task4_fig1.png", dpi=200, bbox_inches="tight")
    t4["fig2"].savefig("outputs/task4_fig2.png", dpi=200, bbox_inches="tight")
    print("\nSaved: task4_fig1.png, task4_fig2.png")
