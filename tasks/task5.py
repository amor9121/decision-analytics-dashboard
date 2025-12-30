import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import stats


def solve_task5(
    csv_path="data/AED4weeks.csv",
    seed=123,
    sample_n=400,
    los_target_min=240,
    prolonged_quantile=0.75,
):
    """
    Task 5 (Polished + 4-in-1 Figure):
    - Tables: 5.1–5.3
    - Figures: fig0 (overview) + fig1 (4-in-1 key driver panel)
    - Headless-safe: no plt.show(); return Figure objects for savefig()
    - No Task 6 (no ML)
    """

    # =========================
    # Global plotting style
    # =========================
    mpl.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 200,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    # =========================
    # 0) Load data & sample
    # =========================
    df = pd.read_csv(csv_path)
    df["ID"] = df["ID"].astype(str).str.strip()

    if "Breachornot" not in df.columns:
        raise KeyError("Expected column 'Breachornot' not found in the CSV.")
    if "LoS" not in df.columns:
        raise KeyError("Expected column 'LoS' not found in the CSV.")
    if "Age" not in df.columns:
        raise KeyError("Expected column 'Age' not found in the CSV.")

    breach_raw = df["Breachornot"].astype(str).str.lower().str.strip()
    df["Breach"] = breach_raw.isin(["breach", "yes", "y", "1", "true"])

    sample = df.sample(n=sample_n, random_state=seed).copy()

    breach_count = int(sample["Breach"].sum())
    breach_rate_pct = breach_count / len(sample) * 100

    # =========================
    # Table 5.1 – Prolonged stay & KPIs
    # =========================
    prolonged_threshold = float(sample["LoS"].quantile(prolonged_quantile))
    sample["Prolonged_Stay"] = sample["LoS"] >= prolonged_threshold
    prolonged_rate_pct = float(sample["Prolonged_Stay"].mean() * 100)

    table_5_1 = pd.DataFrame(
        {
            "Metric": [
                f"Prolonged threshold (LoS ≥ {int(prolonged_quantile*100)}th percentile)",
                "Prolonged stay rate (%)",
                f"4-hour target threshold (minutes)",
                "Breach count",
                "Breach rate (%)",
                "Sample size",
                "Random seed",
            ],
            "Value": [
                round(prolonged_threshold, 2),
                round(prolonged_rate_pct, 2),
                los_target_min,
                breach_count,
                round(breach_rate_pct, 2),
                len(sample),
                seed,
            ],
        }
    )

    # =========================
    # Table 5.2 – Numeric comparisons (Mann–Whitney U)
    # =========================
    def mann_whitney_table(df_, outcome_col, features):
        rows = []
        for f in features:
            if f not in df_.columns:
                continue
            g0 = df_.loc[df_[outcome_col] == False, f].dropna()
            g1 = df_.loc[df_[outcome_col] == True, f].dropna()
            if len(g0) < 5 or len(g1) < 5:
                continue
            stat, p = stats.mannwhitneyu(g0, g1, alternative="two-sided")
            rows.append(
                {
                    "Variable": f,
                    "Median (Non-breach)": round(float(g0.median()), 2),
                    "Median (Breach)": round(float(g1.median()), 2),
                    "n (Non-breach)": int(len(g0)),
                    "n (Breach)": int(len(g1)),
                    "p-value": round(float(p), 4),
                }
            )
        return pd.DataFrame(rows)

    numeric_candidates = [
        "LoS",
        "Age",
        "noofinvestigation",
        "nooftreatment",
        "noofpatients",
    ]
    table_5_2 = mann_whitney_table(sample, "Breach", numeric_candidates)

    # =========================
    # Table 5.3 – Categorical associations (Chi-square)
    # =========================
    def chi_square_table(df_, cat_vars, outcome_col="Breach"):
        rows = []
        for c in cat_vars:
            if c not in df_.columns:
                continue
            ct = pd.crosstab(df_[c], df_[outcome_col])
            if ct.shape[0] < 2 or ct.shape[1] < 2:
                continue
            chi2, p, dof, exp = stats.chi2_contingency(ct)
            rows.append(
                {
                    "Variable": c,
                    "Chi-square": round(float(chi2), 3),
                    "dof": int(dof),
                    "p-value": round(float(p), 4),
                    "Min expected": round(float(np.min(exp)), 2),
                    "Low expected?": bool(np.min(exp) < 5),
                }
            )
        return pd.DataFrame(rows)

    cat_candidates = ["DayofWeek", "Period", "HRG"]
    table_5_3 = chi_square_table(sample, cat_candidates, "Breach")

    # =========================
    # Figures
    # =========================
    figs = {}

    # ---- fig0: Overview (Age + LoS with target & KPI)
    fig0, ax = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)

    ax[0].hist(sample["Age"].dropna(), bins=20, edgecolor="white")
    ax[0].set_title("Patient Age Distribution")
    ax[0].set_xlabel("Age (years)")
    ax[0].set_ylabel("Number of patients")

    ax[1].hist(sample["LoS"].dropna(), bins=25, edgecolor="white")
    ax[1].axvline(
        los_target_min,
        linestyle="--",
        linewidth=1.5,
        label=f"4-hour target ({los_target_min} min)",
    )
    ax[1].set_title("Length of Stay Relative to 4-Hour Target")
    ax[1].set_xlabel("Length of Stay (minutes)")
    ax[1].set_ylabel("Number of patients")
    ax[1].legend(frameon=False)

    fig0.text(
        0.73,
        0.93,
        f"Breach count: {breach_count}\n"
        f"Breach rate: {breach_rate_pct:.2f}% (target < 5%)",
        ha="left",
        va="top",
        fontsize=11,
    )

    figs["fig0"] = fig0

    # ---- fig1: 4-in-1 key driver panel (replaces separate fig1–fig4)
    fig1, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)

    # (A) LoS by Breach
    sample.boxplot(
        column="LoS",
        by="Breach",
        ax=axes[0, 0],
        patch_artist=True,
        boxprops=dict(facecolor="#D9E1F2"),
        medianprops=dict(color="#1F4E79", linewidth=2),
        whiskerprops=dict(linewidth=1.2),
        capprops=dict(linewidth=1.2),
    )
    axes[0, 0].set_title("(A) Length of Stay by Breach Status")
    axes[0, 0].set_xlabel("Breach")
    axes[0, 0].set_ylabel("Length of Stay (minutes)")

    # (B) Breach rate by Period
    if "Period" in sample.columns:
        breach_by_period = (
            sample.groupby("Period")["Breach"].mean() * 100
        ).sort_index()
        breach_by_period.plot(kind="bar", ax=axes[0, 1])
        axes[0, 1].set_title("(B) Breach Rate by Period")
        axes[0, 1].set_xlabel("Period")
        axes[0, 1].set_ylabel("Breach Rate (%)")
        axes[0, 1].tick_params(axis="x", rotation=35)
    else:
        axes[0, 1].text(0.5, 0.5, "Period column not found", ha="center", va="center")
        axes[0, 1].set_axis_off()

    # (C) Breach rate by Day of Week
    if "DayofWeek" in sample.columns:
        breach_by_dow = (
            sample.groupby("DayofWeek")["Breach"].mean() * 100
        ).sort_values(ascending=False)
        breach_by_dow.plot(kind="bar", ax=axes[1, 0])
        axes[1, 0].set_title("(C) Breach Rate by Day of Week")
        axes[1, 0].set_xlabel("Day of Week")
        axes[1, 0].set_ylabel("Breach Rate (%)")
        axes[1, 0].tick_params(axis="x", rotation=35)
    else:
        axes[1, 0].text(
            0.5, 0.5, "DayofWeek column not found", ha="center", va="center"
        )
        axes[1, 0].set_axis_off()

    # (D) Investigations by Breach
    if "noofinvestigation" in sample.columns:
        sample.boxplot(
            column="noofinvestigation",
            by="Breach",
            ax=axes[1, 1],
            patch_artist=True,
            boxprops=dict(facecolor="#E2EFDA"),
            medianprops=dict(color="#1F4E79", linewidth=2),
            whiskerprops=dict(linewidth=1.2),
            capprops=dict(linewidth=1.2),
        )
        axes[1, 1].set_title("(D) Number of Investigations by Breach Status")
        axes[1, 1].set_xlabel("Breach")
        axes[1, 1].set_ylabel("Number of investigations")
    else:
        axes[1, 1].text(
            0.5, 0.5, "noofinvestigation column not found", ha="center", va="center"
        )
        axes[1, 1].set_axis_off()

    # Remove pandas boxplot auto suptitle, then set our own
    fig1.suptitle(
        "Figure 5.1: Key Factors Associated with 4-Hour Target Breaches in the AED",
        fontsize=14,
    )

    figs["fig1"] = fig1

    return {
        "sample": sample,
        "tables": {
            "table_5_1": table_5_1,
            "table_5_2": table_5_2,
            "table_5_3": table_5_3,
        },
        "stats": {
            "breach_count": breach_count,
            "breach_rate_pct": round(breach_rate_pct, 2),
            "prolonged_threshold": round(prolonged_threshold, 2),
            "prolonged_rate_pct": round(prolonged_rate_pct, 2),
            "seed": seed,
            "sample_n": sample_n,
            "los_target_min": los_target_min,
        },
        "figures": figs,
    }


# ===== DEBUG ONLY (REMOVE BEFORE SUBMISSION) =====
if __name__ == "__main__":
    t5 = solve_task5()

    print("\n=== Task 5: Statistical Analysis ===")
    print("\nTable 5.1 – Prolonged stay & KPIs\n", t5["tables"]["table_5_1"])
    print("\nTable 5.2 – Numeric variables vs Breach\n", t5["tables"]["table_5_2"])
    print("\nTable 5.3 – Categorical variables vs Breach\n", t5["tables"]["table_5_3"])
    print("\nBreach rate (%):", t5["stats"]["breach_rate_pct"])

    # Save the 2 figures (overview + 4-in-1)
    t5["figures"]["fig0"].savefig(
        "task5_fig0_overview.png", dpi=200, bbox_inches="tight"
    )
    t5["figures"]["fig1"].savefig(
        "task5_fig1_key_breach_drivers.png", dpi=200, bbox_inches="tight"
    )

    print("\nSaved: task5_fig0_overview.png, task5_fig1_key_breach_drivers.png")
