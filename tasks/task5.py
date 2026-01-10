# tasks/task5.py
from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu, chi2_contingency

sns.set_style("whitegrid")

# -----------------------------
# statsmodels
# -----------------------------
try:
    import statsmodels.formula.api as smf
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    _HAS_SM = True
except Exception:
    smf = None
    variance_inflation_factor = None
    _HAS_SM = False


# -----------------------------
# Helpers
# -----------------------------
def _new_fig(figsize: Tuple[float, float] = (5.5, 3.5)):
    fig, ax = plt.subplots(figsize=figsize)
    return fig, ax


def _breach_int(series: pd.Series) -> pd.Series:
    """
    Convert Breachornot labels to binary:
    breach -> 1, non-breach -> 0
    """
    s = series.astype(str).str.strip().str.lower()
    out = s.map({"breach": 1, "non-breach": 0})
    return pd.to_numeric(out, errors="coerce")


def _time_block(h) -> str:
    try:
        h = int(h)
    except Exception:
        return "Unknown"
    if 0 <= h < 6:
        return "Night"
    if 6 <= h < 12:
        return "Morning"
    if 12 <= h < 18:
        return "Afternoon"
    return "Evening"


def _coef_table(model) -> pd.DataFrame:
    coef = model.params
    se = model.bse

    z = 1.96
    lo = coef.values - z * se.values
    hi = coef.values + z * se.values

    # avoid overflow in exp(); exp(709) ~ 8e307 is near float max
    lo = np.clip(lo, -709, 709)
    hi = np.clip(hi, -709, 709)
    c = np.clip(coef.values, -709, 709)

    out = pd.DataFrame(
        {
            "term": coef.index,
            "coef": coef.values,
            "OR": np.exp(c),
            "CI_low": np.exp(lo),
            "CI_high": np.exp(hi),
            "p_value": model.pvalues.values,
        }
    )
    out = out.replace([np.inf, -np.inf], np.nan)
    return out.round(4)


# -----------------------------
# Main
# -----------------------------
def solve_task5(
    filepath: str = "data/AED4weeks.csv",
    *,
    seed: int = 123,
    n_sample: int = 400,
    los_target_min: int = 240,
) -> Dict[str, Any]:
    """
    Task 5: Statistical analysis (dashboard-ready).

    Returns:
      - summary, stats
      - tables: includes Table 5.1/5.2/5.3 + describe + corr + VIF + logit coef tables
      - figures: corr heatmap + LoS by breach + investigations by breach + breach rate by day
      - logit_m1_summary / logit_m2_summary: text (or failure reason)
      - sample, download_df
    """
    df = pd.read_csv(filepath)
    sample = df.sample(n=n_sample, random_state=seed) if len(df) >= n_sample else df.copy()

    # ---- Required columns ----
    for c in ["Breachornot", "LoS"]:
        if c not in sample.columns:
            raise KeyError(f"Task 5 requires column '{c}'.")

    # ---- Target ----
    sample["breach"] = _breach_int(sample["Breachornot"])

    # =============================
    # A) Table 5.1: headline metrics
    # =============================
    los = sample["LoS"].dropna()
    prolonged_threshold = float(np.percentile(los, 75)) if len(los) else float("nan")

    breach_count = int(sample["breach"].fillna(0).sum())
    breach_rate_pct = (
        float(sample["breach"].mean() * 100) if sample["breach"].notna().any() else float("nan")
    )
    prolonged_rate_pct = (
        float((sample["LoS"] >= prolonged_threshold).mean() * 100)
        if np.isfinite(prolonged_threshold)
        else float("nan")
    )

    table_5_1 = pd.DataFrame(
        {
            "Metric": [
                "Prolonged threshold (LoS ≥ 75th percentile)",
                "Prolonged stay rate (%)",
                "4-hour target threshold (minutes)",
                "Breach count",
                "Breach rate (%)",
                "Sample size",
                "Random seed",
            ],
            "Value": [
                prolonged_threshold,
                prolonged_rate_pct,
                float(los_target_min),
                float(breach_count),
                breach_rate_pct,
                float(len(sample)),
                float(seed),
            ],
        }
    ).round(2)

    # =============================
    # B) Table 5.2: Mann–Whitney U
    # =============================
    numeric_vars = [
        v
        for v in ["LoS", "Age", "noofpatients", "noofinvestigation", "nooftreatment"]
        if v in sample.columns
    ]

    rows_5_2 = []
    for v in numeric_vars:
        a = sample.loc[sample["breach"] == 0, v].dropna()
        b = sample.loc[sample["breach"] == 1, v].dropna()

        p = np.nan
        if len(a) and len(b):
            try:
                _, p = mannwhitneyu(a, b, alternative="two-sided")
            except Exception:
                p = np.nan

        rows_5_2.append(
            {
                "Variable": v,
                "Median (Non-breach)": float(a.median()) if len(a) else np.nan,
                "Median (Breach)": float(b.median()) if len(b) else np.nan,
                "n (Non-breach)": int(len(a)),
                "n (Breach)": int(len(b)),
                "p-value": float(p) if np.isfinite(p) else np.nan,
            }
        )

    table_5_2 = pd.DataFrame(rows_5_2)
    if not table_5_2.empty:
        table_5_2["p-value"] = table_5_2["p-value"].map(lambda x: f"{x:.4f}" if pd.notna(x) else "")

    # =============================
    # C) Table 5.3: Chi-square tests
    # =============================
    cat_vars = [v for v in ["DayofWeek", "Period", "HRG"] if v in sample.columns]
    rows_5_3 = []

    for v in cat_vars:
        tmp = sample.dropna(subset=[v, "breach"]).copy()
        if tmp.empty:
            continue

        ct = pd.crosstab(tmp[v], tmp["breach"])
        if 0 not in ct.columns:
            ct[0] = 0
        if 1 not in ct.columns:
            ct[1] = 0
        ct = ct[[0, 1]]

        try:
            chi2, p, dof, exp = chi2_contingency(ct.values)
            exp_min = float(np.min(exp)) if exp.size else np.nan
            low_expected = bool(exp_min < 5) if np.isfinite(exp_min) else False
        except Exception:
            chi2, p, dof, exp_min, low_expected = np.nan, np.nan, np.nan, np.nan, False

        rows_5_3.append(
            {
                "Variable": v,
                "Chi-square": float(chi2) if np.isfinite(chi2) else np.nan,
                "dof": int(dof) if pd.notna(dof) else np.nan,
                "p-value": float(p) if np.isfinite(p) else np.nan,
                "Min expected": exp_min,
                "Low expected?": low_expected,
            }
        )

    table_5_3 = pd.DataFrame(rows_5_3).round(4)
    if not table_5_3.empty:
        table_5_3["p-value"] = table_5_3["p-value"].map(lambda x: f"{x:.4f}" if pd.notna(x) else "")

    # =============================
    # D) Notebook-like modelling outputs
    #    - describe, Spearman corr
    #    - VIF
    #    - logit m1, logit m2 (+ summaries)
    # =============================
    model_cols = [
        c
        for c in ["breach", "noofpatients", "noofinvestigation", "nooftreatment", "Age", "Period"]
        if c in sample.columns
    ]
    model_df = sample[model_cols].dropna().copy()

    model_df_describe = None
    corr_matrix = None
    vif_df = None

    logit_m1_summary = None
    logit_m2_summary = None
    logit_m1_coef = None
    logit_m2_coef = None

    if not model_df.empty:
        model_df_describe = model_df.describe().T.reset_index().rename(columns={"index": "variable"})

        corr_vars = [c for c in ["noofpatients", "noofinvestigation", "nooftreatment", "Age"] if c in model_df.columns]
        if len(corr_vars) >= 2:
            corr_matrix = model_df[corr_vars].corr(method="spearman")

        # ---- LOGIT + VIF ----
        if not _HAS_SM:
            logit_m1_summary = "Logit m1 skipped: statsmodels not installed in this environment."
        else:
            vc = model_df["breach"].value_counts(dropna=True)
            has_both = (vc.get(0, 0) > 0) and (vc.get(1, 0) > 0)

            if not has_both:
                logit_m1_summary = (
                    "Logit m1 skipped: target 'breach' has only one class in model_df.\n"
                    f"value_counts:\n{vc.to_string()}"
                )
            elif len(model_df) < 20:
                logit_m1_summary = f"Logit m1 skipped: too few rows after dropna (n={len(model_df)})."
            else:
                # m1 (your notebook)
                try:
                    m1 = smf.logit(
                        "breach ~ noofpatients + noofinvestigation + nooftreatment + Age",
                        data=model_df,
                    ).fit(disp=False)
                    logit_m1_summary = m1.summary().as_text()
                    logit_m1_coef = _coef_table(m1)
                except Exception as e:
                    logit_m1_summary = f"Logit m1 failed: {type(e).__name__}: {e}"

                # m2 (PeriodBlock)
                if "Period" in model_df.columns:
                    try:
                        mdf2 = model_df.copy()
                        mdf2["PeriodBlock"] = mdf2["Period"].apply(_time_block)
                        m2 = smf.logit(
                            "breach ~ noofpatients + noofinvestigation + nooftreatment + Age + C(PeriodBlock)",
                            data=mdf2,
                        ).fit(disp=False)
                        logit_m2_summary = m2.summary().as_text()
                        logit_m2_coef = _coef_table(m2)
                    except Exception as e:
                        logit_m2_summary = f"Logit m2 failed: {type(e).__name__}: {e}"

                # VIF
                try:
                    need_vif = ["noofpatients", "noofinvestigation", "nooftreatment", "Age"]
                    if all(c in model_df.columns for c in need_vif):
                        X = model_df[need_vif].copy()
                        X["Intercept"] = 1
                        vif_df = pd.DataFrame(
                            {
                                "Variable": X.columns,
                                "VIF": [
                                    variance_inflation_factor(X.values, i)
                                    for i in range(X.shape[1])
                                ],
                            }
                        ).round(4)
                except Exception:
                    vif_df = None

    # =============================
    # E) Figures (ALL)
    # =============================
    figures: Dict[str, Any] = {}

    # Fig 1: correlation heatmap
    if corr_matrix is not None:
        fig, ax = _new_fig()
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
        ax.set_title("Spearman Correlation Matrix of Numeric Predictors")
        fig.tight_layout()
        figures["corr_heatmap"] = fig

    # Fig 2: LoS by breach
    if sample["breach"].notna().any():
        fig, ax = _new_fig()
        tmp = sample.dropna(subset=["LoS", "breach"]).copy()
        tmp["breach_label"] = tmp["breach"].map({0: "Non-breach", 1: "Breach"})
        sns.boxplot(x="breach_label", y="LoS", data=tmp, ax=ax)
        ax.set_xlabel("")
        ax.set_ylabel("LoS (minutes)")
        ax.set_title("LoS by Breach Status")
        fig.tight_layout()
        figures["los_by_breach_box"] = fig

    # Fig 3: Investigations by breach
    if "noofinvestigation" in sample.columns and sample["breach"].notna().any():
        fig, ax = _new_fig()
        tmp = sample.dropna(subset=["noofinvestigation", "breach"]).copy()
        tmp["breach_label"] = tmp["breach"].map({0: "Non-breach", 1: "Breach"})
        sns.boxplot(x="breach_label", y="noofinvestigation", data=tmp, ax=ax)
        ax.set_xlabel("")
        ax.set_ylabel("No. of investigations")
        ax.set_title("Investigations by Breach Status")
        fig.tight_layout()
        figures["investigations_by_breach_box"] = fig

    # Fig 4: breach rate by day of week
    if "DayofWeek" in sample.columns and sample["breach"].notna().any():
        fig, ax = _new_fig()
        tmp = sample.dropna(subset=["DayofWeek", "breach"]).copy()
        rate = tmp.groupby("DayofWeek")["breach"].mean().mul(100).sort_index()
        ax.bar(rate.index.astype(str), rate.values)
        ax.set_xlabel("DayofWeek")
        ax.set_ylabel("Breach rate (%)")
        ax.set_title("Breach Rate by Day of Week")
        ax.tick_params(axis="x", rotation=30)
        fig.tight_layout()
        figures["breach_rate_by_dayofweek_bar"] = fig

    # =============================
    # F) Tables pack (include ALL)
    # =============================
    tables: Dict[str, Any] = {
        "table_5_1": table_5_1,
        "table_5_2": table_5_2,
        "table_5_3": table_5_3,
    }
    if model_df_describe is not None:
        tables["model_df_describe"] = model_df_describe
    if corr_matrix is not None:
        tables["corr_matrix_spearman"] = corr_matrix.reset_index().rename(
            columns={"index": "variable"}
        )
    if vif_df is not None:
        tables["vif_table"] = vif_df
    if logit_m1_coef is not None:
        tables["logit_m1_coef"] = logit_m1_coef
    if logit_m2_coef is not None:
        tables["logit_m2_coef"] = logit_m2_coef

    summary = (
        "Task 5 investigates factors associated with breaches/prolonged stays using a random sample (n=400). "
        "Outputs include descriptive stats for modelling variables, Spearman correlation, VIF, and logistic regression "
        "(where available), plus non-parametric group comparisons and chi-square tests."
    )

    stats = {
        "seed": int(seed),
        "sample_n": int(len(sample)),
        "breach_count": int(breach_count),
        "breach_rate_pct": round(breach_rate_pct, 2) if np.isfinite(breach_rate_pct) else breach_rate_pct,
        "prolonged_threshold": round(prolonged_threshold, 2) if np.isfinite(prolonged_threshold) else prolonged_threshold,
        "prolonged_rate_pct": round(prolonged_rate_pct, 2) if np.isfinite(prolonged_rate_pct) else prolonged_rate_pct,
        "los_target_min": int(los_target_min),
        "statsmodels_available": bool(_HAS_SM),
        "model_df_rows_after_dropna": int(len(model_df)),
    }

    return {
        "name": "Task 5",
        "summary": summary,
        "n": int(len(sample)),
        "seed": int(seed),
        "stats": stats,
        "tables": tables,
        "figures": figures,
        "logit_m1_summary": logit_m1_summary,
        "logit_m2_summary": logit_m2_summary,
        "sample": sample,
        "download_df": sample,
    }
