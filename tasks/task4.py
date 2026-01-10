# tasks/task4.py
from __future__ import annotations

from typing import Dict, Any, Tuple
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _new_fig(figsize: Tuple[float, float] = (5.5, 3.5)):
    fig, ax = plt.subplots(figsize=figsize)
    return fig, ax


def _safe_corr(df: pd.DataFrame, x: str, y: str) -> float:
    tmp = df[[x, y]].dropna()
    if len(tmp) < 2:
        return float("nan")
    return float(tmp.corr().iloc[0, 1])


def _load_df(df_or_path):
    if df_or_path is None:
        path = "data/AED4weeks.csv"
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"[Task 4] Default dataset not found: {path}. "
                "Pass a DataFrame or a path explicitly."
            )
        return pd.read_csv(path)

    if isinstance(df_or_path, str):
        return pd.read_csv(df_or_path)

    if isinstance(df_or_path, pd.DataFrame):
        return df_or_path.copy()

    raise TypeError("solve_task4 expects None, a DataFrame, or a CSV path.")


def _standardise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Make column names robust (so CSV/notebook naming differences won't break Task 4).
    Only maps what Task 4 needs.
    """
    aliases = {
        "Age": ["Age", "age"],
        "LoS": ["LoS", "LOS", "Length of Stay", "length_of_stay"],
        "noofinvestigation": [
            "noofinvestigation",
            "No. of investigations",
            "Investigations",
        ],
        "nooftreatment": ["nooftreatment", "No. of treatments", "Treatments"],
        "Day": ["Day"],
        "DayofWeek": ["DayofWeek", "Day of week", "Day_of_week"],
        "Period": ["Period", "Hour", "Arrival hour"],
        "Breachornot": ["Breachornot", "Breach or not", "Breach"],
        "noofpatients": ["noofpatients", "Number of patients", "Patients"],
        "HRG": ["HRG"],
    }

    rename_map = {}
    for canonical, candidates in aliases.items():
        for c in candidates:
            if c in df.columns:
                rename_map[c] = canonical
                break

    df = df.rename(columns=rename_map)

    required = list(aliases.keys())
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(
            f"[Task 4] Missing required columns: {missing}. "
            f"Available columns: {list(df.columns)}"
        )
    return df


def solve_task4(
    df_or_path=None,
    *,
    seed: int = 123,
    n_sample: int = 400,
) -> Dict[str, Any]:
    """
    Task 4 (simple output):
    - works with solve_task4(), solve_task4(df), solve_task4("path.csv")
    - returns only the keys your app/CLI needs
    - keeps ALL figures
    """
    df = _standardise_columns(_load_df(df_or_path))

    # sample
    sample = df.sample(
        n=min(n_sample, len(df)), random_state=seed, replace=False
    ).copy()

    # breach rate (%)
    breach_props = (
        sample["Breachornot"]
        .astype(str)
        .str.strip()
        .str.lower()
        .value_counts(normalize=True)
        * 100
    )
    breach_rate_pct = float(breach_props.get("breach", np.nan))

    # breach counts
    breach_counts = (
        sample["Breachornot"]
        .value_counts(dropna=False)
        .rename("count")
        .reset_index()
        .rename(columns={"index": "Breachornot"})
    )

    # summaries (nice for dashboard)
    age_summary = sample["Age"].describe().to_frame(name="Age").round(2)
    los_summary = sample["LoS"].describe().to_frame(name="LoS (minutes)").round(2)

    # correlations (for KPI table)
    corr_patients_los = _safe_corr(sample, "noofpatients", "LoS")
    corr_invest_los = _safe_corr(sample, "noofinvestigation", "LoS")

    kpi_table = pd.DataFrame(
        {
            "Metric": [
                "Sample size",
                "Random seed",
                "Breach rate (%)",
                "Corr(no. patients, LoS)",
                "Corr(no. investigations, LoS)",
            ],
            "Value": [
                int(len(sample)),
                int(seed),
                round(breach_rate_pct, 2) if np.isfinite(breach_rate_pct) else np.nan,
                round(corr_patients_los, 3),
                round(corr_invest_los, 3),
            ],
        }
    )

    # ---- breach rate by period (for 2 plots) ----
    b = sample["Breachornot"].astype(str).str.strip().str.lower()
    breach_int = b.map({"breach": 1, "non-breach": 0})
    tmp = sample.copy()
    tmp["Breachornot_int"] = pd.to_numeric(breach_int, errors="coerce")

    breach_by_period = (
        tmp.dropna(subset=["Period", "Breachornot_int"])
        .groupby("Period")["Breachornot_int"]
        .mean()
        .mul(100)
        .sort_index()
    )

    # ---- figures (ALL) ----
    figures: Dict[str, Any] = {}

    # 1) Age histogram
    fig, ax = _new_fig()
    ax.hist(sample["Age"].dropna(), bins=30)
    ax.set_xlabel("Age in Years")
    ax.set_ylabel("Number of patients")
    ax.set_title("Age wise distribution of Patients")
    figures["age_hist"] = fig

    # 2) LoS histogram
    fig, ax = _new_fig()
    ax.hist(sample["LoS"].dropna(), bins=50)
    ax.set_xlabel("Length of Stay")
    ax.set_ylabel("Number of Patients")
    ax.set_title("Distribution of Length of Stay")
    figures["los_hist"] = fig

    # 3) LoS boxplot
    fig, ax = _new_fig()
    ax.boxplot(sample["LoS"].dropna(), vert=False)
    ax.set_xlabel("Length of Stay")
    ax.set_title("Boxplot of LoS")
    figures["los_box"] = fig

    # 4) noofinvestigation histogram
    fig, ax = _new_fig()
    ax.hist(sample["noofinvestigation"].dropna(), bins=30)
    ax.set_xlabel("No. of Investigations")
    ax.set_ylabel("Number of Patients")
    ax.set_title("No. of Investigations for Patients")
    figures["investigation_hist"] = fig

    # 5) nooftreatment histogram
    fig, ax = _new_fig()
    ax.hist(sample["nooftreatment"].dropna(), bins=10)
    ax.set_xlabel("No. of Treatments")
    ax.set_ylabel("Number of Patients")
    ax.set_title("No. of Treatments for Patients")
    figures["treatment_hist"] = fig

    # 6) arrivals over study period
    fig, ax = _new_fig()
    day_series = sample["Day"].value_counts(dropna=False).sort_index()
    ax.plot(day_series.index, day_series.values)
    ax.set_ylabel("Number of Patients")
    ax.set_title("Patient Arrivals Over Study Period")
    figures["arrivals_day_line"] = fig

    # 7) arrivals by day of week
    fig, ax = _new_fig()
    day_count = sample["DayofWeek"].value_counts(dropna=False)
    ax.bar(day_count.index.astype(str), day_count.values)
    ax.set_xlabel("Day of week")
    ax.set_ylabel("Number of patients")
    ax.set_title("Arrivals by day of week")
    ax.tick_params(axis="x", rotation=30)
    figures["arrivals_dow_bar"] = fig

    # 8) arrivals by time of day
    fig, ax = _new_fig()
    period_count = sample["Period"].value_counts(dropna=False).sort_index()
    ax.bar(period_count.index.astype(str), period_count.values)
    ax.set_xlabel("Hour of Arrival")
    ax.set_ylabel("Number of Patients")
    ax.set_title("Arrivals by time of day")
    figures["arrivals_period_bar"] = fig

    # 9) crowding vs LoS
    fig, ax = _new_fig()
    ax.scatter(sample["noofpatients"], sample["LoS"], alpha=0.6)
    ax.set_xlabel("Number of patients already in AED")
    ax.set_ylabel("Length of Stay (minutes)")
    ax.set_title("Crowding vs Length of Stay")
    figures["crowding_vs_los_scatter"] = fig

    # 10) investigations vs LoS
    fig, ax = _new_fig()
    ax.scatter(sample["noofinvestigation"], sample["LoS"], alpha=0.6)
    ax.set_xlabel("Number of Investigations")
    ax.set_ylabel("Length of Stay")
    ax.set_title("Investigations vs Length of Stay")
    figures["investigation_vs_los_scatter"] = fig

    # 11) breach rate by time of day (bar)
    fig, ax = _new_fig()
    if len(breach_by_period):
        ax.bar(breach_by_period.index.astype(str), breach_by_period.values)
    ax.set_xlabel("Hour of arrival (0–23)")
    ax.set_ylabel("Breach rate (%)")
    ax.set_title("Breach rate by time of day (bar)")
    figures["breach_rate_by_period_bar"] = fig

    # 12) breach rate by time of day (line)
    fig, ax = _new_fig()
    if len(breach_by_period):
        ax.plot(breach_by_period.index, breach_by_period.values, marker="o")
    ax.set_xlabel("Hour of arrival (0–23)")
    ax.set_ylabel("Breach rate (%)")
    ax.set_title("Breach rate by time of day (line)")
    figures["breach_rate_by_period_line"] = fig

    # 13) LoS by breach status
    fig, ax = _new_fig()
    g = sample.dropna(subset=["Breachornot", "LoS"]).groupby("Breachornot")
    groups, labels = [], []
    for k, sub in g:
        labels.append(str(k))
        groups.append(sub["LoS"].values)
    if groups:
        ax.boxplot(groups, labels=labels)
    ax.set_xlabel("Breach status")
    ax.set_ylabel("Length of Stay (minutes)")
    ax.set_title("Length of Stay by Breach Status")
    figures["los_by_breach_box"] = fig

    summary = (
        "Descriptive overview of a random AED sample (n=400), showing outcome distribution, "
        "key variable summaries, time patterns, and core relationships with length of stay."
    )

    tables = {
        "breach_counts": breach_counts,
        "kpi_table": kpi_table,
        "age_summary": age_summary,
        "los_summary": los_summary,
    }

    # ---- print_task4 ----

    return {
        "name": "Task 4",
        "n": int(len(sample)),
        "seed": int(seed),
        "summary": summary,
        "breach_rate_pct": breach_rate_pct,
        "breach_counts": breach_counts,
        "kpi_table": kpi_table,
        "age_summary": age_summary,
        "los_summary": los_summary,
        "tables": tables,
        "figures": figures,
        "sample": sample,
        "download_df": sample,
    }
