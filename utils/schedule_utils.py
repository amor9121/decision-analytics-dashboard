import pandas as pd
from core.data import days


def build_final_table(allocation: pd.DataFrame, days, wage: dict) -> pd.DataFrame:
    table = allocation.copy()
    table["Weekly Total"] = table[days].sum(axis=1)
    table["Hourly Wage (£)"] = table.index.map(lambda i: wage.get(i, ""))
    table["Weekly Wage (£)"] = table.index.map(
        lambda i: table.loc[i, "Weekly Total"] * wage[i] if i in wage else ""
    )

    daily = table[days].sum(axis=0)
    daily["Weekly Total"] = table["Weekly Total"].sum()
    daily["Hourly Wage (£)"] = ""
    daily["Weekly Wage (£)"] = table["Weekly Wage (£)"].sum()
    table.loc["Daily Total"] = daily

    return table.round(2)


def check_daily_coverage(allocation: pd.DataFrame, days, required=14) -> dict:
    daily = allocation[days].sum(axis=0)
    ok = bool((daily.round(6) == required).all())
    return {"ok": ok, "daily_totals": daily.round(2)}


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
                "Daily coverage = 14?": coverage_cell,
                "Has skill coverage?": (
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

        rows.append(
            {
                "Task": name,
                "Case": r.get("case", ""),
                "Sample": "n = 400" if name in ["Task 4", "Task 5", "Task 6"] else "",
                "Breach rate (%)": r.get("breach_rate_pct", ""),
                "Main method": (
                    "Descriptive analysis"
                    if name == "Task 4"
                    else (
                        "Statistical tests"
                        if name == "Task 5"
                        else "Prediction (ML)" if name == "Task 6" else ""
                    )
                ),
            }
        )

    return pd.DataFrame(rows)
