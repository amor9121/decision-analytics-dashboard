import pandas as pd


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
