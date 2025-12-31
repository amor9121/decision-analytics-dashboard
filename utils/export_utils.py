import os
import pandas as pd
from .schedule_utils import build_final_table


def grouped_csv_text(results, days, wage) -> str:
    cols = days + ["Weekly Total", "Hourly Wage (£)", "Weekly Wage (£)"]
    blocks = []

    for r in results:
        allocation = r.get("allocation")
        if allocation is None:
            continue

        final_table = build_final_table(allocation, days, wage)
        numeric = final_table[cols].reset_index(drop=True)

        title_row = pd.DataFrame(
            [[r.get("name", "Task")] + [""] * (len(cols) - 1)], columns=cols
        )
        header_row = pd.DataFrame([cols], columns=cols)

        blocks.extend([title_row, header_row, numeric])

    out = pd.concat(blocks, ignore_index=True)
    return out.to_csv(index=False, header=False)


def flat_csv_text(results, days, wage) -> str:
    rows = []
    cols = days + ["Weekly Total", "Hourly Wage (£)", "Weekly Wage (£)"]

    for r in results:
        name = r.get("name", "Task")
        allocation = r.get("allocation")
        if allocation is None:
            continue

        ft = build_final_table(allocation, days, wage)[cols].copy()
        ft.insert(0, "Operator", ft.index)
        ft.insert(0, "Scenario", name)

        ft["Total cost (£)"] = r.get("cost")
        ft["Cost increase (%)"] = r.get("cost_increase_pct")
        ft["Fairness gap"] = r.get("gap")

        rows.append(ft.reset_index(drop=True))

    out = pd.concat(rows, ignore_index=True)
    return out.to_csv(index=False)


def single_task_csv_text(task_result: dict, days, wage) -> str:
    """Return a CSV for one task result (one schedule)."""
    allocation = task_result.get("allocation")
    if allocation is None:
        return ""

    ft = build_final_table(allocation, days, wage).copy()
    ft.insert(0, "Operator", ft.index)
    ft.insert(0, "Scenario", task_result.get("name", "Task"))

    # add useful metadata columns (optional)
    ft["Total cost (£)"] = task_result.get("cost")
    ft["Cost increase (%)"] = task_result.get("cost_increase_pct")
    ft["Fairness gap"] = task_result.get("gap")

    return ft.reset_index(drop=True).to_csv(index=False)


def export_csv(results, days, wage, filepath="outputs/scheduling_grouped.csv"):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    cols = days + ["Weekly Total", "Hourly Wage (£)", "Weekly Wage (£)"]
    blocks = []

    for r in results:
        name = r.get("name", "Task")
        allocation = r.get("allocation")
        if allocation is None:
            continue

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

        table = table[cols].round(2).reset_index(drop=True)

        title_row = pd.DataFrame([[name] + [""] * (len(cols) - 1)], columns=cols)
        header_row = pd.DataFrame([cols], columns=cols)

        blocks.extend([title_row, header_row, table])

    out = pd.concat(blocks, ignore_index=True)

    out.to_csv(filepath, index=False, header=False)
    print(f"Saved: {filepath}")
