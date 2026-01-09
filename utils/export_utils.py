import io
import json
import os
import zipfile
import pandas as pd
from .schedule_utils import build_final_table
from datetime import datetime
from utils.figure_utils import collect_task_figures, figures_zip_bytes


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


def task_bundle_zip_bytes(task_result: dict, days, wage, *, dpi: int = 200) -> bytes:

    task_name = task_result.get("name", "Task")
    safe_name = task_name.replace("/", "-").replace("\\", "-")

    # schedule.csv (Task 1–3 only)
    csv_text = single_task_csv_text(task_result, days, wage)

    # figures.zip (Task 4–6 only) - reuse your existing functions
    figs = collect_task_figures(task_result)
    figs_zip = figures_zip_bytes(figs, dpi=dpi) if figs else None

    # metrics.json (always)
    metrics = {
        "task": task_name,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "cost": task_result.get("cost"),
        "cost_increase_pct": task_result.get("cost_increase_pct"),
        "fairness_gap": task_result.get("gap"),
    }

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(
            f"{safe_name}/metrics.json", json.dumps(metrics, indent=2, default=str)
        )

        if csv_text:
            zf.writestr(f"{safe_name}/schedule.csv", csv_text)

        if figs_zip:
            zf.writestr(f"{safe_name}/figures.zip", figs_zip)

    buf.seek(0)
    return buf.getvalue()


def all_tasks_zip_bytes(results, days, wage, *, dpi=200):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for r in results:

            # ---------- 1. Allocation-based CSV (Task 1–5) ----------
            csv_text = single_task_csv_text(r, days, wage)
            if csv_text:
                task_name = r.get("name", "Task").replace(" ", "_")
                zf.writestr(f"{task_name}.csv", csv_text)

            # ---------- 2. Task figures ----------
            figures = collect_task_figures(r)
            for filename, fig in figures.items():
                fig_buf = io.BytesIO()
                fig.savefig(fig_buf, format="png", dpi=dpi, bbox_inches="tight")
                fig_buf.seek(0)
                zf.writestr(filename, fig_buf.read())

            # ---------- 3. Extra tables ----------
            tables = r.get("tables")
            if isinstance(tables, dict):
                task_name = r.get("name", "Task").replace(" ", "_")
                for key, df in tables.items():
                    if hasattr(df, "to_csv"):
                        csv_text = df.to_csv(index=False)
                        zf.writestr(f"{task_name}_{key}.csv", csv_text)

    return buf.getvalue()


def ensure_task_row(obj, default_name, default_status="Done"):
    """Make sure a task output is a dict with at least name/status so it shows in summary."""
    if isinstance(obj, dict):
        obj.setdefault("name", default_name)
        obj.setdefault("status", default_status)
        obj.setdefault(
            "allocation", None
        )  # descriptive tasks typically have no allocation
        return obj
    return {"name": default_name, "status": default_status, "allocation": None}
