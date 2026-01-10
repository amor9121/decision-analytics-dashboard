import io
import json
import os
import zipfile
import matplotlib.pyplot as plt
import pandas as pd
from .schedule_utils import build_final_table
from datetime import datetime
from utils.figure_utils import collect_task_figures


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
    safe_name = (
        task_name.replace("/", "-")
        .replace("\\", "-")
        .replace(":", "-")
        .replace("|", "-")
    )

    # schedule.csv (Task 1–3 only if available)
    csv_text = single_task_csv_text(task_result, days, wage)

    # figures (Task 4–6 or any task that has figures)
    figs = collect_task_figures(task_result)  # expect dict[str, Figure] or list[Figure]

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
        # metrics.json
        zf.writestr(f"{safe_name}/metrics.json", json.dumps(metrics, indent=2, default=str))

        # schedule.csv
        if csv_text:
            zf.writestr(f"{safe_name}/schedule.csv", csv_text)

        # figures as PNGs (no nested zip)
        if figs:
            # normalize into (name, fig) pairs
            if isinstance(figs, dict):
                items = list(figs.items())
            else:
                # list/iterable of figs
                items = [(f"fig{i}", fig) for i, fig in enumerate(figs, start=1)]

            for key, fig in items:
                if fig is None:
                    continue

                img = io.BytesIO()
                fig.savefig(img, format="png", dpi=dpi, bbox_inches="tight")
                img.seek(0)

                fname = str(key).strip().replace("/", "-").replace("\\", "-")
                zf.writestr(f"{safe_name}/figures/{fname}.png", img.read())

                # optional: free memory in long runs
                try:
                    plt.close(fig)
                except Exception:
                    pass
                # tables as CSV (no nested zip)
                tables = task_result.get("tables") or {}
                if isinstance(tables, dict) and tables:
                    for tname, t in tables.items():
                        if t is None:
                            continue
                        # only export pandas objects nicely
                        if hasattr(t, "to_csv"):
                            csv_buf = io.StringIO()
                            try:
                                t.to_csv(csv_buf, index=False)
                            except Exception:
                                # fallback if index is meaningful
                                t.to_csv(csv_buf, index=True)
                            zf.writestr(f"{safe_name}/tables/{tname}.csv", csv_buf.getvalue())

    buf.seek(0)
    return buf.getvalue()


def all_tasks_zip_bytes(results, days, wage, *, dpi: int = 200) -> bytes:

    def _safe_task_folder(name: str) -> str:
        return (
            str(name).strip()
            .replace("/", "-")
            .replace("\\", "-")
            .replace(":", "-")
            .replace("|", "-")
        )

    def _df_to_csv_text(df) -> str:
        try:
            return df.to_csv(index=False)
        except Exception:
            return df.to_csv(index=True)

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for r in results:
            task_name = r.get("name", "Task")
            task_folder = _safe_task_folder(task_name)

            # -------------------------
            # 0) metrics.json (always)
            # -------------------------
            metrics = {
                "task": task_name,
                "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "n": r.get("n"),
                "seed": r.get("seed"),
                "breach_rate_pct": r.get("breach_rate_pct") or (r.get("stats") or {}).get("breach_rate_pct"),
                "cost": r.get("cost"),
                "cost_increase_pct": r.get("cost_increase_pct"),
                "fairness_gap": r.get("gap"),
                "accuracy": r.get("accuracy"),
                "roc_auc": r.get("roc_auc"),
            }
            zf.writestr(f"{task_folder}/metrics.json", json.dumps(metrics, indent=2, default=str))

            # -------------------------
            # 1) schedule.csv (if exists)
            # -------------------------
            csv_text = single_task_csv_text(r, days, wage)
            if csv_text:
                zf.writestr(f"{task_folder}/schedule/schedule.csv", csv_text)

            # -------------------------
            # 2) tables/*.csv (if exists)
            # -------------------------
            tables = r.get("tables") or {}
            if isinstance(tables, dict) and tables:
                for key, df in tables.items():
                    if df is None:
                        continue
                    if hasattr(df, "to_csv"):
                        safe_key = _safe_task_folder(key)
                        zf.writestr(
                            f"{task_folder}/tables/{safe_key}.csv",
                            _df_to_csv_text(df),
                        )

            # (optional fallback) classic Task 4 tables if not inside r["tables"]
            if task_name == "Task 4" and not (isinstance(tables, dict) and tables):
                for key in ["kpi_table", "breach_counts", "age_summary", "los_summary"]:
                    df = r.get(key)
                    if df is not None and hasattr(df, "to_csv"):
                        zf.writestr(
                            f"{task_folder}/tables/{key}.csv",
                            _df_to_csv_text(df),
                        )

            # -------------------------
            # 3) figures/*.png (if exists)
            # -------------------------
            figures = collect_task_figures(r) or {}
            if isinstance(figures, dict) and figures:
                for fig_name, fig in figures.items():
                    if fig is None:
                        continue
                    safe_fig_name = _safe_task_folder(fig_name)
                    fig_buf = io.BytesIO()
                    fig.savefig(fig_buf, format="png", dpi=dpi, bbox_inches="tight")
                    fig_buf.seek(0)
                    zf.writestr(f"{task_folder}/figures/{safe_fig_name}.png", fig_buf.read())

    buf.seek(0)
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
