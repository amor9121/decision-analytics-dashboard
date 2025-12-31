import streamlit as st
import pandas as pd

from core.data import days, wage
from tasks.task1 import solve_task1
from tasks.task2_s1 import solve_task2_s1
from tasks.task2_s2 import solve_task2_s2
from tasks.task3 import solve_task3

from tasks.task4 import solve_task4
from tasks.task5 import solve_task5

# ===== DEBUG ONLY (REMOVE BEFORE SUBMISSION) =====
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
# =================================================


#  ---- Helpers ----


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


def metric_row(r: dict):
    c1, c2, c3 = st.columns(3)
    c1.metric("Total cost (£)", r.get("cost", "-"))
    c2.metric("Cost increase (%)", r.get("cost_increase_pct", "-"))
    c3.metric("Fairness gap", "-" if r.get("gap") is None else r.get("gap"))


def render_task_block(r: dict):
    allocation = r.get("allocation")
    if allocation is None:
        st.warning("No allocation returned for this task.")
        return

    cov = check_daily_coverage(allocation, days)
    if cov["ok"]:
        st.success("Daily coverage satisfied: 14 hours per day ✅")
    else:
        st.error("Daily coverage NOT satisfied ❌")
    st.caption(f"Daily totals: {cov['daily_totals'].to_dict()}")

    # Task 3: skill coverage
    if r.get("skill_coverage") is not None:
        st.subheader("Skill coverage per day (should be ≥ 6 each)")
        st.dataframe(
            pd.DataFrame(r["skill_coverage"]).T.round(2), use_container_width=True
        )

    st.subheader("Final Schedule Table")
    st.dataframe(build_final_table(allocation, days, wage), use_container_width=True)


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


def render_task_downloads(results, days, wage, n_cols=3):
    with st.expander("⬇️ Downloads", expanded=True):
        cols = st.columns(n_cols)
        i = 0

        for r in results:
            csv_text = single_task_csv_text(r, days, wage)
            if not csv_text:
                continue

            task_name = r.get("name", "Task")
            safe_name = task_name.replace("/", "-")

            with cols[i % n_cols]:
                st.download_button(
                    label=f"Download {task_name}",
                    data=csv_text.encode("utf-8"),
                    file_name=f"{safe_name}.csv",
                    mime="text/csv",
                    key=f"dl_{safe_name}_{i}",  # avoid key collisions
                )
            i += 1


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


# ---- Streamlit UI ----

st.set_page_config(page_title="Decision Analytics Dashboard", layout="wide")
st.title("Decision Analytics Dashboard")

st.markdown(
    """
**What this app does**
- **Task 1**: Baseline cost-minimising schedule (LP)
- **Task 2 – Scenario 1**: Minimise workload inequality **subject to ≤ 1.8% cost increase**
- **Task 2 – Scenario 2**: Find the **fairest** possible schedule, then minimise cost (two-stage)
- **Task 3**: Add **skill constraints** (≥ 6 hours per skill per day) and report feasibility/cost
- **Task 4**: Analyse a random sample of AED patient data to understand workload, patient flow, and breaches
- **Task 5**: Use **statistical analysis** to investigate factors contributing to breaches or prolonged stays
- **Task 6**: Apply **machine learning** to predict whether a patient will breach the 4-hour target
"""
)

# Keep results in session_state (use empty list to avoid NoneType errors)
if "results" not in st.session_state:
    st.session_state["results"] = []

left, right = st.columns([1, 2])
with left:
    run_all = st.button("Run Tasks 1–6", type="primary")
with right:
    st.caption("Tip: run once, then explore tabs + download CSVs.")

# ---- Run All ----

if run_all:
    # ---- Task 1 ----
    t1 = ensure_task_row(solve_task1(), "Task 1")
    t1["case"] = "Baseline (min cost)"
    t1["download_df"] = pd.DataFrame(t1.get("allocation", []))

    baseline_cost = t1.get("cost", 0)

    # ---- Task 2 – Scenario 1 ----
    t2s1 = ensure_task_row(
        solve_task2_s1(baseline_cost),
        "Task 2 - Scenario 1",
    )
    t2s1["case"] = "≤1.8% cost"
    t2s1["download_df"] = pd.DataFrame(t2s1.get("allocation", []))

    # ---- Task 2 – Scenario 2 ----
    t2s2 = ensure_task_row(
        solve_task2_s2(baseline_cost),
        "Task 2 - Scenario 2",
    )
    t2s2["case"] = "Two-stage"
    t2s2["download_df"] = pd.DataFrame(t2s2.get("allocation", []))

    # ---- Task 3 ----
    t3 = ensure_task_row(
        solve_task3(baseline_cost),
        "Task 3",
    )
    t3["case"] = "Skill constraints"
    t3["download_df"] = pd.DataFrame(t3.get("allocation", []))

    # ---- Task 4 ----
    t4 = ensure_task_row(solve_task4(), "Task 4")
    t4["case"] = "Random sample (n=400)"
    t4["download_df"] = t4.get("download_df", pd.DataFrame())

    # t4 = {
    #    "name": "Task 4",
    #    "case": "None",
    #    "status": "Pending",
    #    "allocation": None,
    #    "download_df": pd.DataFrame(),
    # }

    # ---- Task 5 ----
    t5 = ensure_task_row(solve_task5(), "Task 5")
    t5["case"] = "Statistical analysis of breach factors"
    t5["download_df"] = t5.get("download_df", pd.DataFrame())

    # ---- Task 6 ----
    t6 = {
        "name": "Task 6",
        "case": "Prediction (ML)",
        "status": "Pending",
        "allocation": None,
        "download_df": pd.DataFrame(),
    }

    # ---- Save ALL results once ----
    st.session_state["results"] = [t1, t2s1, t2s2, t3, t4, t5, t6]

results = st.session_state["results"]

if not results:
    st.info("Click **Run Tasks 1–6** to generate results.")
    st.stop()

# ---- Summary ----

schedule_summary_df = build_schedule_summary(results)
aed_summary_df = build_aed_summary(results)

st.subheader("1. Scheduling Optimisation Summary (Tasks 1–3)")
st.dataframe(schedule_summary_df, use_container_width=True, hide_index=True)

st.subheader("2. AED Analytics & Prediction Summary (Tasks 4–6)")
st.dataframe(aed_summary_df, use_container_width=True, hide_index=True)


# ---- Tabs ----

st.subheader("3. Results")
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(
    [
        "Task 1",
        "Task 2 – Scenario 1",
        "Task 2 – Scenario 2",
        "Task 3",
        "Task 4 (AED)",
        "Task 5",
        "Task 6",
    ]
)

# ---- Map by name ----

by_name = {r.get("name"): r for r in results}

with tab1:
    st.info(
        "Task 1 minimises total labour cost subject to availability, daily coverage, and minimum weekly hours."
    )
    r = by_name.get("Task 1") or results[0]
    metric_row(r)
    render_task_block(r)

with tab2:
    st.info(
        "Scenario 1 minimises fairness gap (max–min weekly hours) subject to cost ≤ baseline × 1.018."
    )
    r = (
        by_name.get("Task 2 - Senerio 1")
        or by_name.get("Task 2 - Scenario 1")
        or results[1]
    )
    metric_row(r)
    render_task_block(r)

with tab3:
    st.info(
        "Scenario 2 finds the minimum possible fairness gap first, then minimises cost under that gap."
    )
    r = (
        by_name.get("Task 2 - Senerio 2")
        or by_name.get("Task 2 - Scenario 2")
        or results[2]
    )
    metric_row(r)
    render_task_block(r)

with tab4:
    st.info(
        "Task 3 adds skill constraints: each day, Programming ≥ 6 and Troubleshooting ≥ 6 hours."
    )
    r = by_name.get("Task 3") or results[3]
    metric_row(r)
    render_task_block(r)

with tab5:
    st.info(
        "Task 4 analyses AED patient data to understand workload, patient flow, and breaches."
    )

    r = by_name.get("Task 4")
    if r is None:
        r = results[4] if len(results) > 4 else {}

    # ---- Summary ----
    st.write(r.get("summary", ""))

    # ---- Metrics row ----
    c1, c2, c3 = st.columns(3)
    c1.metric("Sample size", r.get("n", 400))
    c2.metric("Random seed", r.get("seed", 123))
    br = r.get("breach_rate_pct", None)
    c3.metric("Breach rate (%)", f"{br:.2f}" if isinstance(br, (int, float)) else "-")

    st.divider()

    # ---- KPI Table (Breach vs Non-breach) ----
    st.subheader("KPI table (Breach vs Non-breach)")
    if "kpi_table" in r and r["kpi_table"] is not None:
        st.dataframe(r["kpi_table"])
    else:
        st.warning(
            "kpi_table not available. (Make sure solve_task4() returns 'kpi_table')"
        )

    st.divider()

    # ---- Numerical summaries ----
    st.subheader("Numerical summaries")

    st.write("Age summary")
    if "age_summary" in r and r["age_summary"] is not None:
        st.dataframe(r["age_summary"])
    else:
        st.warning("age_summary not available.")

    st.write("Length of Stay (LoS) summary")
    if "los_summary" in r and r["los_summary"] is not None:
        st.dataframe(r["los_summary"])
    else:
        st.warning("los_summary not available.")

    st.divider()

    # ---- Figures ----
    st.subheader("Figure 1 – Core relationships")
    if "fig1" in r and r["fig1"] is not None:
        st.pyplot(r["fig1"])
    else:
        st.warning("fig1 not available.")

    st.subheader("Figure 2 – Additional insights")
    if "fig2" in r and r["fig2"] is not None:
        st.pyplot(r["fig2"])
    else:
        st.warning("fig2 not available.")

    st.divider()

    # ---- Sample preview ----
    with st.expander("Sample preview (first 20 rows)", expanded=True):
        if "sample" in r and r["sample"] is not None:
            st.dataframe(r["sample"].head(20))
        else:
            st.write("sample not available.")

with tab6:
    st.info(
        "Task 5 investigates potential factors associated with breaches or prolonged stays "
        "using statistical analysis and management-focused summaries."
    )

    r = by_name.get("Task 5") or results[5]

    # --- Tables
    st.subheader("Table 5.1 – Prolonged stay & key KPIs")
    st.dataframe(r["tables"]["table_5_1"], use_container_width=True)

    st.subheader("Table 5.2 – Numeric variables vs Breach")
    st.dataframe(r["tables"]["table_5_2"], use_container_width=True)

    st.subheader("Table 5.3 – Categorical variables vs Breach")
    st.dataframe(r["tables"]["table_5_3"], use_container_width=True)

    st.divider()

    # --- Figures
    st.subheader("Figure 5.0 – Overview of breach context")
    st.pyplot(r["figures"]["fig0"], clear_figure=False)

    st.subheader("Figure 5.1 – Key factors associated with breaches")
    st.pyplot(r["figures"]["fig1"], clear_figure=False)


# ---- Download ----

st.divider()
st.subheader("4. Downloads")

with st.expander("Downloads", expanded=True):
    cols = st.columns(4)  # 3 buttons per row
    i = 0

    for r in results:
        csv_text = single_task_csv_text(r, days, wage)
        task_name = r.get("name", "Task")

        if csv_text:
            with cols[i % 4]:
                st.download_button(
                    label=f"{task_name}",
                    data=csv_text.encode("utf-8"),
                    file_name=f"{task_name}.csv".replace("/", "-"),
                    mime="text/csv",
                    key=f"dl_{task_name}",
                )
            i += 1
