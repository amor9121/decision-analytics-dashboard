import streamlit as st
import pandas as pd

from core.data import days, wage
from tasks.task1 import solve_task1
from tasks.task2_s1 import solve_task2_s1
from tasks.task2_s2 import solve_task2_s2
from tasks.task3 import solve_task3
from tasks.task4 import solve_task4


# ===== DEBUG ONLY (REMOVE BEFORE SUBMISSION) =====
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
# =================================================


# -----------------------------
# Helpers (Tasks 1–3)
# -----------------------------
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


def build_summary(results: list[dict]) -> pd.DataFrame:
    rows = []
    for r in results:
        allocation = r.get("allocation")
        status = r.get("status", "Optimal")

        coverage_ok = None
        if allocation is not None:
            coverage_ok = check_daily_coverage(allocation, days)["ok"]

        rows.append(
            {
                "Task": r.get("name"),
                "Status": status,
                "Total cost (£)": r.get("cost"),
                "Cost increase (%)": r.get("cost_increase_pct"),
                "Fairness gap": r.get("gap"),
                "Daily coverage = 14?": "✅" if coverage_ok else "❌",
                "Has skill coverage?": (
                    "✅" if r.get("skill_coverage") is not None else ""
                ),
            }
        )
    return pd.DataFrame(rows)


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Scheduling UI", layout="wide")
st.title("Scheduling Results Dashboard")

st.markdown(
    """
**What this app does**
- **Task 1**: Baseline cost-minimising schedule (LP)
- **Task 2 – Scenario 1**: Minimise workload inequality **subject to ≤ 1.8% cost increase**
- **Task 2 – Scenario 2**: Find the **fairest** possible schedule, then minimise cost (two-stage)
- **Task 3**: Add **skill constraints** (≥ 6 hours per skill per day) and report feasibility/cost
- **Task 4**: Analyse a random sample of AED patient data to understand workload, patient flow, and factors contributing to breaches or prolonged stays
"""
)

# Keep results in session_state
if "results" not in st.session_state:
    st.session_state["results"] = None
if "task4" not in st.session_state:
    st.session_state["task4"] = None

left, right = st.columns([1, 2])
with left:
    run_all = st.button("Run Tasks 1–4", type="primary")
with right:
    st.caption("Tip: run once, then explore tabs + download CSVs.")

if run_all:
    t1 = solve_task1()
    baseline_cost = t1["cost"]

    s1 = solve_task2_s1(baseline_cost)
    s2 = solve_task2_s2(baseline_cost)
    t3 = solve_task3(baseline_cost)

    # Task 4
    t4 = solve_task4()

    # Store
    st.session_state["results"] = [t1, s1, s2, t3]  # scheduling tasks only
    st.session_state["task4"] = t4

results = st.session_state["results"]

if results is None:
    st.info("Click **Run Tasks 1–4** to generate results.")
    st.stop()

# ---- Summary (Tasks 1–3) ----
st.subheader("Summary (key KPIs) — Tasks 1–3")
summary_df = build_summary(results)
st.dataframe(summary_df, use_container_width=True)

# ---- Downloads (Tasks 1–3) ----
c1, c2 = st.columns(2)
with c1:
    st.download_button(
        "Download grouped CSV (formatted by task)",
        data=grouped_csv_text(results, days, wage),
        file_name="scheduling_grouped.csv",
        mime="text/csv",
    )
with c2:
    st.download_button(
        "Download flat CSV (easy to check)",
        data=flat_csv_text(results, days, wage),
        file_name="scheduling_flat.csv",
        mime="text/csv",
    )

st.divider()

# ---- Tabs ----
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Task 1", "Task 2 – Scenario 1", "Task 2 – Scenario 2", "Task 3", "Task 4 (AED)"]
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

    t4 = st.session_state.get("task4")

    if t4 is None:
        st.warning("Please click **Run Tasks 1–4** to generate Task 4 results.")
    else:
        st.write(t4.get("summary", ""))

        c1, c2, c3 = st.columns(3)
        c1.metric("Sample size", t4.get("n", 400))
        c2.metric("Random seed", t4.get("seed", 123))
        c3.metric("Breach rate (%)", t4.get("breach_rate_pct", "-"))

        st.divider()

        st.subheader("Numerical summaries")
        st.write("Age")
        st.dataframe(t4["age_summary"])
        st.write("Length of Stay (LoS)")
        st.dataframe(t4["los_summary"])

        st.divider()

        st.subheader("Figure 1 – Core relationships")
        st.pyplot(t4["fig1"])

        st.subheader("Figure 2 – Additional insights")
        st.pyplot(t4["fig2"])
