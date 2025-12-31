import pandas as pd
import streamlit as st
from .schedule_utils import check_daily_coverage, build_final_table
from .export_utils import single_task_csv_text
from core.data import days, wage


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
