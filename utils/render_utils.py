import pandas as pd
import streamlit as st
from utils.schedule_utils import check_daily_coverage, build_final_table
from utils.export_utils import single_task_csv_text
from utils.figure_utils import collect_task_figures, figures_zip_bytes
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
            task_name = r.get("name", "Task")
            safe_name = task_name.replace("/", "-")

            with cols[i % n_cols]:
                # ---- CSV download (existing) ----
                csv_text = single_task_csv_text(r, days, wage)
                if csv_text:
                    st.download_button(
                        label=f"Download {task_name} (CSV)",
                        data=csv_text.encode("utf-8"),
                        file_name=f"{safe_name}.csv",
                        mime="text/csv",
                        key=f"dl_csv_{safe_name}_{i}",
                    )

                # ---- Figures download (new) ----
                figs = collect_task_figures(r)
                if figs:
                    zip_bytes = figures_zip_bytes(figs, dpi=200)
                    st.download_button(
                        label=f"Download {task_name} Figures (ZIP)",
                        data=zip_bytes,
                        file_name=f"{safe_name}_figures.zip",
                        mime="application/zip",
                        key=f"dl_fig_{safe_name}_{i}",
                    )

            i += 1


def show_tidy_summary_expander(
    tidy_summary: dict,
    title: str = "Data tidying checks",
    expanded: bool = False,
):
    """
    Render data tidying checks in a Streamlit expander.
    """

    if tidy_summary is None:
        st.info("tidy_summary not available.")
        return

    with st.expander(title, expanded=expanded):

        st.caption(
            "Initial checks confirm that the dataset is already in a tidy format. "
            "No further data transformation was required."
        )

        c1, c2, c3 = st.columns(3)
        c1.metric("Rows", tidy_summary.get("n_rows", "-"))
        c2.metric("Columns", tidy_summary.get("n_columns", "-"))
        c3.metric("Total missing", tidy_summary.get("total_missing", "-"))

        st.divider()

        # Data types
        dtypes = tidy_summary.get("data_types", {})
        if dtypes:
            st.subheader("Variable types")
            st.dataframe(
                pd.DataFrame({"Variable": dtypes.keys(), "Type": dtypes.values()}),
                use_container_width=True,
                hide_index=True,
            )

        # Categorical variables
        cats = tidy_summary.get("categorical_variables", {})
        if cats:
            st.subheader("Categorical variables")
            st.dataframe(
                pd.DataFrame(
                    {
                        "Variable": cats.keys(),
                        "Number of unique levels": cats.values(),
                    }
                ),
                use_container_width=True,
                hide_index=True,
            )
