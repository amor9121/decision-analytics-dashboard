import streamlit as st
import pandas as pd
import os

# ---- tasks ----
from core.data import days, wage
from tasks.task1 import solve_task1
from tasks.task2_s1 import solve_task2_s1
from tasks.task2_s2 import solve_task2_s2
from tasks.task3 import solve_task3
from tasks.task4 import solve_task4
from tasks.task5 import solve_task5
from tasks.task6 import solve_task6
from tasks.task8 import (
    show_patient_lookup,
    show_range_filter,
    show_modify_delete,
    show_audit_log,
    clear_audit_log,
    reset_managed_data,
)

# ---- utils ----
from utils.result_utils import build_schedule_summary, build_aed_summary
from utils.render_utils import (
    render_task_block,
    metric_row,
)
from utils.cli_utils import print_results
from utils.export_utils import (
    single_task_csv_text,
    all_tasks_zip_bytes,
    ensure_task_row,
    task_bundle_zip_bytes
)
from utils.state_utils import init_state
from contextlib import redirect_stdout

# ---- path ----
from pathlib import Path

RAW = Path("data/AED4weeks.csv")
MANAGED = Path("outputs/aed_managed.csv")

# ---- Safety check: raw must exist ----
if not RAW.exists():
    st.error(f"Raw dataset not found: {RAW}")
    st.stop()

# ---- Initialise managed data (Cloud-safe) ----
if not MANAGED.exists():
    reset_managed_data()

# ---- Now it is safe to initialise app state ----
init_state(RAW, MANAGED)


# ---- Run All ----
@st.cache_data(show_spinner=True)
def run_all_cached():
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

    # ---- Task 5 ----
    t5 = ensure_task_row(solve_task5(), "Task 5")
    t5["case"] = "Statistical analysis of breach factors"
    t5["download_df"] = t5.get("download_df", pd.DataFrame())

    # ---- Task 6 ----
    t6 = ensure_task_row(solve_task6("data/AED4weeks.csv"), "Task 6")
    figs = t6.get("figures")
    if isinstance(figs, list):
        t6["figures"] = {f"fig{i}": fig for i, fig in enumerate(figs)}
    t6["case"] = "Prediction (ML)"
    t6["download_df"] = t6.get("download_df", pd.DataFrame()) 

    # ---- Save ALL results once ----
    return [t1, t2s1, t2s2, t3, t4, t5, t6]


# ---- Streamlit UI ----
st.set_page_config(page_title="Decision Analytics Dashboard", layout="centered")
st.markdown(
    """
    <style>
    .block-container {
        max-width: 1000px;
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
st.title("📈 Decision Analytics Dashboard")

with st.expander("What this app does", expanded=True):
    st.markdown(
        """
    **### Key Decision Questions Supported**:
    - **Task 1**: Baseline cost-minimising schedule (LP).
    - **Task 2 – Scenario 1**: Minimise workload inequality **subject to ≤ 1.8% cost increase**.
    - **Task 2 – Scenario 2**: Find the **fairest** possible schedule, then minimise cost (two-stage).
    - **Task 3**: Add **skill constraints** (≥ 6 hours per skill per day) and report feasibility/cost
    - **Task 4**: Analyse a random sample of AED patient data to understand workload, patient flow, and breaches.
    - **Task 5**: Use **statistical analysis** to investigate factors contributing to breaches or prolonged stays.
    - **Task 6**: Apply **machine learning** to predict whether a patient will breach the 4-hour target.
    - **Task 7**: The Streamlit dashboard provides interactive access to all analyses from Tasks 1–8.
    - **Task 8 Extension**: Additional data management capabilities (patient lookup, range filtering, modify/delete with logging) are integrated into the same interface.
    """
    )

# Keep results in session_state (use empty list to avoid NoneType errors)
if "results" not in st.session_state:
    st.session_state["results"] = None

c1, c_spacer, c2 = st.columns([1, 4, 1])

with c1:
    run_all = st.button("Run Tasks", type="primary")

with c2:
    if st.button("Reset Results", help="Clear cached results and re-run if needed"):
        run_all_cached.clear()
        st.session_state["results"] = None
        st.session_state.pop("results_txt_written", None)
        st.rerun()


if run_all and st.session_state["results"] is None:
    st.session_state["results"] = run_all_cached()

results = st.session_state["results"]

if results is None:
    st.info("**Tips**: Click **Run Tasks** once to generate results.\n")
    st.stop()
else:
    # ---- write outputs/results.txt once per run ----
    if "results_txt_written" not in st.session_state:
        os.makedirs("outputs", exist_ok=True)
        with open("outputs/results.txt", "w", encoding="utf-8") as f:
            with redirect_stdout(f):
                print_results(results, days, wage)
        st.session_state["results_txt_written"] = True

    st.info(
        "**Tips**: Click **Reset Results** to clears cached outputs and enables re-running when needed."
    )

# ---- Summary ----

schedule_summary_df = build_schedule_summary(results)
aed_summary_df = build_aed_summary(results)

st.header("Summary")
st.caption("Overview of optimisation and analytics results for comparison.")
st.subheader("1. Scheduling Optimisation Summary (Tasks 1–3)")
st.dataframe(schedule_summary_df, use_container_width=True, hide_index=True)

st.subheader("2. AED Analytics & Prediction Summary (Tasks 4–6)")
st.dataframe(aed_summary_df, use_container_width=True, hide_index=True)


# ---- Tabs ----

st.header("Task Results")
st.caption("Select a task below to view detailed results.")
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs(
    [
        "Task 1",
        "Task 2 – Scenario 1",
        "Task 2 – Scenario 2",
        "Task 3",
        "Task 4",
        "Task 5",
        "Task 6",
        "Task 7",
        "Task 8",
    ]
)

# ---- Map by name ----

by_name = {r.get("name"): r for r in results}

with tab1:
    st.info(
        "**Task 1 – Baseline Scheduling (LP)**: minimises total labour cost subject to availability, daily coverage, and minimum weekly hours."
    )
    r = by_name.get("Task 1") or results[0]
    metric_row(r)
    render_task_block(r)

with tab2:
    st.info(
        "**Task 2 – Fairness vs Cost (Scenario 1)**: minimises fairness gap (max–min weekly hours) subject to cost ≤ baseline × 1.018."
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
        "**Task 2 – Fairness vs Cost (Scenario 2)**: finds the minimum possible fairness gap first, then minimises cost under that gap."
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
        "**Task 3 – Skill-Constrained Scheduling**: adds skill constraints: each day, Programming ≥ 6 and Troubleshooting ≥ 6 hours."
    )
    r = by_name.get("Task 3") or results[3]
    metric_row(r)
    render_task_block(r)

with tab5:
    st.info(
        "**Task 4 – AED Workload & Breach Analysis**: analyses AED patient data to understand workload, patient flow, and breaches."
    )

    r = by_name.get("Task 4")
    if r is None:
        r = results[4] if len(results) > 4 else {}

    # ---- Summary ----
    st.write(r.get("summary", ""))

    # ---- Metrics row ----
    c1, c2, c3 = st.columns(3)
    c1.metric("Sample size", r.get("n", 400))
    c2.metric("Random seed", r.get("seed", 12345))
    br = r.get("breach_rate_pct", None)
    c3.metric("Breach rate (%)", f"{br:.2f}" if isinstance(br, (int, float)) else "-")

    st.divider()

    # ---- KPI Table ----
    st.subheader("KPI table (Breach vs Non-breach)")
    if r.get("kpi_table") is not None:
        st.dataframe(r["kpi_table"], use_container_width=True)
    else:
        st.warning(
            "kpi_table not available. (Make sure solve_task4() returns 'kpi_table')"
        )

    # Optional: breach counts
    if r.get("breach_counts") is not None:
        st.markdown("**Breach counts**")
        st.dataframe(r["breach_counts"], use_container_width=True)

    st.divider()

    # ---- Numerical summaries ----
    st.subheader("Numerical summaries")
    colA, colB = st.columns(2)

    with colA:
        st.write("Age summary")
        if r.get("age_summary") is not None:
            st.dataframe(r["age_summary"], use_container_width=True)
        else:
            st.warning("age_summary not available.")

    with colB:
        st.write("Length of Stay (LoS) summary")
        if r.get("los_summary") is not None:
            st.dataframe(r["los_summary"], use_container_width=True)
        else:
            st.warning("los_summary not available.")

    st.divider()

    # ---- Figures (SHOW ALL) ----
    st.subheader("Figures (all)")
    figures = r.get("figures", {}) or {}
    if isinstance(figures, dict) and figures:
        keys = list(figures.keys())
        for i in range(0, len(keys), 2):
            left, right = st.columns(2)

            k1 = keys[i]
            with left:
                st.caption(k1)
                st.pyplot(figures[k1], use_container_width=True)

            if i + 1 < len(keys):
                k2 = keys[i + 1]
                with right:
                    st.caption(k2)
                    st.pyplot(figures[k2], use_container_width=True)
    else:
        st.warning(
            "No figures found. Make sure solve_task4() returns a 'figures' dict."
        )

    st.divider()

    # ---- Sample preview ----
    with st.expander("Sample preview (first 20 rows)", expanded=True):
        sample_df = r.get("sample", None)
        if sample_df is not None:
            st.dataframe(sample_df.head(20), use_container_width=True)
        else:
            st.write("sample not available.")

with tab6:
    st.info(
        "**Task 5 – Statistical analysis**: non-parametric tests, chi-square tests, "
        "and (if available) logistic regression."
    )

    r = by_name.get("Task 5") or {}

    # ---- Summary ----
    if r.get("summary"):
        st.write(r["summary"])

    # =========================
    # Headline statistics
    # =========================

    stats = r.get("stats") or {}

    c1, c2, c3 = st.columns(3)
    c1.metric("Sample size", stats.get("sample_n", r.get("n", 400)))
    c2.metric("Random seed", stats.get("seed", r.get("seed", 123)))
    c3.metric(
        "Breach rate (%)",
        f"{stats.get('breach_rate_pct'):.2f}"
        if isinstance(stats.get("breach_rate_pct"), (int, float))
        else "-",
    )

    stats = r.get("stats") or {}
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Prolonged threshold (min)", stats.get("prolonged_threshold", "-"))
    c2.metric("Prolonged rate (%)", stats.get("prolonged_rate_pct", "-"))
    c3.metric("4-hour target (min)", stats.get("los_target_min", "-"))
    c4.metric("Breach count", stats.get("breach_count", "-"))

    st.divider()

    # =========================
    # Table 5.2 – Numerical variables
    # =========================
    st.subheader("Numerical variables by breach status (Mann–Whitney U)")

    t52 = (r.get("tables") or {}).get("table_5_2")
    if t52 is not None:
        st.dataframe(t52, use_container_width=True)
    else:
        st.warning("Table 5.2 not available.")

    st.divider()

    # =========================
    # Table 5.3 – Categorical variables
    # =========================
    st.subheader("Categorical variables (chi-square tests)")

    t53 = (r.get("tables") or {}).get("table_5_3")
    if t53 is not None:
        st.dataframe(t53, use_container_width=True)
    else:
        st.warning("Table 5.3 not available.")

    st.divider()

    # =========================
    # Modelling diagnostics tables
    # =========================
    st.subheader("Modelling diagnostics")

    tables = r.get("tables") or {}

    for key, title in [
        ("model_df_describe", "Descriptive statistics of modelling variables"),
        ("corr_matrix_spearman", "Spearman correlation matrix"),
        ("vif_table", "Variance Inflation Factors (VIF)"),
        ("logit_m1_coef", "Logistic regression coefficients (Model m1)"),
        ("logit_m2_coef", "Logistic regression coefficients (Model m2)"),
    ]:
        if key in tables:
            with st.expander(title, expanded=True):
                st.dataframe(tables[key], use_container_width=True)

    st.divider()

    # =========================
    # Logistic regression summaries (text)
    # =========================
    st.subheader("Logistic regression model summaries")

    if r.get("logit_m1_summary"):
        with st.expander("m1.summary()", expanded=True):
            st.code(r["logit_m1_summary"])

    if r.get("logit_m2_summary"):
        with st.expander("m2.summary()", expanded=True):
            st.code(r["logit_m2_summary"])

    st.divider()

    # =========================
    # Figures
    # =========================
    st.subheader("Figures")

    figs = r.get("figures") or {}
    if isinstance(figs, dict) and figs:
        keys = list(figs.keys())
        for i in range(0, len(keys), 2):
            col1, col2 = st.columns(2)

            with col1:
                st.caption(keys[i])
                st.pyplot(figs[keys[i]], use_container_width=True)

            if i + 1 < len(keys):
                with col2:
                    st.caption(keys[i + 1])
                    st.pyplot(figs[keys[i + 1]], use_container_width=True)
    else:
        st.warning("No figures available (solve_task5() returned no 'figures').")

with tab7:

    st.info(
        "**Task 6 – Logistic Regression** with RFECV and balanced class weights. "
        "This task builds a classification model to predict AED breaches "
        "and evaluates performance using ROC-AUC, recall, and robustness checks."
    )

    r = solve_task6("data/AED4weeks.csv")

    # 3) Key metrics
    st.subheader("Key metrics")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Sample size", r.get("n", "-"))

    acc = r.get("accuracy", None)
    c2.metric("Test accuracy", f"{acc:.2%}" if isinstance(acc, (int, float)) else "-")

    roc_auc_val = r.get("roc_auc", None)
    c3.metric(
        "ROC-AUC",
        f"{roc_auc_val:.3f}" if isinstance(roc_auc_val, (int, float)) else "-",
    )

    seed = r.get("seed", 123)
    c4.metric("Random seed", seed if seed is not None else "-")

    st.divider()

    # 4) Selected features
    st.subheader("Selected features (RFECV)")
    feats = r.get("selected_features", [])
    if feats:
        st.write(feats)
    else:
        st.info("No selected features returned.")

    # 5) Classification report
    st.divider()
    st.subheader("Classification report")
    rep = r.get("classification_report", "")
    if rep:
        st.code(rep, language="text")
    else:
        st.info("No classification report returned.")

    st.divider()

    # 6) Confusion matrix (as table)
    st.subheader("Confusion matrix")
    cm = r.get("confusion_matrix", None)
    if cm is not None:
        cm_df = pd.DataFrame(
            cm,
            index=["Actual: On Time", "Actual: Breach"],
            columns=["Pred: On Time", "Pred: Breach"],
        )
        st.dataframe(cm_df, use_container_width=True)
    else:
        st.info("No confusion matrix returned.")

    # 7) Odds ratio table
    st.divider()
    st.subheader("Odds ratio table")
    or_df = r.get("odds_ratio_table", None)
    if or_df is not None:
        st.dataframe(or_df, use_container_width=True)
    else:
        st.info("No odds ratio table returned.")

    st.divider()

    # 8) Cross-validation
    st.subheader("Cross-validation")
    cv_recall = r.get("cv_recall", None)
    cv_auc = r.get("cv_auc", None)

    if cv_recall is not None:
        st.write(f"Recall (5-fold mean): {float(cv_recall.mean()):.2%}")
        st.write(f"Recall (5-fold std): {float(cv_recall.std()):.2%}")
    else:
        st.write("Recall (5-fold): -")

    if cv_auc is not None:
        st.write(f"ROC-AUC (5-fold mean): {float(cv_auc.mean()):.4f}")
        st.write(f"ROC-AUC (5-fold std): {float(cv_auc.std()):.4f}")
    else:
        st.write("ROC-AUC (5-fold): -")

    st.divider()

    # 9) Render ALL figures
    st.subheader("Figures")

    raw_figs = r.get("figures", {})
    figs = list(raw_figs.values()) if isinstance(raw_figs, dict) else raw_figs

    titles = [
        "RFECV Feature Selection Process",  # figs[0]
        "Confusion Matrix (Balanced Weights)",  # figs[1]
        "ROC Curve",  # figs[2]
        "Odds Ratio Analysis (Risk Factors)",  # figs[3]
        "5-Fold Cross-Validation Stability",  # figs[4]
        "Learning Curve (Check for Overfitting)",  # figs[5]
    ]

    if not figs:
        st.info("No figures returned.")

    else:
        # ---------- Row 1: Main performance (half width) ----------
        col1, col2 = st.columns(2)

        with col1:
            st.caption(titles[0])
            st.pyplot(figs[0], clear_figure=False, use_container_width=True)

        with col2:
            st.caption(titles[2])
            st.pyplot(figs[2], clear_figure=False, use_container_width=True)

        st.divider()

        # ---------- Row 2: Diagnostics (half width, secondary) ----------
        col1, col2 = st.columns(2)

        with col1:
            st.caption(titles[1])
            st.pyplot(figs[1], clear_figure=False, use_container_width=True)

        with col2:
            st.caption(titles[4])
            st.pyplot(figs[4], clear_figure=False, use_container_width=True)

        st.divider()

        # ---------- Row 3: Interpretation (full width) ----------
        st.caption(titles[3])
        st.pyplot(figs[3], clear_figure=False, use_container_width=True)

        st.divider()

        # ---------- Row 4: Model behaviour (full width) ----------
        st.caption(titles[5])
        st.pyplot(figs[5], clear_figure=False, use_container_width=True)

with tab8:
    st.info(
        """
        **Task 7 – Decision Analytics Dashboard**: integrates the analytical outputs from All Tasks into a single interactive
        interface, providing model transparency (algorithm selection rationale) and
        downloadable outputs for further analysis and reporting.
        """
    )
    st.markdown(
        """
        🧩 **About Task 7:**

        Task 7 focuses on integrating and operationalising the results from Tasks 1–6.  
        Downloaded outputs can be used for further analysis, reporting, and decision support outside this dashboard.
        """
    )
    path = "outputs/results.txt" 
    PREVIEW_LINES = 500

    if not os.path.exists(path):
        st.warning(f"`{path}` not found. Please generate it first (run main.py).")
    else:
        try:
            txt = open(path, "r", encoding="utf-8").read()
        except UnicodeDecodeError:
            txt = open(path, "r", encoding="utf-8", errors="replace").read()

        lines = txt.splitlines()
        preview = "\n".join(lines[:PREVIEW_LINES])

        st.subheader("All Task Results Preview")
        st.caption(
            "Scrollable preview of the complete console output (`results.txt`). "
            "This acts as an appendix-style record of all task results for transparency and reproducibility."
        )
        st.text_area(
            label="",
            value=preview,
            height=500,     
            disabled=True,  
        )
        st.caption(f"Showing {min(PREVIEW_LINES, len(lines))} of {len(lines)} lines")

        # ---- Full content (collapsed) ----
        with st.expander("Show full results.txt", expanded=False):
            st.code(txt, language="text")

        # ---- Download ----
        st.download_button(
            label="⬇️ Download results.txt",
            data=txt.encode("utf-8", errors="replace"),
            file_name="results.txt",
            mime="text/plain",
        )   
    # ---- Download ----
    st.subheader("Downloads")
    st.caption(
        """
        Download a single ZIP bundle for each task, containing all relevant outputs for reporting and review. 
        Tasks 1–3 include scheduling tables, while Tasks 4–6 include analytical outputs such as tables and figures. 
        All tasks provide summary metrics.
        """
    )
    with st.expander("Downloads", expanded=True):

        st.markdown("**Download All Results**")
        zip_all = all_tasks_zip_bytes(results, days, wage, dpi=200)
        st.download_button(
            label="📦 Download EVERYTHING (ZIP)",
            data=zip_all,
            file_name="decision_analytics_all_results.zip",
            mime="application/zip",
            key="dl_everything_zip",
            use_container_width=True,
        )

        cols = st.columns(4)
        i = 0

        for r in results:
            task_name = r.get("name", "Task")
            safe_name = (
                task_name.replace("/", "-")
                .replace("\\", "-")
                .replace(":", "-")
                .replace("|", "-")
            )

            with cols[i % 4]:
                st.markdown(f"**{task_name}**")

                # -------------------------
                # Tasks 1–3: Schedule only
                # -------------------------
                if task_name in ["Task 1", "Task 2", "Task 2 - Senerio 1", "Task 2 - Senerio 2", "Task 3"]:
                    csv_text = single_task_csv_text(r, days, wage)
                    if csv_text:
                        st.download_button(
                            label="⬇️ Schedule (CSV)",
                            data=csv_text.encode("utf-8"),
                            file_name=f"{safe_name}.csv",
                            mime="text/csv",
                            key=f"dl_csv_{safe_name}_{i}",
                            use_container_width=True,
                        )
                    else:
                        st.caption("No schedule output available.")

                # -------------------------
                # Tasks 4–6: Full ZIP only
                # -------------------------
                elif task_name in ["Task 4", "Task 5", "Task 6"]:
                    bundle_bytes = task_bundle_zip_bytes(r, days, wage, dpi=200)
                    st.download_button(
                        label="📦 All results (ZIP)",
                        data=bundle_bytes,
                        file_name=f"{safe_name}_all_results.zip",
                        mime="application/zip",
                        key=f"dl_all_{safe_name}_{i}",
                        use_container_width=True,
                    )

                else:
                    st.caption("No downloadable outputs for this task.")

            i += 1

with tab9:
    st.info(
        "**Task 8 – Data Management Extension**: "
        "extends the existing interactive decision support system by providing "
        "basic data management (CRUD) capabilities for the AED dataset. "
        "Users can retrieve, filter, modify, and remove patient records, "
        "with all actions recorded in an audit log."
    )

    # ---- Patient lookup ----
    st.subheader("Patient lookup")
    show_patient_lookup()

    st.divider()

    # ---- Range filter ----
    st.subheader("Range-based patient filter")
    show_range_filter()

    st.divider()

    # ---- Modify / Delete ----
    st.subheader("Modify or delete patient records")
    show_modify_delete()

    st.divider()

    # ---- Audit log ----
    st.subheader("Audit log")
    show_audit_log()

    st.divider()

    st.subheader("Resets")

    # ---- Reset Audit Log (toggle -> warning -> confirm) ----
    st.subheader("⚠️ Reset Audit Log")

    audit_reset_toggle = st.toggle(
        "Enable audit log reset (this will delete the audit history)",
        key="audit_reset_toggle",
    )

    if audit_reset_toggle:
        st.warning("This action cannot be undone.")

        if st.button("Confirm clear audit log", key="confirm_clear_audit_log"):
            try:
                clear_audit_log()

                # reset the on-screen dataframe too
                st.session_state["audit_view_df"] = pd.DataFrame(
                    columns=["timestamp", "action", "patient_id", "detail"]
                )

                st.success("Audit log cleared.")
                st.toast("Audit log cleared.")
                st.rerun()
            except Exception as e:
                st.error(str(e))

    st.divider()

    # ---- Reset Managed Data (toggle -> warning -> confirm) ----
    st.subheader("⚠️ Reset Data")

    reset_toggle = st.toggle(
        "Enable data reset (this will discard all modifications)",
        key="data_reset_toggle",
    )

    if reset_toggle:
        st.warning("This action cannot be undone.")

        if st.button("Confirm reset", key="confirm_reset_data"):
            try:
                reset_managed_data()
                st.success("Managed dataset has been reset to its original state.")
                st.rerun()
            except Exception as e:
                st.error(str(e))
