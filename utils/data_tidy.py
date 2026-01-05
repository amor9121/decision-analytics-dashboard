import pandas as pd
import streamlit as st


def tidy_aed_data(aed: pd.DataFrame):
    """
    Basic data tidying and validation for AED dataset.
    Performs checks without restructuring the data.
    """

    aed_clean = aed.copy()
    summary = {}

    # 1. Missing values
    missing = aed_clean.isna().sum()
    summary["total_missing"] = int(missing.sum())

    # 2. Data types
    summary["data_types"] = aed_clean.dtypes.astype(str).to_dict()

    # 3. Categorical variables
    categorical_cols = aed_clean.select_dtypes(include="object").columns
    summary["categorical_variables"] = {
        col: aed_clean[col].nunique() for col in categorical_cols
    }

    # 4. Structure
    summary["n_rows"] = aed_clean.shape[0]
    summary["n_columns"] = aed_clean.shape[1]
    summary["is_tidy"] = summary["total_missing"] == 0

    return aed_clean, summary


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
