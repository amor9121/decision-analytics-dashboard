import os
import pandas as pd
import streamlit as st

from utils.mgmt_utils import (
    get_id_col,
    ensure_soft_delete_col,
    active_df,
    save_managed_df,
    log_action,
)

RAW_PATH = "data/AED4weeks.csv"
MANAGED_PATH = "outputs/aed_managed.csv"


def _assert_not_raw_path(path: str):
    if "data/" in path:
        raise RuntimeError("❌ Attempt to write to raw data is not allowed.")


def show_patient_lookup():

    df = st.session_state["df"]
    df = ensure_soft_delete_col(df)
    st.session_state["df"] = df

    id_col = get_id_col(df)

    include_deleted = st.checkbox(
        "Include deleted records", value=False, key="t8_lookup_incdel"
    )
    dfx = active_df(df, include_deleted)

    pid = st.text_input(
        f"Enter patient ID ({id_col})", placeholder="e.g. P10010", key="t8_lookup_pid"
    )

    if st.button("Search", type="primary", key="t8_lookup_btn"):
        result = dfx[dfx[id_col].astype(str) == str(pid)]

        log_action(
            "LOOKUP",
            patient_id=pid,
            detail=f"id_col={id_col}, found={not result.empty}, include_deleted={include_deleted}",
        )

        if result.empty:
            st.warning("Patient not found.")
        else:
            st.success("Patient record found.")
            st.dataframe(result, use_container_width=True)


def show_range_filter():

    df = st.session_state["df"]
    df = ensure_soft_delete_col(df)
    st.session_state["df"] = df

    include_deleted = st.checkbox(
        "Include deleted records", value=False, key="t8_range_incdel"
    )
    dfx = active_df(df, include_deleted)

    numeric_cols = [
        c
        for c in dfx.columns
        if pd.api.types.is_numeric_dtype(dfx[c]) and c != "is_deleted"
    ]
    if not numeric_cols:
        st.info("No numeric variables available.")
        return

    col = st.selectbox("Select a variable", numeric_cols, key="t8_range_col")

    min_val = float(dfx[col].min())
    max_val = float(dfx[col].max())

    vmin = st.number_input("Minimum value", value=min_val, key="t8_range_min")
    vmax = st.number_input("Maximum value", value=max_val, key="t8_range_max")

    if st.button("Apply filter", type="primary", key="t8_range_btn"):
        if vmin > vmax:
            st.error("Minimum value must be smaller than maximum value.")
            return

        filtered = dfx[dfx[col].between(vmin, vmax)]

        log_action(
            "RANGE_FILTER",
            detail=f"{col}: {vmin}–{vmax}, rows={len(filtered)}, include_deleted={include_deleted}",
        )

        st.write(f"Matched records: **{len(filtered)}**")
        st.dataframe(filtered, use_container_width=True)


def show_modify_delete():
    _assert_not_raw_path(MANAGED_PATH)

    if not os.path.exists(MANAGED_PATH):
        st.error(
            "Managed dataset not found. "
            "Please reset the managed data before modifying records."
        )
        return

    df = pd.read_csv(MANAGED_PATH)
    df = ensure_soft_delete_col(df)
    st.session_state["df"] = df

    id_col = get_id_col(df)

    include_deleted = st.checkbox(
        "Include deleted records", value=False, key="t8_mod_incdel"
    )
    dfx = active_df(df, include_deleted)

    pid = st.text_input(f"Patient ID to modify or delete ({id_col})", key="t8_mod_pid")

    record = dfx[dfx[id_col].astype(str) == str(pid)]
    if record.empty:
        st.info("Enter a valid patient ID to proceed.")
        return

    st.dataframe(record, use_container_width=True)

    st.markdown("#### Modify record")

    editable_cols = [c for c in df.columns if c not in [id_col, "is_deleted"]]
    if not editable_cols:
        st.info("No editable fields available.")
        return

    col = st.selectbox("Field to modify", editable_cols, key="t8_mod_col")
    new_value = st.text_input("New value", key="t8_mod_new")

    if st.button("Update", type="primary", key="t8_mod_update"):
        idx = record.index[0]
        old_value = df.loc[idx, col]

        if pd.api.types.is_numeric_dtype(df[col]):
            try:
                new_value_casted = float(new_value)
            except ValueError:
                st.error("Invalid numeric value.")
                return
        else:
            new_value_casted = new_value

        df.loc[idx, col] = new_value_casted

        save_managed_df(df)
        st.session_state["df"] = df

        log_action(
            "UPDATE",
            patient_id=pid,
            detail=f"id_col={id_col}, {col}: {old_value} → {new_value_casted}",
        )
        st.success("Record updated successfully.")

    st.markdown("#### Delete record")

    confirm = st.checkbox(
        "I confirm that I want to delete this record", key="t8_del_confirm"
    )

    if st.button("Delete", disabled=not confirm, key="t8_del_btn"):
        idx = record.index[0]
        df.loc[idx, "is_deleted"] = True

        save_managed_df(df)
        st.session_state["df"] = df

        log_action(
            "DELETE",
            patient_id=pid,
            detail=f"id_col={id_col}, soft delete",
        )
        st.success("Record deleted.")
        st.rerun()


def show_audit_log():

    log_path = st.session_state.get("log_path", "outputs/audit_log.csv")

    if not os.path.exists(log_path):
        st.info("No audit log available yet.")
        return

    log_df = pd.read_csv(log_path)
    st.dataframe(log_df.tail(50), use_container_width=True)

    st.download_button(
        label="Download audit log (CSV)",
        data=log_df.to_csv(index=False),
        file_name="audit_log.csv",
        mime="text/csv",
    )


def clear_audit_log():
    os.makedirs(os.path.dirname("outputs/audit_log.csv"), exist_ok=True)
    pd.DataFrame(columns=["timestamp", "action", "patient_id", "detail"]).to_csv(
        "outputs/audit_log.csv", index=False
    )


def reset_managed_data():
    """
    Reset the managed AED dataset to its original baseline state.
    This discards all modifications and restores the raw dataset.
    """
    if not os.path.exists(RAW_PATH):
        raise FileNotFoundError("Raw AED dataset not found.")

    os.makedirs(os.path.dirname(MANAGED_PATH), exist_ok=True)

    df_raw = pd.read_csv(RAW_PATH)

    # Optional: ensure soft-delete column exists
    if "is_deleted" not in df_raw.columns:
        df_raw["is_deleted"] = False

    df_raw.to_csv(MANAGED_PATH, index=False)
