import os
from datetime import datetime
import pandas as pd
import streamlit as st

# Paths
DEFAULT_MANAGED_PATH = "outputs/aed_managed.csv"
DEFAULT_LOG_PATH = "outputs/audit_log.csv"


# ID column handling
def detect_id_col(df: pd.DataFrame) -> str | None:
    """
    Try to detect a patient/attendance ID column automatically.
    """
    candidates = [
        "patient_id",
        "patient id",
        "patientid",
        "attendance_id",
        "attendance id",
        "attendanceid",
        "attend_id",
        "attendid",
        "case_id",
        "case id",
        "id",
    ]

    col_map = {c.strip().lower(): c for c in df.columns}

    for key in candidates:
        if key in col_map:
            return col_map[key]

    # fallback: any column containing 'id'
    for lc, orig in col_map.items():
        if "id" in lc:
            return orig

    return None


def get_id_col(df: pd.DataFrame) -> str:
    """
    Return the ID column name.
    - Use cached value if exists
    - Else auto-detect
    - Else ask user once
    """
    if (
        "mgmt_id_col" in st.session_state
        and st.session_state["mgmt_id_col"] in df.columns
    ):
        return st.session_state["mgmt_id_col"]

    auto = detect_id_col(df)
    if auto is not None:
        st.session_state["mgmt_id_col"] = auto
        return auto

    st.warning("Could not auto-detect the patient ID column. Please select it.")
    chosen = st.selectbox(
        "Select ID column",
        list(df.columns),
        key="mgmt_id_col_select",
    )
    st.session_state["mgmt_id_col"] = chosen
    return chosen


# Data persistence
def ensure_soft_delete_col(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure the DataFrame has an is_deleted column.
    """
    if "is_deleted" not in df.columns:
        df["is_deleted"] = False
    return df


def active_df(df: pd.DataFrame, include_deleted: bool) -> pd.DataFrame:
    """
    Return active records unless include_deleted=True.
    """
    if include_deleted:
        return df
    if "is_deleted" not in df.columns:
        return df
    return df[df["is_deleted"] == False]  # noqa: E712


def save_managed_df(df: pd.DataFrame):
    """
    Save managed DataFrame to disk and keep path consistent.
    """
    path = st.session_state.get("managed_path", DEFAULT_MANAGED_PATH)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)


# Audit logging
def log_action(
    action: str,
    patient_id=None,
    detail: str = "",
):
    """
    Append a single audit log entry.
    """
    log_path = st.session_state.get("log_path", DEFAULT_LOG_PATH)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    row = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "action": action,
        "patient_id": "" if patient_id is None else patient_id,
        "detail": detail,
    }

    df_log = pd.DataFrame([row])

    if os.path.exists(log_path):
        df_log.to_csv(log_path, mode="a", header=False, index=False)
    else:
        df_log.to_csv(log_path, index=False)
