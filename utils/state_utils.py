import pandas as pd
import streamlit as st
from pathlib import Path


@st.cache_data
def load_df(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def init_state(raw_path: Path, managed_path: Path):
    # Pick managed data if it exists, else raw
    chosen = managed_path if managed_path.exists() else raw_path

    if "df" not in st.session_state:
        df = load_df(chosen)

        # Ensure soft-delete column exists
        if "is_deleted" not in df.columns:
            df["is_deleted"] = False

        st.session_state["df"] = df

    if "raw_path" not in st.session_state:
        st.session_state["raw_path"] = str(raw_path)

    if "managed_path" not in st.session_state:
        st.session_state["managed_path"] = str(managed_path)
