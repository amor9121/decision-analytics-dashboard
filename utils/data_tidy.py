import pandas as pd


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
