import numpy as np
import pandas as pd

ID_COLS = ["Prospect ID", "Lead Number"]

HIGH_NULL_THRESHOLD = 0.40
LOW_VARIANCE_THRESHOLD = 0.95


def clean_raw_leads(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df = df.replace("Select", np.nan)

    df = df.drop(columns=[c for c in ID_COLS if c in df.columns], errors="ignore")

    missing_frac = df.isnull().mean()
    high_null_cols = missing_frac[missing_frac > HIGH_NULL_THRESHOLD].index.tolist()
    df = df.drop(columns=high_null_cols)

    low_variance_cols = []
    for col in df.select_dtypes(include=["object"]).columns:
        top_freq = df[col].value_counts(normalize=True, dropna=False).iloc[0]
        if top_freq > LOW_VARIANCE_THRESHOLD:
            low_variance_cols.append(col)
    df = df.drop(columns=low_variance_cols)

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if "Converted" in num_cols:
        num_cols.remove("Converted")
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()

    for col in num_cols:
        df[col] = df[col].fillna(df[col].median())
    for col in cat_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

    return df


def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode categorical columns."""
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    return pd.get_dummies(df, columns=cat_cols, drop_first=True)


def align_columns(df_encoded: pd.DataFrame, expected_columns: list) -> pd.DataFrame:
    return df_encoded.reindex(columns=expected_columns, fill_value=0)
