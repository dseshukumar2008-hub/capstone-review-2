"""
Data Preprocessing Module for Groceries Transaction Mining Pipeline.
Handles loading, verification, duplicate removal, missing value handling,
Transaction ID creation, and basket aggregation.
"""

import pandas as pd
from typing import Dict, Any, Tuple, List

REQUIRED_COLUMNS = ['Member_number', 'Date', 'itemDescription']


def load_dataset(filepath: str) -> pd.DataFrame:
    """
    Loads CSV dataset and verifies presence of required columns.
    """
    df = pd.read_csv(filepath)
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Dataset missing required columns: {missing_cols}")
    return df


def verify_columns(df: pd.DataFrame) -> bool:
    """
    Checks whether required columns are present in the DataFrame.
    """
    return all(col in df.columns for col in REQUIRED_COLUMNS)


def get_dataset_statistics(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculates summary statistics of the raw CSV dataset programmatically.
    """
    return {
        "raw_rows": int(len(df)),
        "raw_columns": int(len(df.columns)),
        "missing_values": {col: int(df[col].isnull().sum()) for col in df.columns},
        "total_missing_values": int(df.isnull().sum().sum()),
        "exact_duplicate_rows": int(df.duplicated().sum()),
        "unique_customers": int(df['Member_number'].nunique()) if 'Member_number' in df.columns else 0,
        "unique_products": int(df['itemDescription'].nunique()) if 'itemDescription' in df.columns else 0,
        "unique_dates": int(df['Date'].nunique()) if 'Date' in df.columns else 0,
    }


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Removes exact duplicate rows and handles missing values.
    """
    # Verify required columns first
    if not verify_columns(df):
        raise ValueError("DataFrame lacks required columns for cleaning.")

    # Drop exact duplicates
    df_clean = df.drop_duplicates().copy()

    # Drop any missing values if present
    df_clean = df_clean.dropna(subset=REQUIRED_COLUMNS).copy()

    # Clean string columns
    df_clean['itemDescription'] = df_clean['itemDescription'].astype(str).str.strip()
    df_clean['Date'] = df_clean['Date'].astype(str).str.strip()
    df_clean['Member_number'] = df_clean['Member_number'].astype(str).str.strip()

    return df_clean


def create_transaction_id(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates Transaction_ID = Member_number + '_' + Date
    """
    df_result = df.copy()
    df_result['Transaction_ID'] = (
        df_result['Member_number'].astype(str) + '_' + df_result['Date'].astype(str)
    )
    return df_result


def create_baskets(df: pd.DataFrame) -> pd.DataFrame:
    """
    Groups items by Transaction_ID (Member_number + Date), removes repeated
    occurrences of the same item within the same basket, and produces a
    clean transaction-level representation.
    """
    if 'Transaction_ID' not in df.columns:
        df = create_transaction_id(df)

    # Group items into sets to eliminate repeated items in the same basket
    baskets = (
        df.groupby('Transaction_ID')['itemDescription']
        .apply(lambda items: sorted(list(set(items))))
        .reset_index()
    )
    baskets.rename(columns={'itemDescription': 'items'}, inplace=True)
    return baskets


def get_transaction_statistics(baskets_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculates summary statistics of the clean transaction-level dataset.
    """
    basket_sizes = baskets_df['items'].apply(len)
    all_items = [item for sublist in baskets_df['items'] for item in sublist]
    unique_items = set(all_items)

    return {
        "num_transactions": int(len(baskets_df)),
        "num_unique_items": int(len(unique_items)),
        "avg_basket_size": float(basket_sizes.mean()) if len(basket_sizes) > 0 else 0.0,
        "min_basket_size": int(basket_sizes.min()) if len(basket_sizes) > 0 else 0,
        "max_basket_size": int(basket_sizes.max()) if len(basket_sizes) > 0 else 0,
        "total_item_purchases": int(basket_sizes.sum()),
    }
