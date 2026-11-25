"""
Dataset Split Module for Groceries Transaction Mining Pipeline.
Splits transaction baskets at the TRANSACTION level into Development (80%)
and Validation (20%) sets using a reproducible random seed.
"""

import pandas as pd
from typing import Tuple
from sklearn.model_selection import train_test_split


def split_transactions(
    baskets_df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits the transaction-level dataset into Development (mining) and Validation sets.

    Parameters:
    -----------
    baskets_df : pd.DataFrame
        DataFrame containing transaction baskets (e.g. 'Transaction_ID', 'items').
    test_size : float, default=0.2
        Fraction of transactions to allocate to the validation set.
    random_state : int, default=42
        Reproducible random seed for shuffling and splitting.

    Returns:
    --------
    dev_baskets : pd.DataFrame
        80% Development Set for pattern discovery.
    val_baskets : pd.DataFrame
        20% Validation Set for pattern evaluation on unseen data.
    """
    if 'Transaction_ID' not in baskets_df.columns or 'items' not in baskets_df.columns:
        raise ValueError("baskets_df must contain 'Transaction_ID' and 'items' columns.")

    dev_baskets, val_baskets = train_test_split(
        baskets_df,
        test_size=test_size,
        random_state=random_state,
        shuffle=True
    )

    # Reset indices cleanly
    dev_baskets = dev_baskets.reset_index(drop=True)
    val_baskets = val_baskets.reset_index(drop=True)

    return dev_baskets, val_baskets
