"""
One-Hot Encoding Module for Groceries Transaction Mining Pipeline.
Converts transaction baskets into a binary (bool/int) transaction-item matrix.
"""

import pandas as pd
from typing import List, Union, Tuple
from mlxtend.preprocessing import TransactionEncoder


def encode_transactions(
    baskets: Union[pd.DataFrame, List[List[str]]],
    encoder: TransactionEncoder = None
) -> Tuple[pd.DataFrame, TransactionEncoder]:
    """
    Encodes transaction baskets into a binary boolean DataFrame compatible with mlxtend.apriori.

    Parameters:
    -----------
    baskets : pd.DataFrame (with 'items' column) or List[List[str]]
        The list of transaction item baskets.
    encoder : TransactionEncoder, optional
        Pre-fitted TransactionEncoder instance (e.g. from Development Set).

    Returns:
    --------
    encoded_df : pd.DataFrame
        Boolean DataFrame where rows are transactions and columns are items.
    encoder : TransactionEncoder
        Fitted TransactionEncoder instance.
    """
    if isinstance(baskets, pd.DataFrame):
        if 'items' in baskets.columns:
            basket_lists = baskets['items'].tolist()
        else:
            raise ValueError("DataFrame must contain an 'items' column.")
    else:
        basket_lists = baskets

    if encoder is None:
        encoder = TransactionEncoder()
        array = encoder.fit_transform(basket_lists)
    else:
        array = encoder.transform(basket_lists)

    encoded_df = pd.DataFrame(array, columns=encoder.columns_)
    return encoded_df, encoder
