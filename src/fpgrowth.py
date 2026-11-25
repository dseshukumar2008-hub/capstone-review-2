"""
FP-Growth Mining Module for Groceries Transaction Mining Pipeline.
Implements FP-Growth using mlxtend, reusing the same encoded transaction matrix
as Apriori for a fair algorithm comparison.
"""

import pandas as pd
from typing import Tuple
from mlxtend.frequent_patterns import fpgrowth, association_rules


def run_fpgrowth(
    encoded_df: pd.DataFrame,
    min_support: float = 0.001
) -> pd.DataFrame:
    """
    Executes FP-Growth algorithm on the one-hot encoded Development set.

    Parameters
    ----------
    encoded_df : pd.DataFrame
        One-hot encoded boolean DataFrame (same matrix used by Apriori).
    min_support : float, default=0.001
        Minimum support threshold (same scale as Apriori).

    Returns
    -------
    freq_itemsets : pd.DataFrame
        DataFrame with columns 'support', 'itemsets', 'itemset_len'.
    """
    if encoded_df.empty:
        return pd.DataFrame(columns=['support', 'itemsets', 'itemset_len'])

    freq_itemsets = fpgrowth(
        encoded_df,
        min_support=min_support,
        use_colnames=True
    )
    freq_itemsets['itemset_len'] = freq_itemsets['itemsets'].apply(len)
    return freq_itemsets.sort_values(by='support', ascending=False).reset_index(drop=True)


def generate_fpgrowth_rules(
    freq_itemsets: pd.DataFrame,
    min_confidence: float = 0.05,
    sort_by: str = 'lift',
    ascending: bool = False
) -> pd.DataFrame:
    """
    Generates association rules from FP-Growth frequent itemsets.
    Identical API to mining.generate_rules() for drop-in comparison.

    Parameters
    ----------
    freq_itemsets : pd.DataFrame
        Frequent itemsets from run_fpgrowth().
    min_confidence : float, default=0.05
        Minimum confidence threshold.
    sort_by : str, default='lift'
        Metric to sort rules by.
    ascending : bool, default=False
        Sort order.

    Returns
    -------
    rules_df : pd.DataFrame
        DataFrame with antecedents, consequents, support, confidence, lift, leverage.
    """
    _empty = pd.DataFrame(columns=[
        'antecedents', 'consequents', 'support', 'confidence', 'lift', 'leverage'
    ])

    if freq_itemsets.empty or len(freq_itemsets) == 0:
        return _empty

    try:
        rules = association_rules(
            freq_itemsets,
            metric='confidence',
            min_threshold=min_confidence
        )
    except ValueError:
        return _empty

    if rules.empty:
        return _empty

    if 'leverage' not in rules.columns:
        rules['leverage'] = rules['support'] - (
            rules['antecedent support'] * rules['consequent support']
        )

    rules = rules.copy()
    if sort_by in rules.columns:
        rules = rules.sort_values(by=sort_by, ascending=ascending).reset_index(drop=True)

    return rules
