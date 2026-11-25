"""
Apriori Mining Module for Groceries Transaction Mining Pipeline.
Runs Apriori on Development set to generate Frequent Itemsets and Association Rules.
Calculates Support, Confidence, Lift, and Leverage metrics.
"""

import pandas as pd
from typing import Tuple, Optional
from mlxtend.frequent_patterns import apriori, association_rules


def run_apriori(
    encoded_dev_df: pd.DataFrame,
    min_support: float = 0.001
) -> pd.DataFrame:
    """
    Executes Apriori algorithm on the one-hot encoded Development set.

    Parameters:
    -----------
    encoded_dev_df : pd.DataFrame
        One-hot encoded binary DataFrame of Development transactions.
    min_support : float, default=0.001
        Minimum support threshold.

    Returns:
    --------
    freq_itemsets : pd.DataFrame
        DataFrame containing 'support' and 'itemsets'.
    """
    if encoded_dev_df.empty:
        return pd.DataFrame(columns=['support', 'itemsets'])

    freq_itemsets = apriori(
        encoded_dev_df,
        min_support=min_support,
        use_colnames=True
    )
    freq_itemsets['itemset_len'] = freq_itemsets['itemsets'].apply(len)
    return freq_itemsets.sort_values(by='support', ascending=False).reset_index(drop=True)


def generate_rules(
    freq_itemsets: pd.DataFrame,
    min_confidence: float = 0.05,
    sort_by: str = 'lift',
    ascending: bool = False
) -> pd.DataFrame:
    """
    Generates association rules X -> Y from frequent itemsets and computes rule metrics.

    Parameters:
    -----------
    freq_itemsets : pd.DataFrame
        DataFrame of frequent itemsets from run_apriori.
    min_confidence : float, default=0.05
        Minimum confidence threshold.
    sort_by : str, default='lift'
        Metric column name to sort the generated rules by.
    ascending : bool, default=False
        Sort order.

    Returns:
    --------
    rules_df : pd.DataFrame
        DataFrame containing antecedents, consequents, support, confidence, lift, leverage.
    """
    if freq_itemsets.empty or len(freq_itemsets) == 0:
        return pd.DataFrame(columns=[
            'antecedents', 'consequents', 'support', 'confidence', 'lift', 'leverage'
        ])

    try:
        rules = association_rules(
            freq_itemsets,
            metric='confidence',
            min_threshold=min_confidence
        )
    except ValueError:
        # Returns empty if no rules pass min_confidence
        return pd.DataFrame(columns=[
            'antecedents', 'consequents', 'support', 'confidence', 'lift', 'leverage'
        ])

    if rules.empty:
        return pd.DataFrame(columns=[
            'antecedents', 'consequents', 'support', 'confidence', 'lift', 'leverage'
        ])

    # Ensure leverage column exists
    if 'leverage' not in rules.columns:
        # Calculate leverage: support(X u Y) - support(X) * support(Y)
        rules['leverage'] = rules['support'] - (rules['antecedent support'] * rules['consequent support'])

    # Format antecedents and consequents as clean strings / frozensets
    rules = rules.copy()
    if sort_by in rules.columns:
        rules = rules.sort_values(by=sort_by, ascending=ascending).reset_index(drop=True)

    return rules
