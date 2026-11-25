"""
Validation Module for Groceries Transaction Mining Pipeline.
Evaluates rules discovered on the Development set against the unseen Validation set.
Calculates Validation Support, Confidence, Lift, and Leverage side-by-side with Development metrics.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List


def evaluate_single_rule_on_validation(
    antecedent: set,
    consequent: set,
    val_encoded_df: pd.DataFrame
) -> Dict[str, float]:
    """
    Evaluates a single association rule X -> Y on the validation transaction matrix.

    Parameters:
    -----------
    antecedent : set or frozenset
        Itemset X.
    consequent : set or frozenset
        Itemset Y.
    val_encoded_df : pd.DataFrame
        One-hot encoded boolean DataFrame of Validation transactions.

    Returns:
    --------
    metrics : dict
        val_support, val_confidence, val_lift, val_leverage, val_antecedent_support, val_consequent_support
    """
    n_val = len(val_encoded_df)
    if n_val == 0:
        return {
            "val_support": 0.0,
            "val_confidence": 0.0,
            "val_lift": 0.0,
            "val_leverage": 0.0,
            "val_antecedent_support": 0.0,
            "val_consequent_support": 0.0
        }

    ant_items = list(antecedent)
    con_items = list(consequent)
    union_items = list(set(antecedent) | set(consequent))

    # Check if all items exist in validation dataset columns
    cols = set(val_encoded_df.columns)

    if not set(ant_items).issubset(cols):
        ant_supp = 0.0
    else:
        ant_supp = float(val_encoded_df[ant_items].all(axis=1).mean())

    if not set(con_items).issubset(cols):
        con_supp = 0.0
    else:
        con_supp = float(val_encoded_df[con_items].all(axis=1).mean())

    if not set(union_items).issubset(cols):
        rule_supp = 0.0
    else:
        rule_supp = float(val_encoded_df[union_items].all(axis=1).mean())

    # Calculate confidence, lift, leverage
    val_conf = (rule_supp / ant_supp) if ant_supp > 0 else 0.0
    val_lift = (val_conf / con_supp) if con_supp > 0 else 0.0
    val_leverage = rule_supp - (ant_supp * con_supp)

    return {
        "val_support": rule_supp,
        "val_confidence": val_conf,
        "val_lift": val_lift,
        "val_leverage": val_leverage,
        "val_antecedent_support": ant_supp,
        "val_consequent_support": con_supp
    }


def evaluate_rules_on_validation(
    rules_df: pd.DataFrame,
    val_encoded_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Evaluates a set of discovered association rules on the validation dataset.
    Returns a combined DataFrame with Development vs Validation metrics.

    Parameters:
    -----------
    rules_df : pd.DataFrame
        DataFrame of rules generated from the Development set.
    val_encoded_df : pd.DataFrame
        One-hot encoded boolean DataFrame of Validation transactions.

    Returns:
    --------
    comparison_df : pd.DataFrame
        Combined comparison of Development and Validation metrics for each rule.
    """
    if rules_df.empty:
        return pd.DataFrame()

    results = []
    for _, row in rules_df.iterrows():
        ant = row['antecedents']
        con = row['consequents']

        val_metrics = evaluate_single_rule_on_validation(ant, con, val_encoded_df)

        dev_supp = float(row['support'])
        dev_conf = float(row['confidence'])
        dev_lift = float(row['lift'])
        dev_lev = float(row.get('leverage', dev_supp - (row.get('antecedent support', 0) * row.get('consequent support', 0))))

        val_supp = val_metrics['val_support']
        val_conf = val_metrics['val_confidence']
        val_lift = val_metrics['val_lift']
        val_lev = val_metrics['val_leverage']

        results.append({
            "antecedents": str(set(ant)),
            "consequents": str(set(con)),
            "dev_support": dev_supp,
            "val_support": val_supp,
            "diff_support": val_supp - dev_supp,
            "dev_confidence": dev_conf,
            "val_confidence": val_conf,
            "diff_confidence": val_conf - dev_conf,
            "dev_lift": dev_lift,
            "val_lift": val_lift,
            "diff_lift": val_lift - dev_lift,
            "dev_leverage": dev_lev,
            "val_leverage": val_lev,
            "diff_leverage": val_lev - dev_lev,
        })

    comparison_df = pd.DataFrame(results)
    return comparison_df
