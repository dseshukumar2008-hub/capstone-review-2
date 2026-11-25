"""
Rule Quality Evaluation Module.
Ranks association rules by multiple metrics, filters by user thresholds,
and identifies redundant rules using a clear, explicit definition.

Redundancy Definition (used here):
    Rule R1 (A → C) is redundant with respect to R2 (B → C) if:
      - B ⊂ A  (B is a proper subset of A — a more general antecedent)
      - conf(R2) >= conf(R1)  (the more general rule is at least as confident)
    In that case R1 adds no information beyond R2 and is marked redundant.

This is the standard "non-redundant rule" definition used in data mining literature.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple


def rank_rules(rules_df: pd.DataFrame, sort_by: str = 'lift') -> pd.DataFrame:
    """
    Returns rules sorted by the specified metric with a rank column added.

    Parameters
    ----------
    rules_df : pd.DataFrame
        Association rules DataFrame (from generate_rules or generate_fpgrowth_rules).
    sort_by : str
        Column name to rank by. Must be one of: support, confidence, lift, leverage.

    Returns
    -------
    ranked_df : pd.DataFrame
        Rules sorted descending by sort_by, with a 'rank' column.
    """
    if rules_df.empty:
        return rules_df.copy()

    valid_metrics = ['support', 'confidence', 'lift', 'leverage']
    if sort_by not in valid_metrics:
        sort_by = 'lift'

    ranked = rules_df.sort_values(by=sort_by, ascending=False).reset_index(drop=True)
    ranked.insert(0, 'rank', range(1, len(ranked) + 1))
    return ranked


def filter_rules(
    rules_df: pd.DataFrame,
    min_support: float = 0.0,
    min_confidence: float = 0.0,
    min_lift: float = 1.0,
    min_leverage: float = 0.0
) -> pd.DataFrame:
    """
    Filters rules by user-specified metric thresholds.

    Parameters
    ----------
    rules_df : pd.DataFrame
        Association rules DataFrame.
    min_support : float
        Minimum support threshold.
    min_confidence : float
        Minimum confidence threshold.
    min_lift : float
        Minimum lift threshold (default 1.0 = no negative association).
    min_leverage : float
        Minimum leverage threshold.

    Returns
    -------
    filtered_df : pd.DataFrame
    """
    if rules_df.empty:
        return rules_df.copy()

    mask = (
        (rules_df['support'] >= min_support) &
        (rules_df['confidence'] >= min_confidence) &
        (rules_df['lift'] >= min_lift) &
        (rules_df['leverage'] >= min_leverage)
    )
    return rules_df[mask].reset_index(drop=True)


def identify_redundant_rules(rules_df: pd.DataFrame) -> pd.DataFrame:
    """
    Identifies redundant rules and annotates the DataFrame.

    Redundancy criterion:
        Rule R1: A → C is redundant if there exists R2: B → C where
        B ⊂ A (strict subset) and conf(R2) >= conf(R1).

    Parameters
    ----------
    rules_df : pd.DataFrame
        Association rules DataFrame with 'antecedents', 'consequents', 'confidence'.

    Returns
    -------
    annotated_df : pd.DataFrame
        Same DataFrame with two new columns:
        - 'is_redundant' (bool)
        - 'redundant_reason' (str or '')
    """
    if rules_df.empty:
        result = rules_df.copy()
        result['is_redundant'] = False
        result['redundant_reason'] = ''
        return result

    df = rules_df.copy().reset_index(drop=True)
    df['is_redundant'] = False
    df['redundant_reason'] = ''

    n = len(df)
    for i in range(n):
        if df.at[i, 'is_redundant']:
            continue  # already marked
        ant_i = df.at[i, 'antecedents']   # frozenset
        con_i = df.at[i, 'consequents']   # frozenset
        conf_i = df.at[i, 'confidence']

        for j in range(n):
            if i == j:
                continue
            ant_j = df.at[j, 'antecedents']
            con_j = df.at[j, 'consequents']
            conf_j = df.at[j, 'confidence']

            # R_i is redundant if there exists R_j with:
            #   same consequent, ant_j ⊂ ant_i, conf_j >= conf_i
            if (con_i == con_j) and ant_j < ant_i and conf_j >= conf_i:
                df.at[i, 'is_redundant'] = True
                ant_j_str = ", ".join(sorted(ant_j))
                ant_i_str = ", ".join(sorted(ant_i))
                df.at[i, 'redundant_reason'] = (
                    f"Rule '{ant_i_str} → ...' is dominated by "
                    f"'{ant_j_str} → ...' (subset antecedent with conf "
                    f"{conf_j:.3f} ≥ {conf_i:.3f})"
                )
                break

    return df


def get_quality_summary(
    rules_df: pd.DataFrame,
    min_support: float = 0.0,
    min_confidence: float = 0.0,
    min_lift: float = 1.0,
    min_leverage: float = 0.0
) -> Dict[str, Any]:
    """
    Returns a summary dict of rule quality statistics for the given thresholds.

    Parameters
    ----------
    rules_df : pd.DataFrame
        Full association rules DataFrame.
    min_support, min_confidence, min_lift, min_leverage : float
        Filter thresholds.

    Returns
    -------
    summary : dict
        total_rules, after_filter, redundant_rules, non_redundant_rules,
        pct_redundant, avg_lift, avg_confidence, avg_support
    """
    if rules_df.empty:
        return {
            "total_rules": 0, "after_filter": 0,
            "redundant_rules": 0, "non_redundant_rules": 0,
            "pct_redundant": 0.0, "avg_lift": 0.0,
            "avg_confidence": 0.0, "avg_support": 0.0
        }

    total = len(rules_df)
    filtered = filter_rules(rules_df, min_support, min_confidence, min_lift, min_leverage)
    annotated = identify_redundant_rules(filtered)

    n_redundant = int(annotated['is_redundant'].sum())
    n_after = len(filtered)

    return {
        "total_rules": total,
        "after_filter": n_after,
        "redundant_rules": n_redundant,
        "non_redundant_rules": n_after - n_redundant,
        "pct_redundant": round(100.0 * n_redundant / n_after, 1) if n_after > 0 else 0.0,
        "avg_lift": round(float(filtered['lift'].mean()), 4) if not filtered.empty else 0.0,
        "avg_confidence": round(float(filtered['confidence'].mean()), 4) if not filtered.empty else 0.0,
        "avg_support": round(float(filtered['support'].mean()), 4) if not filtered.empty else 0.0,
    }
