"""
Adaptive Parameter Recommendation Module.
Analyzes dataset characteristics and recommends explainable
minimum support and confidence thresholds for association rule mining.

Recommendations are rule-based and fully transparent — no trained model is used.
"""

import pandas as pd
from typing import Dict, Any


def recommend_parameters(
    baskets_df: pd.DataFrame,
    dev_fraction: float = 0.8
) -> Dict[str, Any]:
    """
    Recommends minimum support and confidence values based on dataset statistics.

    Logic:
    ------
    - Very sparse datasets (avg basket size < 2) → higher support needed
    - Large datasets (> 5000 transactions) → lower support is feasible
    - Density (avg_items / total_items) drives the support tier
    - Confidence is set conservatively (0.1–0.3) to capture meaningful rules
      without over-filtering in sparse retail data.

    Parameters
    ----------
    baskets_df : pd.DataFrame
        Full transaction basket DataFrame with 'items' column.
    dev_fraction : float, default=0.8
        Fraction used for development (used to scale effective transaction count).

    Returns
    -------
    dict with keys:
        min_support       : float  — recommended minimum support
        min_confidence    : float  — recommended minimum confidence
        explanation       : dict   — per-field reasoning
        dataset_summary   : dict   — raw stats used for the recommendation
    """
    n_transactions = len(baskets_df)
    all_items = [item for basket in baskets_df['items'] for item in basket]
    n_unique_items = len(set(all_items))
    avg_basket_size = float(pd.Series([len(b) for b in baskets_df['items']]).mean())

    # Effective development transactions
    n_dev = int(n_transactions * dev_fraction)

    # ── Support recommendation ────────────────────────────────────────────────
    # Target: at least ~30 transactions covering an itemset (robust rule)
    # Lower bound: max(0.0005, 30 / n_dev)
    # Upper bound: 0.05 (anything above that gives trivially common items only)
    support_floor = max(0.0005, round(30 / n_dev, 4))
    support_ceil = 0.05

    if avg_basket_size < 2.0:
        # Very sparse — items rarely co-occur; need lower support to find anything
        rec_support = max(support_floor, 0.0005)
        sup_reason = (
            f"Average basket size is {avg_basket_size:.2f} (very sparse). "
            "Low support (≥ 0.0005) is required to discover any co-occurrence patterns."
        )
    elif n_dev < 1000:
        # Small dataset — support must be higher for statistical reliability
        rec_support = min(support_ceil, max(support_floor, 0.01))
        sup_reason = (
            f"Development set is small ({n_dev} transactions). "
            "Support ≥ 0.01 ensures each itemset appears in ≥ 10 transactions."
        )
    elif n_dev < 5000:
        rec_support = round(max(support_floor, 0.002), 4)
        sup_reason = (
            f"Medium dataset ({n_dev} dev transactions, {n_unique_items} items). "
            f"Support = {round(max(support_floor, 0.002), 4)} balances coverage and specificity."
        )
    else:
        rec_support = round(max(support_floor, 0.001), 4)
        sup_reason = (
            f"Large dataset ({n_dev} dev transactions). "
            f"Low support (≥ {round(max(support_floor, 0.001), 4)}) is feasible; "
            "each itemset still covers many transactions."
        )

    # ── Confidence recommendation ─────────────────────────────────────────────
    # Retail basket data is typically sparse (confidence < 50% is normal).
    # A floor of 0.05 is standard for exploratory analysis.
    # A floor of 0.1 is better when rules will drive business decisions.
    if avg_basket_size < 2.0:
        rec_confidence = 0.05
        conf_reason = (
            "Very sparse baskets → low confidence threshold (0.05) is appropriate "
            "to surface any directional associations."
        )
    elif avg_basket_size < 3.0:
        rec_confidence = 0.1
        conf_reason = (
            "Moderately sparse baskets. Confidence ≥ 0.10 filters out very weak "
            "associations while retaining exploratory rules."
        )
    else:
        rec_confidence = 0.15
        conf_reason = (
            "Larger average baskets support a slightly higher confidence floor (0.15), "
            "focusing on stronger purchasing associations."
        )

    return {
        "min_support": rec_support,
        "min_confidence": rec_confidence,
        "explanation": {
            "min_support": sup_reason,
            "min_confidence": conf_reason,
        },
        "dataset_summary": {
            "total_transactions": n_transactions,
            "dev_transactions": n_dev,
            "unique_items": n_unique_items,
            "avg_basket_size": round(avg_basket_size, 3),
        }
    }
