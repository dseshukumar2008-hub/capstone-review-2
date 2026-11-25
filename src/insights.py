"""
Business Insights Module for Groceries Transaction Mining Pipeline.
Translates association rules into understandable retail business observations.

Each insight is traceable to a specific rule and its metrics.
Rule-based (no external AI API required). Phrasing is conservative —
observed associations, not proven causal relationships.
"""

import pandas as pd
from typing import List, Dict, Any


# ── Insight type definitions ──────────────────────────────────────────────────

_INSIGHT_THRESHOLDS = {
    "bundle_min_conf": 0.30,   # High confidence → strong bundle candidate
    "cross_sell_min_lift": 1.5, # Lift above 1.5 → statistically notable co-purchase
    "placement_min_lift": 2.0,  # Lift above 2 → strong proximity / display candidate
    "loyalty_min_sup": 0.02,   # High support → common enough to be actionable
}


def _format_items(itemset) -> str:
    """Formats a frozenset or iterable of items as a readable string."""
    try:
        items = sorted(itemset)
    except TypeError:
        items = list(itemset)
    if len(items) == 1:
        return f"'{items[0]}'"
    if len(items) == 2:
        return f"'{items[0]}' and '{items[1]}'"
    return ", ".join(f"'{i}'" for i in items[:-1]) + f", and '{items[-1]}'"


def generate_insights(rules_df: pd.DataFrame, top_n: int = 30) -> List[Dict[str, Any]]:
    """
    Generates business insight observations from association rules.

    Parameters
    ----------
    rules_df : pd.DataFrame
        Association rules with antecedents, consequents, support, confidence, lift, leverage.
    top_n : int, default=30
        Maximum number of insights to return (uses top rules by lift).

    Returns
    -------
    insights : list of dict
        Each dict has: type, title, description, antecedents, consequents,
        support, confidence, lift, leverage.
    """
    if rules_df.empty:
        return []

    df = rules_df.copy()
    # Convert frozensets to string for display if needed
    df = df.sort_values(by='lift', ascending=False).head(top_n * 3)  # extra headroom

    insights = []
    seen_pairs = set()

    for _, row in df.iterrows():
        ant = row['antecedents']
        con = row['consequents']
        sup = float(row['support'])
        conf = float(row['confidence'])
        lift = float(row['lift'])
        lev = float(row.get('leverage', 0.0))

        pair_key = (frozenset(ant), frozenset(con))
        if pair_key in seen_pairs:
            continue
        seen_pairs.add(pair_key)

        ant_str = _format_items(ant)
        con_str = _format_items(con)

        # ── Product Placement / Display Opportunity ──
        if lift >= _INSIGHT_THRESHOLDS["placement_min_lift"]:
            insights.append({
                "type": "Product Placement",
                "icon": "🗂️",
                "title": f"Display {_format_items(con)} near {_format_items(ant)}",
                "description": (
                    f"Customers who purchase {ant_str} are {lift:.1f}× more likely than average "
                    f"to also buy {con_str} (lift = {lift:.2f}, confidence = {conf:.1%}). "
                    f"Placing these products in proximity or in the same aisle section "
                    f"may increase capture of this co-purchase tendency."
                ),
                "antecedents": ant,
                "consequents": con,
                "support": sup,
                "confidence": conf,
                "lift": lift,
                "leverage": lev,
            })

        # ── Bundle / Promotional Offer ──
        if conf >= _INSIGHT_THRESHOLDS["bundle_min_conf"]:
            insights.append({
                "type": "Bundle Offer",
                "icon": "🎁",
                "title": f"Bundle candidate: {_format_items(ant)} + {_format_items(con)}",
                "description": (
                    f"In {conf:.1%} of transactions containing {ant_str}, "
                    f"customers also purchased {con_str}. "
                    f"This strong association (confidence = {conf:.2f}) makes this pair "
                    f"a candidate for a promotional bundle or multi-buy discount."
                ),
                "antecedents": ant,
                "consequents": con,
                "support": sup,
                "confidence": conf,
                "lift": lift,
                "leverage": lev,
            })

        # ── Cross-Selling Opportunity ──
        if lift >= _INSIGHT_THRESHOLDS["cross_sell_min_lift"] and conf < _INSIGHT_THRESHOLDS["bundle_min_conf"]:
            insights.append({
                "type": "Cross-Selling",
                "icon": "↗️",
                "title": f"Cross-sell {_format_items(con)} to buyers of {_format_items(ant)}",
                "description": (
                    f"Customers purchasing {ant_str} are {lift:.1f}× more likely than a random "
                    f"customer to also buy {con_str} (support = {sup:.3f}, lift = {lift:.2f}). "
                    f"This pattern may support targeted cross-sell recommendations at checkout "
                    f"or in digital product suggestion lists."
                ),
                "antecedents": ant,
                "consequents": con,
                "support": sup,
                "confidence": conf,
                "lift": lift,
                "leverage": lev,
            })

        # ── High-Volume Association (loyalty / frequent pattern) ──
        if sup >= _INSIGHT_THRESHOLDS["loyalty_min_sup"] and lift >= 1.0:
            insights.append({
                "type": "Frequent Pattern",
                "icon": "📊",
                "title": f"Common co-purchase: {_format_items(ant)} with {_format_items(con)}",
                "description": (
                    f"This item combination appears in {sup:.1%} of all transactions "
                    f"(support = {sup:.4f}), making it one of the more common co-purchase "
                    f"patterns in the dataset. It may be worth including in loyalty programme "
                    f"or regular-shopper analysis."
                ),
                "antecedents": ant,
                "consequents": con,
                "support": sup,
                "confidence": conf,
                "lift": lift,
                "leverage": lev,
            })

        if len(insights) >= top_n:
            break

    return insights[:top_n]


def insights_to_dataframe(insights: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Converts the insights list to a display-friendly DataFrame.

    Parameters
    ----------
    insights : list of dict

    Returns
    -------
    pd.DataFrame with columns: Type, Title, Antecedents, Consequents,
                                Support, Confidence, Lift, Leverage
    """
    if not insights:
        return pd.DataFrame(columns=[
            'Type', 'Title', 'Antecedents', 'Consequents',
            'Support', 'Confidence', 'Lift', 'Leverage'
        ])

    rows = []
    for ins in insights:
        rows.append({
            "Type": ins.get("icon", "") + " " + ins.get("type", ""),
            "Title": ins.get("title", ""),
            "Antecedents": ", ".join(sorted(ins["antecedents"])),
            "Consequents": ", ".join(sorted(ins["consequents"])),
            "Support": round(ins["support"], 4),
            "Confidence": round(ins["confidence"], 4),
            "Lift": round(ins["lift"], 4),
            "Leverage": round(ins["leverage"], 4),
        })

    return pd.DataFrame(rows)
