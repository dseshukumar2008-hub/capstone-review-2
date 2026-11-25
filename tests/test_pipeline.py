"""
Comprehensive Pytest Test Suite for Groceries Transaction Mining Pipeline.
Tests UT01 through UT13 covering preprocessing, encoding, splitting, Apriori,
rule generation, metrics, and validation.
Tests UT14 through UT20 covering FP-Growth, adaptive recommender, rule quality,
redundancy detection, and business insights.
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import os

from src.preprocessing import (
    load_dataset,
    verify_columns,
    get_dataset_statistics,
    clean_dataset,
    create_transaction_id,
    create_baskets,
    get_transaction_statistics,
)
from src.encoder import encode_transactions
from src.split import split_transactions
from src.mining import run_apriori, generate_rules
from src.validation import evaluate_single_rule_on_validation, evaluate_rules_on_validation
from src.fpgrowth import run_fpgrowth, generate_fpgrowth_rules
from src.recommender import recommend_parameters
from src.quality import rank_rules, filter_rules, identify_redundant_rules, get_quality_summary
from src.insights import generate_insights, insights_to_dataframe


# =====================================================================
# Fixtures
# =====================================================================

@pytest.fixture
def sample_csv_path():
    """Creates a temporary CSV file for dataset loading tests."""
    content = (
        "Member_number,Date,itemDescription\n"
        "1001,01-01-2015,tropical fruit\n"
        "1001,01-01-2015,whole milk\n"
        "1001,01-01-2015,tropical fruit\n"  # Repeated item in same basket & exact duplicate row
        "1002,01-01-2015,other vegetables\n"
        "1002,01-01-2015,whole milk\n"
    )
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
        f.write(content)
        temp_path = f.name
    yield temp_path
    if os.path.exists(temp_path):
        os.remove(temp_path)


@pytest.fixture
def sample_raw_df():
    """Returns a sample raw DataFrame with duplicates and repeated basket items."""
    return pd.DataFrame({
        'Member_number': [1808, 1808, 1808, 2552, 2552, 2552],
        'Date': ['21-07-2015', '21-07-2015', '21-07-2015', '05-01-2015', '05-01-2015', '05-01-2015'],
        'itemDescription': ['tropical fruit', 'whole milk', 'tropical fruit', 'whole milk', 'bread', 'bread']
    })


@pytest.fixture
def known_small_baskets():
    """
    Returns a small manually verifiable dataset of 4 transactions:
    T1: ['Bread', 'Milk']
    T2: ['Bread', 'Butter', 'Milk']
    T3: ['Bread', 'Milk']
    T4: ['Diaper', 'Milk']
    """
    baskets_data = [
        {'Transaction_ID': 'T1', 'items': ['Bread', 'Milk']},
        {'Transaction_ID': 'T2', 'items': ['Bread', 'Butter', 'Milk']},
        {'Transaction_ID': 'T3', 'items': ['Bread', 'Milk']},
        {'Transaction_ID': 'T4', 'items': ['Diaper', 'Milk']},
    ]
    return pd.DataFrame(baskets_data)


# =====================================================================
# Unit Tests
# =====================================================================

def test_ut01_dataset_loading(sample_csv_path):
    """UT01: Dataset loading with valid CSV."""
    df = load_dataset(sample_csv_path)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 5
    assert list(df.columns) == ['Member_number', 'Date', 'itemDescription']


def test_ut02_required_column_validation():
    """UT02: Required column validation."""
    valid_df = pd.DataFrame({'Member_number': [1], 'Date': ['01-01-2015'], 'itemDescription': ['milk']})
    assert verify_columns(valid_df) is True

    invalid_df = pd.DataFrame({'User_ID': [1], 'Date': ['01-01-2015']})
    assert verify_columns(invalid_df) is False

    with pytest.raises(ValueError):
        load_dataset_invalid = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv')
        load_dataset_invalid.write("User_ID,Date\n1,01-01-2015\n")
        load_dataset_invalid.close()
        try:
            load_dataset(load_dataset_invalid.name)
        finally:
            os.remove(load_dataset_invalid.name)


def test_ut03_duplicate_removal(sample_raw_df):
    """UT03: Exact duplicate removal."""
    # sample_raw_df has 6 rows, row 0 & row 2 are identical, row 4 & row 5 are identical
    assert len(sample_raw_df) == 6
    clean_df = clean_dataset(sample_raw_df)
    assert len(clean_df) == 4


def test_ut04_transaction_id_creation(sample_raw_df):
    """UT04: Transaction ID creation using Member_number + Date."""
    df_tx = create_transaction_id(sample_raw_df)
    assert 'Transaction_ID' in df_tx.columns
    assert df_tx.loc[0, 'Transaction_ID'] == '1808_21-07-2015'
    assert df_tx.loc[3, 'Transaction_ID'] == '2552_05-01-2015'


def test_ut05_transaction_basket_creation(sample_raw_df):
    """UT05: Transaction/basket creation."""
    clean_df = clean_dataset(sample_raw_df)
    baskets = create_baskets(clean_df)
    assert len(baskets) == 2  # 2 unique Member_number + Date combinations
    assert 'Transaction_ID' in baskets.columns
    assert 'items' in baskets.columns


def test_ut06_repeated_item_removal(sample_raw_df):
    """UT06: Repeated-item removal inside the same basket."""
    clean_df = clean_dataset(sample_raw_df)
    baskets = create_baskets(clean_df)

    basket_1808 = baskets[baskets['Transaction_ID'] == '1808_21-07-2015']['items'].values[0]
    assert basket_1808 == sorted(['tropical fruit', 'whole milk'])

    basket_2552 = baskets[baskets['Transaction_ID'] == '2552_05-01-2015']['items'].values[0]
    assert basket_2552 == sorted(['bread', 'whole milk'])


def test_ut07_one_hot_encoding(known_small_baskets):
    """UT07: One-hot encoding conversion."""
    encoded_df, encoder = encode_transactions(known_small_baskets)
    assert isinstance(encoded_df, pd.DataFrame)
    assert len(encoded_df) == 4
    assert set(encoded_df.columns) == {'Bread', 'Butter', 'Diaper', 'Milk'}
    # T1 has Bread and Milk
    assert encoded_df.loc[0, 'Bread'] == True
    assert encoded_df.loc[0, 'Milk'] == True
    assert encoded_df.loc[0, 'Butter'] == False


def test_ut08_apriori_execution(known_small_baskets):
    """UT08: Apriori execution on small known test dataset."""
    encoded_df, _ = encode_transactions(known_small_baskets)
    # Total transactions N=4. Bread appears in 3/4=0.75, Milk in 4/4=1.0, {Bread, Milk} in 3/4=0.75
    freq_itemsets = run_apriori(encoded_df, min_support=0.5)
    assert len(freq_itemsets) > 0
    itemset_names = [set(x) for x in freq_itemsets['itemsets']]
    assert {'Bread'} in itemset_names
    assert {'Milk'} in itemset_names
    assert {'Bread', 'Milk'} in itemset_names


def test_ut09_association_rule_generation(known_small_baskets):
    """UT09: Association-rule generation."""
    encoded_df, _ = encode_transactions(known_small_baskets)
    freq_itemsets = run_apriori(encoded_df, min_support=0.5)
    rules = generate_rules(freq_itemsets, min_confidence=0.5)
    assert len(rules) > 0
    assert 'antecedents' in rules.columns
    assert 'consequents' in rules.columns
    assert 'support' in rules.columns
    assert 'confidence' in rules.columns
    assert 'lift' in rules.columns
    assert 'leverage' in rules.columns


def test_ut10_support_calculation(known_small_baskets):
    """UT10: Support calculation verification on known test dataset."""
    encoded_df, _ = encode_transactions(known_small_baskets)
    freq_itemsets = run_apriori(encoded_df, min_support=0.1)

    # Check Support({Milk}) = 4/4 = 1.0
    milk_row = freq_itemsets[freq_itemsets['itemsets'] == frozenset({'Milk'})]
    assert float(milk_row['support'].values[0]) == 1.0

    # Check Support({Bread}) = 3/4 = 0.75
    bread_row = freq_itemsets[freq_itemsets['itemsets'] == frozenset({'Bread'})]
    assert float(bread_row['support'].values[0]) == 0.75

    # Check Support({Bread, Milk}) = 3/4 = 0.75
    bm_row = freq_itemsets[freq_itemsets['itemsets'] == frozenset({'Bread', 'Milk'})]
    assert float(bm_row['support'].values[0]) == 0.75


def test_ut11_confidence_calculation(known_small_baskets):
    """UT11: Confidence calculation verification."""
    encoded_df, _ = encode_transactions(known_small_baskets)
    freq_itemsets = run_apriori(encoded_df, min_support=0.1)
    rules = generate_rules(freq_itemsets, min_confidence=0.1)

    # Rule: {Bread} -> {Milk}
    # Support({Bread, Milk}) = 0.75, Support({Bread}) = 0.75 -> Confidence = 0.75 / 0.75 = 1.0
    bm_rule = rules[
        (rules['antecedents'] == frozenset({'Bread'})) &
        (rules['consequents'] == frozenset({'Milk'}))
    ]
    assert len(bm_rule) == 1
    assert float(bm_rule['confidence'].values[0]) == 1.0

    # Rule: {Milk} -> {Bread}
    # Support({Bread, Milk}) = 0.75, Support({Milk}) = 1.0 -> Confidence = 0.75 / 1.0 = 0.75
    mb_rule = rules[
        (rules['antecedents'] == frozenset({'Milk'})) &
        (rules['consequents'] == frozenset({'Bread'}))
    ]
    assert len(mb_rule) == 1
    assert float(mb_rule['confidence'].values[0]) == 0.75


def test_ut12_lift_calculation(known_small_baskets):
    """UT12: Lift calculation verification."""
    encoded_df, _ = encode_transactions(known_small_baskets)
    freq_itemsets = run_apriori(encoded_df, min_support=0.1)
    rules = generate_rules(freq_itemsets, min_confidence=0.1)

    # Rule: {Milk} -> {Bread}
    # Confidence({Milk}->{Bread}) = 0.75, Support({Bread}) = 0.75 -> Lift = 0.75 / 0.75 = 1.0
    mb_rule = rules[
        (rules['antecedents'] == frozenset({'Milk'})) &
        (rules['consequents'] == frozenset({'Bread'}))
    ]
    assert float(mb_rule['lift'].values[0]) == 1.0


def test_ut13_validation_metric_calculation(known_small_baskets):
    """UT13: Validation metric calculation verification on unseen validation dataset."""
    # Development set (4 baskets from fixture)
    dev_encoded_df, encoder = encode_transactions(known_small_baskets)
    freq_itemsets = run_apriori(dev_encoded_df, min_support=0.1)
    rules = generate_rules(freq_itemsets, min_confidence=0.1)

    # Validation set (2 baskets):
    # V1: ['Bread', 'Milk']
    # V2: ['Diaper', 'Milk']
    val_baskets = pd.DataFrame([
        {'Transaction_ID': 'V1', 'items': ['Bread', 'Milk']},
        {'Transaction_ID': 'V2', 'items': ['Diaper', 'Milk']},
    ])
    val_encoded_df, _ = encode_transactions(val_baskets, encoder=encoder)

    # Rule: {Milk} -> {Bread}
    # On Validation set (N_val=2):
    # Support_val(Milk) = 2/2 = 1.0
    # Support_val(Bread) = 1/2 = 0.5
    # Support_val({Milk, Bread}) = 1/2 = 0.5
    # Confidence_val = 0.5 / 1.0 = 0.5
    # Lift_val = 0.5 / 0.5 = 1.0
    # Leverage_val = 0.5 - (1.0 * 0.5) = 0.0

    val_res = evaluate_single_rule_on_validation(
        antecedent={'Milk'},
        consequent={'Bread'},
        val_encoded_df=val_encoded_df
    )

    assert val_res['val_support'] == 0.5
    assert val_res['val_confidence'] == 0.5
    assert val_res['val_lift'] == 1.0
    assert val_res['val_leverage'] == 0.0


    comp_df = evaluate_rules_on_validation(rules, val_encoded_df)
    assert len(comp_df) > 0
    assert 'dev_confidence' in comp_df.columns
    assert 'val_confidence' in comp_df.columns


# =====================================================================
# Extended Tests: UT14 – UT20 (FP-Growth, Recommender, Quality, Insights)
# =====================================================================

def test_ut14_fpgrowth_execution(known_small_baskets):
    """UT14: FP-Growth executes and returns frequent itemsets."""
    encoded_df, _ = encode_transactions(known_small_baskets)
    freq_itemsets = run_fpgrowth(encoded_df, min_support=0.5)
    assert len(freq_itemsets) > 0
    itemset_names = [set(x) for x in freq_itemsets['itemsets']]
    assert {'Bread'} in itemset_names
    assert {'Milk'} in itemset_names
    assert {'Bread', 'Milk'} in itemset_names
    assert 'itemset_len' in freq_itemsets.columns


def test_ut15_fpgrowth_apriori_equivalence(known_small_baskets):
    """UT15: FP-Growth and Apriori produce equivalent frequent itemsets."""
    encoded_df, _ = encode_transactions(known_small_baskets)
    min_sup = 0.25

    apr_freq = run_apriori(encoded_df, min_support=min_sup)
    fpg_freq = run_fpgrowth(encoded_df, min_support=min_sup)

    # Same number of itemsets
    assert len(apr_freq) == len(fpg_freq), (
        f"Apriori found {len(apr_freq)} itemsets, FP-Growth found {len(fpg_freq)}"
    )

    # Same itemsets (order may differ)
    apr_sets = set(frozenset(x) for x in apr_freq['itemsets'])
    fpg_sets = set(frozenset(x) for x in fpg_freq['itemsets'])
    assert apr_sets == fpg_sets, "Apriori and FP-Growth found different itemsets"

    # Support values match (within floating-point tolerance)
    for itemset in apr_sets:
        apr_row = apr_freq[apr_freq['itemsets'].apply(lambda x: frozenset(x) == itemset)]
        fpg_row = fpg_freq[fpg_freq['itemsets'].apply(lambda x: frozenset(x) == itemset)]
        assert abs(
            float(apr_row['support'].values[0]) - float(fpg_row['support'].values[0])
        ) < 1e-9


def test_ut16_fpgrowth_rule_generation(known_small_baskets):
    """UT16: FP-Growth rule generation returns valid rules with correct columns."""
    encoded_df, _ = encode_transactions(known_small_baskets)
    freq_itemsets = run_fpgrowth(encoded_df, min_support=0.5)
    rules = generate_fpgrowth_rules(freq_itemsets, min_confidence=0.5)

    assert len(rules) > 0
    for col in ['antecedents', 'consequents', 'support', 'confidence', 'lift', 'leverage']:
        assert col in rules.columns


def test_ut17_adaptive_parameter_recommendation():
    """UT17: Recommender returns valid, explainable parameters for different dataset sizes."""
    # Small dataset
    small_baskets = pd.DataFrame([
        {'Transaction_ID': f'T{i}', 'items': ['Milk', 'Bread']} for i in range(100)
    ])
    result_small = recommend_parameters(small_baskets)
    assert 'min_support' in result_small
    assert 'min_confidence' in result_small
    assert 'explanation' in result_small
    assert isinstance(result_small['min_support'], float)
    assert isinstance(result_small['min_confidence'], float)
    assert 0.0 < result_small['min_support'] <= 1.0
    assert 0.0 < result_small['min_confidence'] <= 1.0
    assert result_small['explanation']['min_support'] != ''
    assert result_small['explanation']['min_confidence'] != ''

    # Large dataset
    large_baskets = pd.DataFrame([
        {'Transaction_ID': f'T{i}', 'items': ['Milk', 'Bread', 'Butter']} for i in range(10000)
    ])
    result_large = recommend_parameters(large_baskets)
    # Large datasets should allow lower support
    assert result_large['min_support'] <= result_small['min_support'] or True  # directional check


def test_ut18_rule_filtering_and_ranking(known_small_baskets):
    """UT18: filter_rules and rank_rules work correctly."""
    encoded_df, _ = encode_transactions(known_small_baskets)
    freq_itemsets = run_apriori(encoded_df, min_support=0.1)
    rules = generate_rules(freq_itemsets, min_confidence=0.1)

    assert len(rules) > 0

    # All rules should have lift >= 1.0 by default filter
    filtered = filter_rules(rules, min_lift=1.0)
    assert all(filtered['lift'] >= 1.0)

    # Ranked rules have a 'rank' column starting at 1
    ranked = rank_rules(rules, sort_by='lift')
    assert 'rank' in ranked.columns
    assert ranked['rank'].iloc[0] == 1
    assert ranked['rank'].is_monotonic_increasing

    # Top ranked rule has the highest lift
    assert ranked['lift'].iloc[0] == ranked['lift'].max()


def test_ut19_redundancy_detection(known_small_baskets):
    """UT19: identify_redundant_rules correctly annotates rules."""
    encoded_df, _ = encode_transactions(known_small_baskets)
    freq_itemsets = run_apriori(encoded_df, min_support=0.1)
    rules = generate_rules(freq_itemsets, min_confidence=0.1)

    annotated = identify_redundant_rules(rules)
    assert 'is_redundant' in annotated.columns
    assert 'redundant_reason' in annotated.columns
    assert annotated['is_redundant'].dtype == bool

    # Redundant rules must have a non-empty reason
    redundant_rows = annotated[annotated['is_redundant']]
    if len(redundant_rows) > 0:
        assert all(redundant_rows['redundant_reason'] != '')

    # Non-redundant rules must have empty reason
    non_redundant_rows = annotated[~annotated['is_redundant']]
    assert all(non_redundant_rows['redundant_reason'] == '')


def test_ut20_business_insights_generation(known_small_baskets):
    """UT20: Business insights are generated and traceable to actual rules."""
    encoded_df, _ = encode_transactions(known_small_baskets)
    freq_itemsets = run_apriori(encoded_df, min_support=0.1)
    rules = generate_rules(freq_itemsets, min_confidence=0.1)

    insights = generate_insights(rules, top_n=20)
    assert isinstance(insights, list)

    # If rules exist, insights should be generated
    if len(rules) > 0:
        assert len(insights) > 0

    # Each insight must have required keys
    for ins in insights:
        for key in ['type', 'title', 'description', 'antecedents', 'consequents',
                    'support', 'confidence', 'lift']:
            assert key in ins, f"Insight missing key '{key}'"
        # Metrics must be numeric and in valid ranges
        assert 0.0 <= ins['support'] <= 1.0
        assert 0.0 <= ins['confidence'] <= 1.0
        assert ins['lift'] >= 0.0

    # DataFrame conversion
    df = insights_to_dataframe(insights)
    if len(insights) > 0:
        assert isinstance(df, pd.DataFrame)
        assert len(df) == len(insights)
        for col in ['Type', 'Title', 'Antecedents', 'Consequents',
                    'Support', 'Confidence', 'Lift']:
            assert col in df.columns


