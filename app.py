"""
Scalable Association Rule Mining and Estimator
Restructured 9-Page Integrated Architecture with Multi-Tab Workflows
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import io
import json
import subprocess
import sys
import xml.etree.ElementTree as ET
import time
import plotly.express as px
import plotly.graph_objects as go

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
from src.fpgrowth import run_fpgrowth, generate_fpgrowth_rules
from src.recommender import recommend_parameters
from src.quality import rank_rules, filter_rules, identify_redundant_rules, get_quality_summary
from src.insights import generate_insights, insights_to_dataframe
from src.validation import evaluate_rules_on_validation
from src.metrics import measure_performance, get_system_resource_info


# =====================================================================
# Page Configuration & Global Styling
# =====================================================================
st.set_page_config(
    page_title="Scalable Association Rule Mining and Estimator",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Theme Management
if "theme_mode" not in st.session_state:
    st.session_state.theme_mode = "dark"

is_dark = (st.session_state.theme_mode == "dark")

bg_canvas    = "#0B1120" if is_dark else "#F8FAFC"
bg_sidebar   = "#0F172A" if is_dark else "#FFFFFF"
bg_card      = "#1E293B" if is_dark else "#FFFFFF"
border_card  = "#334155" if is_dark else "#E2E8F0"
border_hover = "#475569" if is_dark else "#CBD5E1"
text_primary = "#F8FAFC" if is_dark else "#0F172A"
text_secondary="#CBD5E1" if is_dark else "#334155"
text_muted   = "#94A3B8" if is_dark else "#64748B"

accent_blue   = "#38BDF8"
accent_teal   = "#2DD4BF"
accent_purple = "#A78BFA"
accent_green  = "#34D399"
accent_orange = "#FBBF24"
accent_red    = "#F87171"

shadow_card = "0 4px 8px -1px rgba(0,0,0,0.35)" if is_dark else "0 2px 6px 0 rgba(0,0,0,0.06)"
header_bg   = "linear-gradient(135deg, #1E293B 0%, #0F172A 100%)" if is_dark else "linear-gradient(135deg, #FFFFFF 0%, #F1F5F9 100%)"

btn_bg         = "#1E293B" if is_dark else "#F1F5F9"
btn_border     = "#334155" if is_dark else "#E2E8F0"
btn_text       = "#CBD5E1" if is_dark else "#334155"
btn_hover_bg   = "#283548" if is_dark else "#E2E8F0"
btn_active_bg  = "linear-gradient(90deg, #1E293B 0%, #0369A144 100%)" if is_dark else "linear-gradient(90deg, #FFFFFF 0%, #DBEAFE 100%)"
btn_active_bdr = "#38BDF8" if is_dark else "#2563EB"

PLOTLY_TEMPLATE = "plotly_dark" if is_dark else "plotly_white"
PLOTLY_PAPER_BG = "#0B1120" if is_dark else "#F8FAFC"
PLOTLY_PLOT_BG  = "#1E293B" if is_dark else "#FFFFFF"
PLOTLY_FONT_CLR = "#F8FAFC" if is_dark else "#0F172A"
PLOTLY_GRID_CLR = "#334155" if is_dark else "#E2E8F0"


def apply_plotly_theme(fig, height=380):
    fig.update_layout(
        paper_bgcolor=PLOTLY_PAPER_BG,
        plot_bgcolor=PLOTLY_PLOT_BG,
        font=dict(color=PLOTLY_FONT_CLR, family="Inter, -apple-system, sans-serif"),
        height=height,
        margin=dict(l=24, r=24, t=44, b=24),
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="rgba(0,0,0,0)"),
        xaxis=dict(gridcolor=PLOTLY_GRID_CLR, zerolinecolor=PLOTLY_GRID_CLR),
        yaxis=dict(gridcolor=PLOTLY_GRID_CLR, zerolinecolor=PLOTLY_GRID_CLR),
    )
    return fig


st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    html, body, [class*="css"] {{ font-family: 'Inter', -apple-system, sans-serif; }}

    .block-container, div.block-container, [data-testid="stAppViewBlockContainer"] {{
        padding-top: 1.2rem !important; padding-bottom: 1.5rem !important;
        padding-left: 2rem !important; padding-right: 2rem !important;
        max-width: 100% !important;
    }}
    header[data-testid="stHeader"] {{ display: none !important; height: 0px !important; visibility: hidden !important; }}
    [data-testid="stAppDeployButton"], .stDeployButton, [data-testid="stToolbar"], [data-testid="stDecoration"] {{
        display: none !important; visibility: hidden !important; width: 0px !important; height: 0px !important;
    }}
    #MainMenu {{ visibility: hidden !important; display: none !important; }}
    footer {{ visibility: hidden !important; display: none !important; }}
    .stApp {{ background-color: {bg_canvas} !important; color: {text_primary} !important; }}
    
    section[data-testid="stSidebar"] {{
        background-color: {bg_sidebar} !important;
        border-right: 1px solid {border_card} !important;
    }}
    
    div[data-testid="stSidebar"] div.stButton {{ margin-bottom: 4px !important; width: 100% !important; }}
    div[data-testid="stSidebar"] div.stButton > button {{
        width: 100% !important; display: flex !important; justify-content: flex-start !important;
        align-items: center !important; border-radius: 8px !important; padding: 9px 12px !important;
        font-size: 0.86rem !important; font-weight: 600 !important; text-align: left !important;
        transition: all 0.15s ease-in-out !important; min-height: 40px !important; box-sizing: border-box !important;
    }}
    div[data-testid="stSidebar"] div.stButton > button[kind="secondary"] {{
        background-color: {btn_bg} !important; border: 1px solid {btn_border} !important;
        border-left: 4px solid transparent !important; color: {btn_text} !important;
    }}
    div[data-testid="stSidebar"] div.stButton > button[kind="secondary"]:hover {{
        background-color: {btn_hover_bg} !important; color: {text_primary} !important;
        border-color: {border_hover} !important; transform: translateX(2px);
    }}
    div[data-testid="stSidebar"] div.stButton > button[kind="primary"] {{
        background: {btn_active_bg} !important; border: 1px solid {btn_active_bdr} !important;
        border-left: 4px solid {btn_active_bdr} !important; color: {text_primary} !important;
        font-weight: 700 !important; box-shadow: 0 2px 8px rgba(56, 189, 248, 0.18) !important;
    }}
    
    .nav-label {{
        font-size: 0.7rem; font-weight: 800; color: {text_muted}; letter-spacing: 0.08em;
        text-transform: uppercase; margin: 12px 0 6px 2px;
    }}
    
    .kpi-card {{
        background: {bg_card}; border: 1px solid {border_card}; border-radius: 12px;
        padding: 16px 18px; box-shadow: {shadow_card}; transition: transform 0.15s, border-color 0.15s;
    }}
    .kpi-card:hover {{ border-color: {border_hover}; transform: translateY(-2px); }}
    .kpi-value {{ font-size: 2.0rem; font-weight: 800; color: {accent_blue}; line-height: 1.1; margin-bottom: 4px; }}
    .kpi-label {{ font-size: 0.78rem; font-weight: 700; text-transform: uppercase; color: {text_primary};
        letter-spacing: 0.05em; margin-bottom: 2px; }}
    .kpi-desc {{ font-size: 0.74rem; color: {text_muted}; }}

    .page-header {{
        background: {header_bg}; border: 1px solid {border_card}; border-radius: 12px;
        padding: 16px 22px; margin-bottom: 18px; box-shadow: {shadow_card};
    }}
    .page-title {{ font-size: 1.55rem; font-weight: 800; color: {text_primary}; letter-spacing: -0.02em; margin: 0; }}
    .page-subtitle {{ font-size: 0.88rem; color: {text_muted}; margin: 4px 0 0 0; }}
    .section-divider {{ border: none; border-top: 1px solid {border_card}; margin: 18px 0; }}

    .badge-ok {{
        background: {'#064E3B' if is_dark else '#D1FAE5'}; color: {'#34D399' if is_dark else '#065F46'};
        font-size: 0.72rem; font-weight: 700; padding: 3px 9px; border-radius: 10px;
        display: inline-block; border: 1px solid {'#059669' if is_dark else '#10B981'};
    }}
    .badge-warn {{
        background: {'#451A03' if is_dark else '#FEF3C7'}; color: {'#FBBF24' if is_dark else '#92400E'};
        font-size: 0.72rem; font-weight: 700; padding: 3px 9px; border-radius: 10px;
        display: inline-block; border: 1px solid {'#D97706' if is_dark else '#F59E0B'};
    }}

    .insight-card {{
        background: {bg_card}; border: 1px solid {border_card}; border-radius: 10px;
        padding: 16px; margin-bottom: 12px;
    }}
    h1, h2, h3, h4, h5, h6 {{ color: {text_primary} !important; }}
    p, span, label {{ color: {text_secondary}; }}

    div[data-testid="stMetricValue"] {{ color: {accent_blue} !important; font-weight: 800 !important; }}
    div[data-testid="stMetricLabel"] {{ color: {text_muted} !important; font-weight: 600 !important; }}
    
    div.stButton > button[kind="primary"] {{
        background-color: #0284C7 !important; color: #FFFFFF !important; border: none !important;
        box-shadow: 0 4px 12px rgba(2, 132, 199, 0.35) !important;
    }}
    div[data-testid="stDataFrame"] {{ border: 1px solid {border_card}; border-radius: 8px; }}
    div[data-testid="stExpander"] {{ border: 1px solid {border_card} !important; border-radius: 8px !important; background-color: {bg_card} !important; }}
    div.stAlert {{ background-color: {bg_card} !important; border: 1px solid {border_card} !important; color: {text_primary} !important; border-radius: 8px !important; }}
    div[data-testid="stTab"] button {{ color: {text_muted} !important; font-weight: 600 !important; font-size: 0.88rem !important; }}
    div[data-testid="stTab"] button[aria-selected="true"] {{ color: {accent_blue} !important; border-bottom-color: {accent_blue} !important; font-weight: 700 !important; }}
</style>
""", unsafe_allow_html=True)


# =====================================================================
# UI Helper Functions
# =====================================================================
def kpi(value, label, desc="", color=None):
    c = color or accent_blue
    return f"""<div class="kpi-card">
        <div class="kpi-value" style="color:{c};">{value}</div>
        <div class="kpi-label">{label}</div>
        <div class="kpi-desc">{desc}</div>
    </div>"""


def render_page_header(title, subtitle=""):
    st.markdown(f"""<div class="page-header">
        <h2 class="page-title">{title}</h2>
        <p class="page-subtitle">{subtitle}</p>
    </div>""", unsafe_allow_html=True)


# =====================================================================
# Unit Test Runner (UT01 to UT20)
# =====================================================================
def run_actual_pytest():
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    xml_path = os.path.join(results_dir, "test_results.xml")
    res = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/test_pipeline.py", f"--junitxml={xml_path}", "-v"],
        capture_output=True, text=True,
    )
    meta = {
        "test_ut01_dataset_loading": ("UT01","src.preprocessing","Dataset loading with valid CSV","Valid DataFrame loaded"),
        "test_ut02_required_column_validation": ("UT02","src.preprocessing","Required column verification","Returns True for valid schema"),
        "test_ut03_duplicate_removal": ("UT03","src.preprocessing","Exact duplicate removal","Drops exact duplicate rows"),
        "test_ut04_transaction_id_creation": ("UT04","src.preprocessing","Transaction ID creation","Creates Transaction_ID"),
        "test_ut05_transaction_basket_creation": ("UT05","src.preprocessing","Transaction basket aggregation","Groups items into baskets"),
        "test_ut06_repeated_item_removal": ("UT06","src.preprocessing","Inner basket item deduplication","Removes repeated items"),
        "test_ut07_one_hot_encoding": ("UT07","src.encoder","One-hot binary matrix encoding","Converts baskets to boolean DataFrame"),
        "test_ut08_apriori_execution": ("UT08","src.mining","Apriori algorithm execution","Discovers frequent itemsets"),
        "test_ut09_association_rule_generation": ("UT09","src.mining","Association rule generation","Generates valid rules"),
        "test_ut10_support_calculation": ("UT10","src.mining","Support metric verification","Support matches manual formula"),
        "test_ut11_confidence_calculation": ("UT11","src.mining","Confidence metric verification","Confidence matches manual formula"),
        "test_ut12_lift_calculation": ("UT12","src.mining","Lift metric verification","Lift matches manual formula"),
        "test_ut13_validation_metric_calculation": ("UT13","src.validation","Validation metric evaluation","Calculates val metrics"),
        "test_ut14_fpgrowth_execution": ("UT14","src.fpgrowth","FP-Growth itemset discovery","Discovers frequent itemsets"),
        "test_ut15_fpgrowth_apriori_equivalence": ("UT15","src.fpgrowth","FP-Growth vs Apriori equivalence","Identical itemsets"),
        "test_ut16_fpgrowth_rule_generation": ("UT16","src.fpgrowth","FP-Growth rule generation","Valid rules with all columns"),
        "test_ut17_adaptive_parameter_recommendation": ("UT17","src.recommender","Adaptive parameter recommendation","Returns explainable params"),
        "test_ut18_rule_filtering_and_ranking": ("UT18","src.quality","Rule filtering and ranking","Correct output"),
        "test_ut19_redundancy_detection": ("UT19","src.quality","Redundant rule identification","Annotates redundant rules"),
        "test_ut20_business_insights_generation": ("UT20","src.insights","Business insight generation","Traceable insights"),
    }
    test_results = []
    if os.path.exists(xml_path):
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            for tc in root.iter("testcase"):
                name = tc.attrib.get("name")
                failure = tc.find("failure")
                error = tc.find("error")
                status = "FAILED" if (failure is not None or error is not None) else "PASSED"
                actual = "Assertion passed" if status == "PASSED" else (
                    failure.attrib.get("message","Assertion error") if failure is not None else "Runtime error")
                if name in meta:
                    ut_id, module, desc, expected = meta[name]
                else:
                    ut_id, module, desc, expected = "UT??", "src", name, "Test execution"
                test_results.append({
                    "Test ID": ut_id, "Module": module, "Test Case": desc,
                    "Expected Result": expected, "Actual Result": actual,
                    "Status": status, "func_name": name,
                })
        except Exception as e:
            st.error(f"Error parsing XML: {e}")
    return test_results, res.stdout, res.returncode


# =====================================================================
# Dataset Source & Cache Management
# =====================================================================
default_dataset_path = os.path.join("data", "Groceries_dataset.csv")
if not os.path.exists(default_dataset_path) and os.path.exists(os.path.join("archive", "Groceries_dataset.csv")):
    default_dataset_path = os.path.join("archive", "Groceries_dataset.csv")

if "dataset_selection" not in st.session_state:
    st.session_state.dataset_selection = "Default Dataset (Groceries)"

if "uploaded_file_bytes" not in st.session_state:
    st.session_state.uploaded_file_bytes = None
    st.session_state.uploaded_file_name = None


# =====================================================================
# Sidebar: Branding & 9 Main Pages Navigation
# =====================================================================
st.sidebar.markdown(f"""
<div style="padding:12px 14px;background:{header_bg};border-radius:10px;border:1px solid {border_card};margin-bottom:12px;box-shadow:{shadow_card};">
    <div style="font-size:1.1rem;color:{accent_blue};font-weight:800;margin-bottom:2px;">◈</div>
    <h1 style="font-size:0.92rem;font-weight:800;color:{text_primary};margin:0;line-height:1.3;">SCALABLE ASSOCIATION RULE<br/>MINING AND ESTIMATOR</h1>
    <p style="font-size:0.72rem;font-weight:600;color:{accent_purple};margin:4px 0 0 0;">Transaction Pattern Platform</p>
</div>
""", unsafe_allow_html=True)

# 9 Consolidated Navigation Pages
MAIN_PAGES = [
    ("📊 Dashboard", "Dashboard"),
    ("📁 Dataset Management", "Dataset Management"),
    ("⚙️ Data Preprocessing", "Data Preprocessing"),
    ("🎯 Adaptive Parameters", "Adaptive Parameters"),
    ("⚡ Association Rule Mining", "Association Rule Mining"),
    ("🔍 Rule Evaluation", "Rule Evaluation"),
    ("🤖 AI Business Intelligence", "AI Business Intelligence"),
    ("📈 Algorithm Comparison", "Algorithm Comparison"),
    ("📋 Testing & Reports", "Testing & Reports"),
]

if "current_page" not in st.session_state:
    st.session_state.current_page = "Dashboard"

st.sidebar.markdown("<p class='nav-label'>MAIN NAVIGATION (9 PAGES)</p>", unsafe_allow_html=True)

for label, page_key in MAIN_PAGES:
    is_active = (st.session_state.current_page == page_key)
    if st.sidebar.button(label, key=f"nav_{page_key}", use_container_width=True,
                         type="primary" if is_active else "secondary"):
        st.session_state.current_page = page_key
        st.rerun()

current_page = st.session_state.current_page

# Dataset Active Badge in Sidebar
st.sidebar.markdown("<hr style='border:none;border-top:1px solid #334155;margin:12px 0;'>", unsafe_allow_html=True)
st.sidebar.markdown("<p class='nav-label'>ACTIVE DATASET SOURCE</p>", unsafe_allow_html=True)

raw_df = None
active_dataset_label = "None"

if st.session_state.dataset_selection == "Uploaded CSV Dataset" and st.session_state.uploaded_file_bytes is not None:
    try:
        raw_df = pd.read_csv(io.BytesIO(st.session_state.uploaded_file_bytes))
        active_dataset_label = f"Uploaded: {st.session_state.uploaded_file_name}"
    except Exception:
        raw_df = None
elif os.path.exists(default_dataset_path):
    raw_df = pd.read_csv(default_dataset_path)
    active_dataset_label = "Default: Groceries (38k)"

if raw_df is not None:
    st.sidebar.markdown(f"""
    <div style="display:flex;align-items:center;gap:8px;padding:6px 12px;background:{bg_card};border-radius:18px;border:1px solid {border_card};">
        <span style="color:{accent_green};font-size:0.75rem;">●</span>
        <span style="font-size:0.78rem;font-weight:700;color:{text_primary};">{active_dataset_label}</span>
    </div>
    """, unsafe_allow_html=True)
else:
    st.sidebar.markdown(f"""
    <div style="display:flex;align-items:center;gap:8px;padding:6px 12px;background:{bg_card};border-radius:18px;border:1px solid {border_card};">
        <span style="color:{accent_red};font-size:0.75rem;">○</span>
        <span style="font-size:0.78rem;font-weight:700;color:{text_muted};">No Dataset Loaded</span>
    </div>
    """, unsafe_allow_html=True)


# =====================================================================
# Global Top Header (Dual Theme Toggle & Title)
# =====================================================================
top_col1, top_col2 = st.columns([0.82, 0.18])
with top_col1:
    st.markdown(f"""
    <div class="page-header" style="margin-bottom:16px;">
        <h1 style="font-size:1.45rem;font-weight:800;color:{text_primary};margin:0 0 2px 0;">
            Scalable Association Rule Mining and Estimator
        </h1>
        <p style="font-size:0.85rem;color:{text_muted};margin:0;">
            Transaction Mining, Adaptive Thresholds & Decision Support Engine
            &nbsp;|&nbsp; <strong style="color:{accent_blue};">{current_page}</strong>
            &nbsp;|&nbsp; <span style="color:{accent_green};">● Ready</span>
        </p>
    </div>
    """, unsafe_allow_html=True)

with top_col2:
    theme_btn_text = "☀ Light Theme" if is_dark else "☾ Dark Theme"
    if st.button(theme_btn_text, key="theme_toggle_btn", use_container_width=True):
        st.session_state.theme_mode = "light" if is_dark else "dark"
        st.rerun()


# =====================================================================
# Core Data Pipeline (Cached by Dataset Hash / Content)
# =====================================================================
if raw_df is not None:
    if not verify_columns(raw_df):
        st.error("⚠️ The selected CSV dataset is missing required schema columns (`Member_number`, `Date`, `itemDescription`). Please select or upload a valid dataset.")
        st.stop()

    @st.cache_data(show_spinner=False)
    def _execute_pipeline_cache(df_bytes):
        df = pd.read_csv(io.BytesIO(df_bytes))
        clean = clean_dataset(df)
        clean_tx = create_transaction_id(clean)
        baskets = create_baskets(clean_tx)
        dev_b, val_b = split_transactions(baskets, test_size=0.2, random_state=42)
        dev_enc, enc = encode_transactions(dev_b)
        val_enc, _ = encode_transactions(val_b, encoder=enc)
        raw_stats = get_dataset_statistics(df)
        bsk_stats = get_transaction_statistics(baskets)
        return clean, baskets, dev_b, val_b, dev_enc, val_enc, enc, raw_stats, bsk_stats

    df_bytes = raw_df.to_csv(index=False).encode()
    (clean_df, baskets_df, dev_baskets, val_baskets,
     dev_encoded, val_encoded, encoder,
     raw_stats, basket_stats) = _execute_pipeline_cache(df_bytes)

    # Initialize Mining Session States
    if "freq_dev" not in st.session_state:
        st.session_state.freq_dev = pd.DataFrame()
        st.session_state.rules_df = pd.DataFrame()
        st.session_state.t_time = 0.0
        st.session_state.min_sup = 0.001
        st.session_state.min_conf = 0.05

    if "fpg_freq" not in st.session_state:
        st.session_state.fpg_freq = pd.DataFrame()
        st.session_state.fpg_rules = pd.DataFrame()
        st.session_state.fpg_time = 0.0

    freq_dev  = st.session_state.freq_dev
    rules_df  = st.session_state.rules_df
    t_time    = st.session_state.t_time
    min_sup   = st.session_state.min_sup
    min_conf  = st.session_state.min_conf
    fpg_freq  = st.session_state.fpg_freq
    fpg_rules = st.session_state.fpg_rules
    fpg_time  = st.session_state.fpg_time

    mining_done = not freq_dev.empty
else:
    clean_df = baskets_df = dev_baskets = val_baskets = None
    dev_encoded = val_encoded = encoder = None
    raw_stats = basket_stats = None
    freq_dev = rules_df = fpg_freq = fpg_rules = pd.DataFrame()
    t_time = fpg_time = 0.0
    min_sup = 0.001
    min_conf = 0.05
    mining_done = False


def require_dataset():
    if raw_df is None:
        st.info("👆 Please select or upload a dataset in **📁 Dataset Management** to view this section.")
        st.stop()


# =====================================================================
# 1. PAGE: 📊 DASHBOARD
# =====================================================================
if current_page == "Dashboard":
    render_page_header("Executive Dashboard", "High-level overview of dataset health, transaction patterns, and mining activity.")
    require_dataset()

    k1, k2, k3, k4, k5, k6 = st.columns(6)
    with k1: st.markdown(kpi(f"{basket_stats['num_transactions']:,}", "Transactions", "Total Baskets", accent_blue), unsafe_allow_html=True)
    with k2: st.markdown(kpi(f"{raw_stats['unique_products']:,}", "Unique Items", "Catalog Products", accent_teal), unsafe_allow_html=True)
    with k3: st.markdown(kpi(f"{raw_stats['unique_customers']:,}", "Customers", "Members Tracked", accent_purple), unsafe_allow_html=True)
    with k4: st.markdown(kpi(f"{basket_stats['avg_basket_size']:.2f}", "Avg Basket Size", "Items/Transaction", accent_orange), unsafe_allow_html=True)
    with k5: st.markdown(kpi(f"{len(freq_dev):,}" if mining_done else "—", "Frequent Itemsets", "Apriori Found", accent_green), unsafe_allow_html=True)
    with k6: st.markdown(kpi(f"{len(rules_df):,}" if mining_done else "—", "Association Rules", "Active Rules", accent_blue), unsafe_allow_html=True)

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

    # Dataset Information & Summary Metrics
    c_info1, c_info2 = st.columns([0.45, 0.55])
    with c_info1:
        st.markdown(f"""
        <div style="background:{bg_card};border:1px solid {border_card};border-radius:12px;padding:18px;height:100%;">
            <h4 style="margin:0 0 12px 0;color:{text_primary};font-size:1.05rem;font-weight:700;border-bottom:1px solid {border_card};padding-bottom:8px;">
                📋 Active Dataset Information
            </h4>
            <div style="display:flex;justify-content:space-between;margin-bottom:8px;">
                <span style="color:{text_muted};">Source:</span>
                <strong style="color:{text_primary};">{active_dataset_label}</strong>
            </div>
            <div style="display:flex;justify-content:space-between;margin-bottom:8px;">
                <span style="color:{text_muted};">Raw Rows Ingested:</span>
                <strong style="color:{accent_blue};">{raw_stats['raw_rows']:,}</strong>
            </div>
            <div style="display:flex;justify-content:space-between;margin-bottom:8px;">
                <span style="color:{text_muted};">Exact Duplicates Cleaned:</span>
                <strong style="color:{accent_teal};">{raw_stats['exact_duplicate_rows']:,}</strong>
            </div>
            <div style="display:flex;justify-content:space-between;margin-bottom:8px;">
                <span style="color:{text_muted};">Development Split (80%):</span>
                <strong style="color:{text_primary};">{len(dev_baskets):,} baskets</strong>
            </div>
            <div style="display:flex;justify-content:space-between;">
                <span style="color:{text_muted};">Validation Split (20%):</span>
                <strong style="color:{accent_orange};">{len(val_baskets):,} baskets</strong>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with c_info2:
        st.markdown(f"""
        <div style="background:{bg_card};border:1px solid {border_card};border-radius:12px;padding:18px;height:100%;">
            <h4 style="margin:0 0 12px 0;color:{text_primary};font-size:1.05rem;font-weight:700;border-bottom:1px solid {border_card};padding-bottom:8px;">
                ⚡ Recent Mining Activity
            </h4>
            <div style="display:flex;justify-content:space-between;margin-bottom:8px;">
                <span style="color:{text_muted};">Apriori Baseline Status:</span>
                <span class="{'badge-ok' if mining_done else 'badge-warn'}">{'● Complete' if mining_done else '○ Not Run Yet'}</span>
            </div>
            <div style="display:flex;justify-content:space-between;margin-bottom:8px;">
                <span style="color:{text_muted};">Applied Min Support:</span>
                <strong style="color:{accent_blue};">{min_sup}</strong>
            </div>
            <div style="display:flex;justify-content:space-between;margin-bottom:8px;">
                <span style="color:{text_muted};">Applied Min Confidence:</span>
                <strong style="color:{accent_blue};">{min_conf}</strong>
            </div>
            <div style="display:flex;justify-content:space-between;margin-bottom:8px;">
                <span style="color:{text_muted};">Apriori Execution Time:</span>
                <strong style="color:{text_primary};">{t_time:.4f} s</strong>
            </div>
            <div style="display:flex;justify-content:space-between;">
                <span style="color:{text_muted};">FP-Growth Execution Time:</span>
                <strong style="color:{accent_green};">{fpg_time:.4f} s</strong>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

    # Average Rule Metrics Card (if rules exist)
    if mining_done and not rules_df.empty:
        st.subheader("📈 Average Association Rule Metrics")
        rm1, rm2, rm3, rm4 = st.columns(4)
        rm1.metric("Avg Support", f"{rules_df['support'].mean():.4f}")
        rm2.metric("Avg Confidence", f"{rules_df['confidence'].mean():.4f}")
        rm3.metric("Avg Lift", f"{rules_df['lift'].mean():.4f}")
        rm4.metric("Max Lift", f"{rules_df['lift'].max():.4f}")
        st.markdown("<br>", unsafe_allow_html=True)

    # Summary Charts
    st.subheader("📊 Dataset Distribution Charts")
    ch1, ch2 = st.columns(2)
    with ch1:
        top15 = clean_df['itemDescription'].value_counts().head(15).reset_index()
        top15.columns = ["Item", "Purchases"]
        fig1 = px.bar(top15.sort_values("Purchases"), x="Purchases", y="Item", orientation="h",
                      color="Purchases", color_continuous_scale="Blues",
                      title="Top 15 Most Purchased Grocery Items", template=PLOTLY_TEMPLATE)
        fig1 = apply_plotly_theme(fig1, 400)
        fig1.update_coloraxes(showscale=False)
        st.plotly_chart(fig1, use_container_width=True)

    with ch2:
        bsizes = baskets_df['items'].apply(len).value_counts().sort_index().reset_index()
        bsizes.columns = ["Basket Size", "Transactions"]
        fig2 = px.bar(bsizes, x="Basket Size", y="Transactions", color="Transactions",
                      color_continuous_scale="Teal",
                      title="Basket Size Distribution (Items per Basket)", template=PLOTLY_TEMPLATE)
        fig2 = apply_plotly_theme(fig2, 400)
        fig2.update_coloraxes(showscale=False)
        st.plotly_chart(fig2, use_container_width=True)

    if mining_done and not rules_df.empty:
        st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
        st.subheader("🔬 Rule Exploration Scatter Plot")
        disp_r = rules_df.copy()
        disp_r['Rule'] = disp_r.apply(lambda r: f"{', '.join(sorted(r['antecedents']))} → {', '.join(sorted(r['consequents']))}", axis=1)
        fig3 = px.scatter(disp_r, x="support", y="confidence", color="lift",
                          hover_data=["Rule", "lift", "leverage"], color_continuous_scale="Viridis",
                          title="Discovered Rules: Support vs Confidence (Color = Lift)", template=PLOTLY_TEMPLATE)
        fig3 = apply_plotly_theme(fig3, 380)
        st.plotly_chart(fig3, use_container_width=True)

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
    st.subheader("ℹ️ About Project: Scalable Association Rule Mining and Estimator")
    st.markdown(f"""
    <div style="background:{bg_card};border:1px solid {border_card};border-radius:12px;padding:18px;">
        <h4 style="margin:0 0 8px 0;color:{text_primary};font-size:1.05rem;font-weight:700;">
            Scalable Association Rule Mining and Estimator
        </h4>
        <p style="color:{text_secondary};font-size:0.88rem;line-height:1.6;margin-bottom:10px;">
            <strong>Scalable Association Rule Mining and Estimator</strong> is an end-to-end transaction pattern mining, rule evaluation, and decision-support platform. 
            The system provides high-performance frequent itemset discovery using both Apriori candidate generation and FP-Growth pattern-tree mining, data-driven adaptive parameter recommendations, formal rule redundancy pruning, holdout validation on unseen transactions, and automated translation of association rules into retail business insights.
        </p>
        <div style="display:flex;gap:20px;font-size:0.8rem;color:{text_muted};flex-wrap:wrap;">
            <span>⚡ <strong>Algorithms:</strong> Apriori & FP-Growth</span>
            <span>🎯 <strong>Parameters:</strong> Adaptive Heuristic Optimization</span>
            <span>🔍 <strong>Evaluation:</strong> Quality, Redundancy & Generalization Validation</span>
            <span>🤖 <strong>Intelligence:</strong> Actionable Business Recommendations</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


# =====================================================================
# 2. PAGE: 📁 DATASET MANAGEMENT
# =====================================================================
elif current_page == "Dataset Management":
    render_page_header("Dataset Management", "Upload custom transaction CSVs, inspect dataset health, or switch to benchmark datasets.")

    st.subheader("🔄 Select Active Dataset Source")
    ds_options = ["Default Dataset (Groceries)", "Uploaded CSV Dataset"]
    curr_idx = ds_options.index(st.session_state.dataset_selection) if st.session_state.dataset_selection in ds_options else 0
    selected_ds = st.radio("Choose Dataset to Analyze:", ds_options, index=curr_idx, horizontal=True)

    if selected_ds != st.session_state.dataset_selection:
        st.session_state.dataset_selection = selected_ds
        st.rerun()

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

    # Upload Section
    st.subheader("📤 Upload Custom CSV Dataset")
    new_uploaded_file = st.file_uploader(
        "Upload transaction CSV (Columns required: Member_number, Date, itemDescription)",
        type=["csv"], key="uploader_main"
    )

    if new_uploaded_file is not None:
        file_bytes = new_uploaded_file.getvalue()
        if st.session_state.uploaded_file_bytes != file_bytes:
            st.session_state.uploaded_file_bytes = file_bytes
            st.session_state.uploaded_file_name = new_uploaded_file.name
            st.session_state.dataset_selection = "Uploaded CSV Dataset"
            # Invalidate previous mining state for the new file
            st.session_state.freq_dev = pd.DataFrame()
            st.session_state.rules_df = pd.DataFrame()
            st.session_state.fpg_freq = pd.DataFrame()
            st.session_state.fpg_rules = pd.DataFrame()
            st.success(f"✅ Successfully loaded `{new_uploaded_file.name}` and set as active dataset!")
            st.rerun()

    require_dataset()

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
    st.subheader("📋 Dataset Health & Health Metrics")
    dh1, dh2, dh3, dh4, dh5 = st.columns(5)
    dh1.metric("Total Records", f"{raw_stats['raw_rows']:,}")
    dh2.metric("Total Columns", raw_stats['raw_columns'])
    dh3.metric("Missing Values", raw_stats['total_missing_values'], delta="Clean" if raw_stats['total_missing_values']==0 else "Nulls Handled")
    dh4.metric("Exact Duplicates", f"{raw_stats['exact_duplicate_rows']:,}", delta="Removed" if raw_stats['exact_duplicate_rows']>0 else "None")
    dh5.metric("Clean Records", f"{len(clean_df):,}")

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
    st.subheader("🔎 Raw Dataset Preview (First 20 Records)")
    st.dataframe(raw_df.head(20), use_container_width=True)

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
    st.subheader("🏷 Column Schema & Structure")
    schema_info = []
    for col in raw_df.columns:
        schema_info.append({
            "Column Name": col,
            "Data Type": str(raw_df[col].dtype),
            "Non-Null Count": int(raw_df[col].notna().sum()),
            "Missing Count": int(raw_df[col].isna().sum()),
            "Unique Count": int(raw_df[col].nunique()),
            "Sample Value": str(raw_df[col].iloc[0]) if len(raw_df) > 0 else ""
        })
    st.dataframe(pd.DataFrame(schema_info), use_container_width=True)


# =====================================================================
# 3. PAGE: ⚙️ DATA PREPROCESSING
# =====================================================================
elif current_page == "Data Preprocessing":
    render_page_header("Data Preprocessing Pipeline", "Inspect data cleaning, deduplication, basket formation, and one-hot binary encoding.")
    require_dataset()

    tab_pipe, tab_baskets, tab_matrix = st.tabs([
        "🔄 Pipeline Operations & Summary",
        "🧺 Cleaned Transaction Baskets",
        "📊 One-Hot Encoded Matrix"
    ])

    with tab_pipe:
        st.subheader("Step-by-Step Preprocessing Flow")
        stages_flow = [
            ("1. CSV Ingestion", f"{raw_stats['raw_rows']:,} rows", "Schema checked (`Member_number`, `Date`, `itemDescription`)"),
            ("2. Deduplication", f"-{raw_stats['exact_duplicate_rows']:,} rows", "Exact row duplicates removed"),
            ("3. Null Handling", f"{raw_stats['total_missing_values']} missing", "Checked and cleaned"),
            ("4. Transaction ID", f"{basket_stats['num_transactions']:,} IDs", "Composite `Member_number_Date` key created"),
            ("5. Basket Aggregation", f"Avg {basket_stats['avg_basket_size']:.2f} items", "Inner-basket duplicates removed"),
            ("6. Binary Encoding", f"{dev_encoded.shape[0]}×{dev_encoded.shape[1]}", "One-hot transaction-item boolean matrix"),
            ("7. 80/20 Train Split", "11,970 / 2,993", "Transaction-level reproducible split"),
        ]
        cols = st.columns(len(stages_flow))
        for col, (title, stat, desc) in zip(cols, stages_flow):
            with col:
                st.markdown(f"""
                <div style="background:{bg_card};border:1px solid {border_card};border-radius:10px;padding:12px 8px;text-align:center;">
                    <strong style="font-size:0.8rem;color:{text_primary};">{title}</strong><br/>
                    <span style="font-size:1.05rem;font-weight:800;color:{accent_blue};display:block;margin:6px 0;">{stat}</span>
                    <small style="color:{text_muted};">{desc}</small><br/><br/>
                    <span class="badge-ok">✔ Done</span>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
        st.subheader("📊 Before & After Cleaning Comparison")
        comp_df = pd.DataFrame([
            {"Metric": "Total Records / Rows", "Before Preprocessing": f"{raw_stats['raw_rows']:,}", "After Preprocessing": f"{len(clean_df):,}"},
            {"Metric": "Exact Duplicate Rows", "Before Preprocessing": f"{raw_stats['exact_duplicate_rows']:,}", "After Preprocessing": "0 (Deduplicated)"},
            {"Metric": "Missing Value Rows", "Before Preprocessing": str(raw_stats['total_missing_values']), "After Preprocessing": "0 (Clean)"},
            {"Metric": "Representation Format", "Before Preprocessing": "Record-level purchase log", "After Preprocessing": "Transaction-level item baskets"},
            {"Metric": "Total Unique Baskets", "Before Preprocessing": "N/A", "After Preprocessing": f"{basket_stats['num_transactions']:,}"},
            {"Metric": "Development Set (80%)", "Before Preprocessing": "N/A", "After Preprocessing": f"{len(dev_baskets):,} transactions"},
            {"Metric": "Validation Set (20%)", "Before Preprocessing": "N/A", "After Preprocessing": f"{len(val_baskets):,} transactions"},
        ])
        st.dataframe(comp_df, use_container_width=True)

    with tab_baskets:
        st.subheader("🧺 Transaction Baskets (Sample of 15)")
        bpreview = baskets_df.head(15).copy()
        bpreview['item_count'] = bpreview['items'].apply(len)
        bpreview['items_str'] = bpreview['items'].apply(lambda x: ", ".join(x))
        st.dataframe(bpreview[['Transaction_ID', 'item_count', 'items_str']].rename(
            columns={'item_count': 'Items in Basket', 'items_str': 'Item Basket Set'}), use_container_width=True)

        st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
        st.subheader("📈 Basket Size Distribution Histogram")
        bs_dist = baskets_df['items'].apply(len).value_counts().sort_index().reset_index()
        bs_dist.columns = ["Basket Size", "Number of Baskets"]
        fig_bs = px.bar(bs_dist, x="Basket Size", y="Number of Baskets", color="Number of Baskets",
                        color_continuous_scale="Teal", template=PLOTLY_TEMPLATE)
        fig_bs = apply_plotly_theme(fig_bs, 350)
        fig_bs.update_coloraxes(showscale=False)
        st.plotly_chart(fig_bs, use_container_width=True)

    with tab_matrix:
        st.subheader("📊 One-Hot Encoded Boolean Matrix (Development Set)")
        st.caption(f"Matrix Dimensions: {dev_encoded.shape[0]:,} transactions (rows) × {dev_encoded.shape[1]} unique items (columns).")
        st.dataframe(dev_encoded.iloc[:15, :25].astype(int), use_container_width=True)


# =====================================================================
# 4. PAGE: 🎯 ADAPTIVE PARAMETERS
# =====================================================================
elif current_page == "Adaptive Parameters":
    render_page_header("Adaptive Parameter Recommendation", "Data-driven heuristic recommendation for minimum support and confidence thresholds.")
    require_dataset()

    rec = recommend_parameters(baskets_df, dev_fraction=0.8)

    st.subheader("🔍 Dataset Characteristics Analyzed")
    ac1, ac2, ac3, ac4 = st.columns(4)
    ac1.metric("Total Transactions", f"{rec['dataset_summary']['total_transactions']:,}")
    ac2.metric("Dev Set (80%)", f"{rec['dataset_summary']['dev_transactions']:,}")
    ac3.metric("Unique Products", f"{rec['dataset_summary']['unique_items']:,}")
    ac4.metric("Avg Basket Size", f"{rec['dataset_summary']['avg_basket_size']:.3f} items")

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
    st.subheader("✨ Recommended vs Selected Thresholds")
    r_col1, r_col2 = st.columns(2)

    with r_col1:
        st.markdown(f"""
        <div style="background:{bg_card};border:1px solid {border_card};border-left:4px solid {accent_blue};border-radius:10px;padding:20px;">
            <div style="display:flex;justify-content:space-between;align-items:center;">
                <h4 style="margin:0;color:{accent_blue};font-weight:700;">Recommended Min Support</h4>
                <span class="badge-ok">Recommended: {rec['min_support']}</span>
            </div>
            <p style="font-size:2.6rem;font-weight:800;color:{text_primary};margin:8px 0;">{rec['min_support']}</p>
            <p style="font-size:0.85rem;color:{text_muted};line-height:1.5;">{rec['explanation']['min_support']}</p>
        </div>
        """, unsafe_allow_html=True)

    with r_col2:
        st.markdown(f"""
        <div style="background:{bg_card};border:1px solid {border_card};border-left:4px solid {accent_teal};border-radius:10px;padding:20px;">
            <div style="display:flex;justify-content:space-between;align-items:center;">
                <h4 style="margin:0;color:{accent_teal};font-weight:700;">Recommended Min Confidence</h4>
                <span class="badge-ok">Recommended: {rec['min_confidence']}</span>
            </div>
            <p style="font-size:2.6rem;font-weight:800;color:{text_primary};margin:8px 0;">{rec['min_confidence']}</p>
            <p style="font-size:0.85rem;color:{text_muted};line-height:1.5;">{rec['explanation']['min_confidence']}</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
    st.subheader("⚙️ Customize & Apply Mining Parameters")
    st.info("💡 You can accept the system's recommended values or customize them below. Clicking **Apply Parameters & Execute Mining** will immediately run both Apriori and FP-Growth.")

    p_col1, p_col2 = st.columns(2)
    user_sup = p_col1.number_input(
        "Applied Minimum Support (min_support)",
        min_value=0.0001, max_value=0.5, value=float(min_sup), step=0.0005, format="%.4f"
    )
    user_conf = p_col2.number_input(
        "Applied Minimum Confidence (min_confidence)",
        min_value=0.01, max_value=1.0, value=float(min_conf), step=0.01, format="%.2f"
    )

    if st.button("🚀 Apply Parameters & Execute Mining (Apriori + FP-Growth)", type="primary"):
        with st.spinner("Running Apriori baseline algorithm..."):
            fd, tt, _ = measure_performance(run_apriori, dev_encoded, min_support=user_sup)
            rd = generate_rules(fd, min_confidence=user_conf, sort_by="lift")
            st.session_state.freq_dev = fd
            st.session_state.rules_df = rd
            st.session_state.t_time = tt
            st.session_state.min_sup = user_sup
            st.session_state.min_conf = user_conf

        with st.spinner("Running FP-Growth pattern mining algorithm..."):
            ff, ft, _ = measure_performance(run_fpgrowth, dev_encoded, min_support=user_sup)
            fr = generate_fpgrowth_rules(ff, min_confidence=user_conf, sort_by="lift")
            st.session_state.fpg_freq = ff
            st.session_state.fpg_rules = fr
            st.session_state.fpg_time = ft

        st.success(f"✅ Success! Discovered {len(fd):,} frequent itemsets and {len(rd):,} rules in {tt:.3f}s (Apriori) and {ft:.3f}s (FP-Growth).")
        st.rerun()

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
    st.markdown("""
    > [!NOTE]
    > **Methodology Note**: The adaptive recommendation evaluates transaction count, product space, and average basket size.
    > The goal is to ensure every discovered itemset appears in at least ~30 actual transactions to prevent statistical noise while avoiding over-filtering in sparse retail data.
    """)


# =====================================================================
# 5. PAGE: ⚡ ASSOCIATION RULE MINING
# =====================================================================
elif current_page == "Association Rule Mining":
    render_page_header("Association Rule Mining Engine", "Discover frequent itemsets and association rules using Apriori and FP-Growth.")
    require_dataset()

    tab_apr, tab_freq, tab_rules, tab_fpg = st.tabs([
        "⚡ Tab 1: Apriori Mining",
        "◈ Tab 2: Frequent Itemsets",
        "🔗 Tab 3: Association Rules",
        "🚀 Tab 4: FP-Growth Mining"
    ])

    # ── TAB 1: Apriori Mining ─────────────────────────────────────────
    with tab_apr:
        st.subheader("⚡ Run Apriori Baseline")
        apr_c1, apr_c2, apr_c3 = st.columns(3)
        apr_sup = apr_c1.number_input("Min Support", 0.0001, 0.5, float(min_sup), 0.0005, format="%.4f", key="apr_sup_tab")
        apr_conf = apr_c2.number_input("Min Confidence", 0.01, 1.0, float(min_conf), 0.01, format="%.2f", key="apr_conf_tab")
        apr_sort = apr_c3.selectbox("Sort Rules By", ["lift", "confidence", "support", "leverage"], key="apr_sort_tab")

        if st.button("▶ Run Apriori Baseline", type="primary", key="btn_run_apr_tab"):
            with st.spinner("Executing Apriori algorithm on Development Set..."):
                fd, tt, _ = measure_performance(run_apriori, dev_encoded, min_support=apr_sup)
                rd = generate_rules(fd, min_confidence=apr_conf, sort_by=apr_sort)
                st.session_state.freq_dev = fd
                st.session_state.rules_df = rd
                st.session_state.t_time = tt
                st.session_state.min_sup = apr_sup
                st.session_state.min_conf = apr_conf
            st.success(f"✅ Apriori execution completed in {tt:.4f} seconds!")
            st.rerun()

        st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
        st.subheader("📊 Apriori Execution Status & Summary")
        as1, as2, as3, as4 = st.columns(4)
        as1.metric("Execution Status", "Complete" if mining_done else "Ready")
        as2.metric("Frequent Itemsets", f"{len(freq_dev):,}")
        as3.metric("Association Rules", f"{len(rules_df):,}")
        as4.metric("Execution Time", f"{t_time:.4f} s")

        st.markdown(f"""
        - **Algorithm**: Standard Apriori Baseline (`mlxtend.frequent_patterns.apriori`)
        - **Search Strategy**: Level-wise candidate itemset generation and join/prune cycles
        - **Target Dataset**: 80% Development Set ({len(dev_baskets):,} transactions)
        """)

    # ── TAB 2: Frequent Itemsets ──────────────────────────────────────
    with tab_freq:
        st.subheader("◈ Frequent Itemsets Discovered (Apriori)")
        if freq_dev.empty:
            st.info("Run Apriori in Tab 1 to generate frequent itemsets.")
        else:
            df_itemsets = freq_dev.copy()
            df_itemsets['Itemset'] = df_itemsets['itemsets'].apply(lambda x: ", ".join(sorted(x)))

            fi_c1, fi_c2 = st.columns([0.6, 0.4])
            search_item = fi_c1.text_input("🔍 Search Item within Itemset", "", key="search_item_tab")
            max_len = int(df_itemsets['itemset_len'].max()) if not df_itemsets.empty else 1
            filter_len = fi_c2.slider("Max Itemset Length", 1, max(max_len, 2), max(max_len, 1), key="filter_len_tab")

            df_fi_show = df_itemsets[df_itemsets['itemset_len'] <= filter_len]
            if search_item:
                df_fi_show = df_fi_show[df_fi_show['Itemset'].str.contains(search_item, case=False)]

            st.dataframe(df_fi_show[['Itemset', 'support', 'itemset_len']].rename(
                columns={'support': 'Support', 'itemset_len': 'Size'}), use_container_width=True)
            st.caption(f"Displaying {len(df_fi_show):,} of {len(freq_dev):,} discovered frequent itemsets.")

            st.download_button(
                "⬇ Download Frequent Itemsets CSV",
                df_fi_show[['Itemset', 'support', 'itemset_len']].to_csv(index=False).encode(),
                "frequent_itemsets.csv", "text/csv"
            )

            st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
            st.subheader("📈 Top 15 Frequent Itemsets by Support")
            top15_fi = df_itemsets.nlargest(15, "support")
            fig_fi = px.bar(top15_fi.sort_values("support"), x="support", y="Itemset", orientation="h",
                            color="support", color_continuous_scale="Blues", template=PLOTLY_TEMPLATE)
            fig_fi = apply_plotly_theme(fig_fi, 450)
            fig_fi.update_coloraxes(showscale=False)
            st.plotly_chart(fig_fi, use_container_width=True)

    # ── TAB 3: Association Rules ──────────────────────────────────────
    with tab_rules:
        st.subheader("🔗 Generated Association Rules (Apriori)")
        if rules_df.empty:
            st.info("Run Apriori in Tab 1 to generate association rules.")
        else:
            df_r = rules_df.copy()
            df_r['Antecedents'] = df_r['antecedents'].apply(lambda x: ", ".join(sorted(x)))
            df_r['Consequents'] = df_r['consequents'].apply(lambda x: ", ".join(sorted(x)))

            r_f1, r_f2, r_f3 = st.columns(3)
            search_rule = r_f1.text_input("🔍 Search Antecedent or Consequent", "", key="search_rule_tab")
            min_lift_filter = r_f2.slider("Filter Min Lift", 0.5, float(max(df_r['lift'].max(), 1.5)), 1.0, 0.1, key="slider_lift_tab")
            min_conf_filter = r_f3.slider("Filter Min Confidence", 0.0, 1.0, 0.0, 0.01, key="slider_conf_tab")

            df_r_filtered = df_r[(df_r['lift'] >= min_lift_filter) & (df_r['confidence'] >= min_conf_filter)]
            if search_rule:
                df_r_filtered = df_r_filtered[
                    df_r_filtered['Antecedents'].str.contains(search_rule, case=False) |
                    df_r_filtered['Consequents'].str.contains(search_rule, case=False)
                ]

            st.dataframe(df_r_filtered[['Antecedents', 'Consequents', 'support', 'confidence', 'lift', 'leverage']].rename(
                columns={'support': 'Support', 'confidence': 'Confidence', 'lift': 'Lift', 'leverage': 'Leverage'}),
                use_container_width=True
            )
            st.caption(f"Showing {len(df_r_filtered):,} rules after filters.")

            st.download_button(
                "⬇ Download Association Rules CSV",
                df_r_filtered[['Antecedents', 'Consequents', 'support', 'confidence', 'lift', 'leverage']].to_csv(index=False).encode(),
                "association_rules.csv", "text/csv"
            )

    # ── TAB 4: FP-Growth Mining ───────────────────────────────────────
    with tab_fpg:
        st.subheader("🚀 FP-Growth Frequent Pattern Mining")
        fpg_c1, fpg_c2, fpg_c3 = st.columns(3)
        fpg_s = fpg_c1.number_input("Min Support", 0.0001, 0.5, float(min_sup), 0.0005, format="%.4f", key="fpg_s_tab")
        fpg_c = fpg_c2.number_input("Min Confidence", 0.01, 1.0, float(min_conf), 0.01, format="%.2f", key="fpg_c_tab")
        fpg_srt = fpg_c3.selectbox("Sort Rules By", ["lift", "confidence", "support", "leverage"], key="fpg_srt_tab")

        if st.button("▶ Run FP-Growth", type="primary", key="btn_run_fpg_tab"):
            with st.spinner("Executing FP-Growth tree-based mining..."):
                ff, ft, _ = measure_performance(run_fpgrowth, dev_encoded, min_support=fpg_s)
                fr = generate_fpgrowth_rules(ff, min_confidence=fpg_c, sort_by=fpg_srt)
                st.session_state.fpg_freq = ff
                st.session_state.fpg_rules = fr
                st.session_state.fpg_time = ft
            st.success(f"✅ FP-Growth completed in {ft:.4f} seconds! Discovered {len(ff):,} itemsets.")
            st.rerun()

        st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
        st.subheader("📊 FP-Growth Execution Status")
        fs1, fs2, fs3 = st.columns(3)
        fs1.metric("Frequent Itemsets", f"{len(fpg_freq):,}")
        fs2.metric("Association Rules", f"{len(fpg_rules):,}")
        fs3.metric("Execution Time", f"{fpg_time:.4f} s")

        if not fpg_rules.empty:
            st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
            st.subheader("📋 FP-Growth Discovered Rules")
            df_fpg_disp = fpg_rules.copy()
            df_fpg_disp['Antecedents'] = df_fpg_disp['antecedents'].apply(lambda x: ", ".join(sorted(x)))
            df_fpg_disp['Consequents'] = df_fpg_disp['consequents'].apply(lambda x: ", ".join(sorted(x)))
            st.dataframe(df_fpg_disp[['Antecedents', 'Consequents', 'support', 'confidence', 'lift', 'leverage']].rename(
                columns={'support': 'Support', 'confidence': 'Confidence', 'lift': 'Lift', 'leverage': 'Leverage'}),
                use_container_width=True
            )
            st.download_button(
                "⬇ Download FP-Growth Rules CSV",
                df_fpg_disp[['Antecedents', 'Consequents', 'support', 'confidence', 'lift', 'leverage']].to_csv(index=False).encode(),
                "fpgrowth_rules.csv", "text/csv"
            )


# =====================================================================
# 6. PAGE: 🔍 RULE EVALUATION
# =====================================================================
elif current_page == "Rule Evaluation":
    render_page_header("Rule Evaluation Suite", "Rank association rules, filter by quality metrics, detect redundancy, and validate on unseen data.")
    require_dataset()

    if rules_df.empty:
        st.warning("⚠️ No association rules available. Please run Apriori in **Association Rule Mining** first.")
        st.stop()

    tab_rank, tab_qual, tab_val = st.tabs([
        "🏆 Tab 1: Rule Ranking",
        "🧹 Tab 2: Quality & Redundancy Reduction",
        "✔ Tab 3: Unsupervised Validation"
    ])

    # ── TAB 1: Rule Ranking ───────────────────────────────────────────
    with tab_rank:
        st.subheader("🏆 Multi-Metric Rule Ranking")
        rk1, rk2 = st.columns([0.5, 0.5])
        rank_metric = rk1.selectbox("Primary Ranking Metric", ["lift", "confidence", "support", "leverage"], key="rk_met")
        top_k = rk2.number_input("Display Top K Rules", 5, min(len(rules_df), 100), min(len(rules_df), 20), 5, key="top_k_rk")

        ranked_df = rank_rules(rules_df, sort_by=rank_metric)
        top_ranked = ranked_df.head(int(top_k)).copy()
        top_ranked['Antecedents'] = top_ranked['antecedents'].apply(lambda x: ", ".join(sorted(x)))
        top_ranked['Consequents'] = top_ranked['consequents'].apply(lambda x: ", ".join(sorted(x)))
        top_ranked['Rule'] = top_ranked['Antecedents'] + " → " + top_ranked['Consequents']

        fig_rk = px.bar(top_ranked.sort_values(rank_metric), x=rank_metric, y="Rule", orientation="h",
                        color=rank_metric, color_continuous_scale="Viridis",
                        title=f"Top {top_k} Rules Ranked by {rank_metric.capitalize()}", template=PLOTLY_TEMPLATE)
        fig_rk = apply_plotly_theme(fig_rk, max(380, int(top_k)*22))
        fig_rk.update_coloraxes(showscale=False)
        st.plotly_chart(fig_rk, use_container_width=True)

        st.dataframe(top_ranked[['rank', 'Antecedents', 'Consequents', 'support', 'confidence', 'lift', 'leverage']].rename(
            columns={'rank': 'Rank', 'support': 'Support', 'confidence': 'Confidence', 'lift': 'Lift', 'leverage': 'Leverage'}),
            use_container_width=True
        )

        st.download_button(
            "⬇ Download Ranked Rules CSV",
            top_ranked[['rank', 'Antecedents', 'Consequents', 'support', 'confidence', 'lift', 'leverage']].to_csv(index=False).encode(),
            "ranked_rules.csv", "text/csv"
        )

    # ── TAB 2: Quality & Redundancy Reduction ─────────────────────────
    with tab_qual:
        st.subheader("🧹 Rule Quality & Strict Redundancy Pruning")
        q_c1, q_c2, q_c3, q_c4 = st.columns(4)
        q_sup = q_c1.number_input("Min Support Filter", 0.0, 1.0, 0.0, 0.001, format="%.4f", key="qf_sup")
        q_conf = q_c2.number_input("Min Confidence Filter", 0.0, 1.0, 0.0, 0.01, format="%.2f", key="qf_conf")
        q_lift = q_c3.number_input("Min Lift Filter", 0.0, float(rules_df['lift'].max())+1, 1.0, 0.1, format="%.2f", key="qf_lift")
        q_lev = q_c4.number_input("Min Leverage Filter", float(rules_df['leverage'].min()), float(rules_df['leverage'].max()), 0.0, 0.001, format="%.4f", key="qf_lev")

        summary = get_quality_summary(rules_df, q_sup, q_conf, q_lift, q_lev)

        st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
        qs1, qs2, qs3, qs4, qs5 = st.columns(5)
        qs1.metric("Original Rules", f"{summary['total_rules']:,}")
        qs2.metric("After Quality Filter", f"{summary['after_filter']:,}")
        qs3.metric("Non-Redundant Rules", f"{summary['non_redundant_rules']:,}")
        qs4.metric("Redundant Rules", f"{summary['redundant_rules']:,}")
        qs5.metric("% Redundant", f"{summary['pct_redundant']:.1f}%")

        # Before & After Reduction Chart
        ba_fig = go.Figure(data=[
            go.Bar(name="Initial Rules", x=["Rules"], y=[summary['total_rules']], marker_color=accent_blue),
            go.Bar(name="Filtered Rules", x=["Rules"], y=[summary['after_filter']], marker_color=accent_teal),
            go.Bar(name="Non-Redundant Rules", x=["Rules"], y=[summary['non_redundant_rules']], marker_color=accent_purple),
        ])
        ba_fig.update_layout(barmode='group', title="Rule Counts: Filtered vs Non-Redundant Retained", template=PLOTLY_TEMPLATE)
        ba_fig = apply_plotly_theme(ba_fig, 300)
        st.plotly_chart(ba_fig, use_container_width=True)

        filtered = filter_rules(rules_df, q_sup, q_conf, q_lift, q_lev)
        ranked = rank_rules(filtered, sort_by="lift")
        annotated = identify_redundant_rules(ranked)

        if not annotated.empty:
            disp_annot = annotated.copy()
            disp_annot['Antecedents'] = disp_annot['antecedents'].apply(lambda x: ", ".join(sorted(x)))
            disp_annot['Consequents'] = disp_annot['consequents'].apply(lambda x: ", ".join(sorted(x)))
            cols_disp = ['rank', 'Antecedents', 'Consequents', 'support', 'confidence', 'lift', 'leverage', 'is_redundant', 'redundant_reason']

            sub_all, sub_nr, sub_rd = st.tabs([
                f"All Filtered Rules ({len(disp_annot)})",
                f"Non-Redundant Retained ({summary['non_redundant_rules']})",
                f"Redundant Removed ({summary['redundant_rules']})"
            ])
            with sub_all:
                st.dataframe(disp_annot[cols_disp], use_container_width=True)
            with sub_nr:
                nr_df = disp_annot[~disp_annot['is_redundant']]
                st.dataframe(nr_df[cols_disp], use_container_width=True)
                if not nr_df.empty:
                    st.download_button(
                        "⬇ Download Non-Redundant Rules CSV",
                        nr_df[['Antecedents', 'Consequents', 'support', 'confidence', 'lift', 'leverage']].to_csv(index=False).encode(),
                        "non_redundant_rules.csv", "text/csv"
                    )
            with sub_rd:
                rd_df = disp_annot[disp_annot['is_redundant']]
                if not rd_df.empty:
                    st.dataframe(rd_df[cols_disp], use_container_width=True)
                else:
                    st.info("No redundant rules identified under current thresholds.")

    # ── TAB 3: Unsupervised Validation ────────────────────────────────
    with tab_val:
        st.subheader("✔ Unsupervised Generalization Validation")
        st.markdown(f"""
        Evaluating rules mined on the **Development Set (80% — {len(dev_baskets):,} baskets)** against the separate **Validation Set (20% — {len(val_baskets):,} baskets)**.
        """)

        val_comp = evaluate_rules_on_validation(rules_df, val_encoded)
        val_comp["Rule"] = val_comp.apply(lambda r: f"{r['antecedents']} → {r['consequents']}", axis=1)

        v_s1, v_s2, v_s3, v_s4 = st.columns(4)
        v_s1.metric("Rules Evaluated", f"{len(val_comp):,}")
        v_s2.metric("Avg Δ Confidence", f"{val_comp['diff_confidence'].mean():.4f}")
        v_s3.metric("Avg Δ Lift", f"{val_comp['diff_lift'].mean():.4f}")
        stable_count = (abs(val_comp['diff_confidence']) <= 0.1).sum()
        v_s4.metric("Stable Rules (|Δ Conf| ≤ 0.10)", f"{stable_count:,} / {len(val_comp):,}")

        st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
        st.subheader("📋 Development vs Validation Performance Table")
        disp_val = val_comp[["Rule", "dev_support", "val_support", "diff_support",
                              "dev_confidence", "val_confidence", "diff_confidence",
                              "dev_lift", "val_lift", "diff_lift"]].rename(columns={
            "dev_support": "Dev Sup", "val_support": "Val Sup", "diff_support": "Δ Sup",
            "dev_confidence": "Dev Conf", "val_confidence": "Val Conf", "diff_confidence": "Δ Conf",
            "dev_lift": "Dev Lift", "val_lift": "Val Lift", "diff_lift": "Δ Lift"
        })
        st.dataframe(disp_val, use_container_width=True)

        st.download_button(
            "⬇ Download Validation Metrics CSV",
            disp_val.to_csv(index=False).encode(),
            "validation_comparison.csv", "text/csv"
        )

        st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
        st.subheader("📈 Generalization Error Distribution (Δ Confidence)")
        fig_diff = px.histogram(val_comp, x="diff_confidence", nbins=30,
                                color_discrete_sequence=[accent_blue],
                                title="Distribution of Δ Confidence (Dev Conf - Val Conf)", template=PLOTLY_TEMPLATE)
        fig_diff.add_vline(x=0, line_dash="dash", line_color=accent_orange)
        fig_diff = apply_plotly_theme(fig_diff, 320)
        st.plotly_chart(fig_diff, use_container_width=True)


# =====================================================================
# 7. PAGE: 🤖 AI BUSINESS INTELLIGENCE
# =====================================================================
elif current_page == "AI Business Intelligence":
    render_page_header("AI Business Intelligence Engine", "Transform discovered association rules into executive recommendations and actionable retail insights.")
    require_dataset()

    if rules_df.empty:
        st.warning("⚠️ No association rules found. Run Apriori in **Association Rule Mining** first.")
        st.stop()

    tab_ai, tab_bi = st.tabs([
        "🤖 Tab 1: AI-Assisted Recommendations",
        "💡 Tab 2: Business Insights Dashboard"
    ])

    insights = generate_insights(rules_df, top_n=30)

    # ── TAB 1: AI Recommendations ─────────────────────────────────────
    with tab_ai:
        st.subheader("🤖 Actionable Decision Support Recommendations")
        st.info("Rule-based AI translation: Every recommendation below is computed directly from actual rule co-occurrence, confidence, and lift.")

        if not insights:
            st.warning("No recommendations met current thresholds. Try adjusting support/confidence in Adaptive Parameters.")
        else:
            type_counts = {}
            for ins in insights:
                type_counts[ins['type']] = type_counts.get(ins['type'], 0) + 1

            tc_cols = st.columns(len(type_counts))
            icons = {"Product Placement": "🗂️", "Bundle Offer": "🎁", "Cross-Selling": "↗️", "Frequent Pattern": "📊"}
            for col, (tp, count) in zip(tc_cols, type_counts.items()):
                with col:
                    st.markdown(kpi(count, tp, icons.get(tp, "💡"), accent_blue), unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            groups = {}
            for ins in insights:
                groups.setdefault(ins['type'], []).append(ins)

            for itype, ilist in groups.items():
                icon = ilist[0].get('icon', '')
                with st.expander(f"{icon} **{itype}** ({len(ilist)} recommendations)", expanded=True):
                    for item in ilist:
                        ant_s = ", ".join(sorted(item['antecedents']))
                        con_s = ", ".join(sorted(item['consequents']))
                        st.markdown(f"""
                        <div class="insight-card">
                            <strong style="color:{text_primary};font-size:0.95rem;">{item['title']}</strong>
                            <p style="color:{text_secondary};font-size:0.86rem;margin:8px 0;line-height:1.6;">{item['description']}</p>
                            <div style="display:flex;gap:16px;font-size:0.78rem;color:{text_muted};flex-wrap:wrap;margin-top:8px;">
                                <span>📌 <strong>{ant_s}</strong> → <strong>{con_s}</strong></span>
                                <span>Support: <strong style="color:{accent_blue};">{item['support']:.4f}</strong></span>
                                <span>Confidence: <strong style="color:{accent_teal};">{item['confidence']:.4f}</strong></span>
                                <span>Lift: <strong style="color:{accent_green};">{item['lift']:.4f}</strong></span>
                                <span>Leverage: <strong style="color:{accent_orange};">{item['leverage']:.4f}</strong></span>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

            st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
            ins_df = insights_to_dataframe(insights)
            st.subheader("📋 Recommendations Table")
            st.dataframe(ins_df, use_container_width=True)

            st.download_button(
                "⬇ Download Recommendations CSV",
                ins_df.to_csv(index=False).encode(),
                "ai_recommendations.csv", "text/csv"
            )

    # ── TAB 2: Business Insights Dashboard ────────────────────────────
    with tab_bi:
        st.subheader("💡 Executive Pattern Intelligence")
        bk1, bk2, bk3, bk4 = st.columns(4)
        bk1.metric("Bundle Opportunities", sum(1 for i in insights if i['type']=="Bundle Offer"))
        bk2.metric("Cross-Sell Potential", sum(1 for i in insights if i['type']=="Cross-Selling"))
        bk3.metric("Placement Recommendations", sum(1 for i in insights if i['type']=="Product Placement"))
        bk4.metric("Frequent Core Pairs", sum(1 for i in insights if i['type']=="Frequent Pattern"))

        st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
        bi_c1, bi_c2 = st.columns(2)
        with bi_c1:
            st.subheader("📊 Top Rules by Confidence")
            top_conf = rules_df.nlargest(12, "confidence").copy()
            top_conf['Rule'] = top_conf.apply(lambda r: f"{', '.join(sorted(r['antecedents']))} → {', '.join(sorted(r['consequents']))}", axis=1)
            fig_conf = px.bar(top_conf.sort_values("confidence"), x="confidence", y="Rule", orientation="h",
                              color="confidence", color_continuous_scale="Greens", template=PLOTLY_TEMPLATE)
            fig_conf = apply_plotly_theme(fig_conf, 420)
            fig_conf.update_coloraxes(showscale=False)
            st.plotly_chart(fig_conf, use_container_width=True)

        with bi_c2:
            st.subheader("🚀 Top Rules by Lift Score")
            top_l = rules_df.nlargest(12, "lift").copy()
            top_l['Rule'] = top_l.apply(lambda r: f"{', '.join(sorted(r['antecedents']))} → {', '.join(sorted(r['consequents']))}", axis=1)
            fig_l = px.bar(top_l.sort_values("lift"), x="lift", y="Rule", orientation="h",
                           color="lift", color_continuous_scale="Plasma", template=PLOTLY_TEMPLATE)
            fig_l = apply_plotly_theme(fig_l, 420)
            fig_l.update_coloraxes(showscale=False)
            st.plotly_chart(fig_l, use_container_width=True)


# =====================================================================
# 8. PAGE: 📈 ALGORITHM COMPARISON
# =====================================================================
elif current_page == "Algorithm Comparison":
    render_page_header("Algorithm Comparison: Apriori vs FP-Growth", "Comprehensive benchmark comparing candidate generation against FP-Tree pattern growth.")
    require_dataset()

    if freq_dev.empty or fpg_freq.empty:
        st.info("💡 Please execute both Apriori and FP-Growth in **⚡ Association Rule Mining** or **🎯 Adaptive Parameters** to unlock the comparison.")
        st.stop()

    st.subheader("📊 Side-by-Side Execution Benchmark")
    comp_metrics = pd.DataFrame({
        "Performance Dimension": [
            "Frequent Itemsets Found",
            "Association Rules Generated",
            "Execution Time (seconds)",
            "Applied Minimum Support",
            "Applied Minimum Confidence",
            "Algorithmic Approach",
            "Candidate Generation Required"
        ],
        "Apriori Baseline": [
            f"{len(freq_dev):,}",
            f"{len(rules_df):,}",
            f"{t_time:.4f} s",
            str(min_sup),
            str(min_conf),
            "Breadth-First Candidate Gen",
            "Yes (Combinatorial Candidates)"
        ],
        "FP-Growth": [
            f"{len(fpg_freq):,}",
            f"{len(fpg_rules):,}",
            f"{fpg_time:.4f} s",
            str(min_sup),
            str(min_conf),
            "Recursive FP-Tree Mining",
            "No (Candidate-Free Tree Growth)"
        ],
    })
    st.dataframe(comp_metrics, use_container_width=True)

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
    c_b1, c_b2 = st.columns(2)
    with c_b1:
        st.subheader("⏱ Runtime Speedup Analysis")
        time_comp_df = pd.DataFrame({"Algorithm": ["Apriori Baseline", "FP-Growth"], "Time (s)": [t_time, fpg_time]})
        fig_time = px.bar(time_comp_df, x="Algorithm", y="Time (s)", color="Algorithm",
                          color_discrete_sequence=[accent_blue, accent_green], template=PLOTLY_TEMPLATE)
        fig_time = apply_plotly_theme(fig_time, 320)
        st.plotly_chart(fig_time, use_container_width=True)

        if fpg_time > 0 and t_time > 0:
            speedup = t_time / fpg_time
            if speedup >= 1.0:
                st.success(f"✅ **FP-Growth is {speedup:.2f}× faster** than Apriori for this dataset and threshold.")
            else:
                st.info(f"ℹ Apriori was faster on this run ({1/speedup:.2f}×). Tree construction may dominate on small itemsets.")

    with c_b2:
        st.subheader("📊 Result Equivalence & Consistency")
        cnt_comp_df = pd.DataFrame({
            "Metric": ["Frequent Itemsets", "Association Rules", "Frequent Itemsets", "Association Rules"],
            "Count": [len(freq_dev), len(rules_df), len(fpg_freq), len(fpg_rules)],
            "Algorithm": ["Apriori", "Apriori", "FP-Growth", "FP-Growth"],
        })
        fig_cnt = px.bar(cnt_comp_df, x="Metric", y="Count", color="Algorithm", barmode="group",
                         color_discrete_sequence=[accent_blue, accent_green], template=PLOTLY_TEMPLATE)
        fig_cnt = apply_plotly_theme(fig_cnt, 320)
        st.plotly_chart(fig_cnt, use_container_width=True)

        # Mathematical Equivalence Verification
        apr_sets = set(frozenset(x) for x in freq_dev['itemsets'])
        fpg_sets = set(frozenset(x) for x in fpg_freq['itemsets'])
        if apr_sets == fpg_sets:
            st.success("✅ **Mathematical Equivalence Confirmed**: Both algorithms produced 100% identical frequent itemsets.")
        else:
            st.info(f"Unique to Apriori: {len(apr_sets - fpg_sets)} | Unique to FP-Growth: {len(fpg_sets - apr_sets)}")


# =====================================================================
# 9. PAGE: 📋 TESTING & REPORTS
# =====================================================================
elif current_page == "Testing & Reports":
    render_page_header("Testing & Reports Exporter", "Execute live automated unit tests (UT01–UT20) and download complete technical reports.")
    require_dataset()

    tab_test, tab_rep = st.tabs([
        "🧪 Tab 1: Automated Unit Testing",
        "📄 Tab 2: Reports & Data Export"
    ])

    # ── TAB 1: Unit Testing ───────────────────────────────────────────
    with tab_test:
        st.subheader("🧪 Live Pytest Automated Test Suite (UT01–UT20)")
        if st.button("▶️ Execute Full Pytest Suite (`pytest tests/test_pipeline.py`)", type="primary", key="btn_run_tests"):
            with st.spinner("Executing 20 unit tests across all project modules..."):
                tresults, tstdout, retcode = run_actual_pytest()
                st.session_state.test_data = tresults
                st.session_state.test_stdout = tstdout
                st.session_state.test_rc = retcode

        if "test_data" in st.session_state and st.session_state.test_data:
            tdata = st.session_state.test_data
            total = len(tdata)
            passed = sum(1 for t in tdata if t["Status"] == "PASSED")
            failed = total - passed

            ts1, ts2, ts3, ts4 = st.columns(4)
            ts1.metric("Total Tests", total)
            ts2.metric("Passed Tests", passed)
            ts3.metric("Failed Tests", failed)
            ts4.metric("Suite Verdict", "✔ 100% PASSED" if failed==0 and total>0 else "✘ FAILED")

            st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
            st.subheader("🔬 Module Verification Pipeline Status")
            flow_tests = [
                ("Dataset Loading", "UT01-02", "test_ut01_dataset_loading"),
                ("Preprocessing", "UT03-06", "test_ut03_duplicate_removal"),
                ("Encoding", "UT07", "test_ut07_one_hot_encoding"),
                ("Apriori Mining", "UT08-12", "test_ut08_apriori_execution"),
                ("Validation", "UT13", "test_ut13_validation_metric_calculation"),
                ("FP-Growth", "UT14-16", "test_ut14_fpgrowth_execution"),
                ("Recommender", "UT17", "test_ut17_adaptive_parameter_recommendation"),
                ("Quality & AI", "UT18-20", "test_ut18_rule_filtering_and_ranking"),
            ]
            s_map = {t["func_name"]: (t["Status"]=="PASSED") for t in tdata}
            f_cols = st.columns(len(flow_tests))
            for col, (title, ut_label, fn) in zip(f_cols, flow_tests):
                with col:
                    is_p = s_map.get(fn, False)
                    bg = "#064E3B" if (is_p and is_dark) else ("#D1FAE5" if is_p else "#7F1D1D")
                    brd = "#059669" if is_p else "#DC2626"
                    clr = "#34D399" if is_p else "#F87171"
                    st.markdown(f"""
                    <div style="background:{bg};border:1px solid {brd};border-radius:9px;padding:10px 4px;text-align:center;">
                        <strong style="color:{text_primary};font-size:0.75rem;display:block;">{title}</strong>
                        <small style="color:{text_muted};display:block;margin:2px 0;">{ut_label}</small>
                        <strong style="color:{clr};font-size:0.8rem;">{'✔ PASS' if is_p else '✘ FAIL'}</strong>
                    </div>
                    """, unsafe_allow_html=True)

            st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
            st.subheader("📋 Complete Test Case Results Table")
            res_df = pd.DataFrame(tdata)[["Test ID", "Module", "Test Case", "Expected Result", "Actual Result", "Status"]]
            st.dataframe(
                res_df.style.map(
                    lambda v: 'background-color:#064E3B;color:#34D399;font-weight:bold;' if v=='PASSED'
                              else 'background-color:#7F1D1D;color:#F87171;font-weight:bold;',
                    subset=['Status']
                ), use_container_width=True
            )

            with st.expander("📝 Raw Pytest Console Output"):
                st.code(st.session_state.get("test_stdout", ""), language="text")
        else:
            st.info("Click the button above to execute the automated pytest suite.")

    # ── TAB 2: Reports & Export ───────────────────────────────────────
    with tab_rep:
        st.subheader("📄 Comprehensive Project Reports & Export")
        r_sum1, r_sum2, r_sum3, r_sum4 = st.columns(4)
        r_sum1.metric("Active Dataset", active_dataset_label)
        r_sum2.metric("Applied Min Support", str(min_sup))
        r_sum3.metric("Apriori Rules", f"{len(rules_df):,}")
        r_sum4.metric("FP-Growth Rules", f"{len(fpg_rules):,}")

        st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
        st.subheader("⬇ Individual Dataset & Results Exports (CSV)")
        d_c1, d_c2, d_c3, d_c4 = st.columns(4)

        with d_c1:
            st.markdown("**Frequent Itemsets**")
            if not freq_dev.empty:
                fi_exp = freq_dev.copy()
                fi_exp['itemsets'] = fi_exp['itemsets'].apply(lambda x: ", ".join(sorted(x)))
                st.download_button("⬇ Itemsets CSV", fi_exp[['itemsets', 'support', 'itemset_len']].to_csv(index=False).encode(), "frequent_itemsets.csv", "text/csv")
            else: st.caption("Run Apriori first.")

        with d_c2:
            st.markdown("**Apriori Rules**")
            if not rules_df.empty:
                r_exp = rules_df.copy()
                r_exp['antecedents'] = r_exp['antecedents'].apply(lambda x: ", ".join(sorted(x)))
                r_exp['consequents'] = r_exp['consequents'].apply(lambda x: ", ".join(sorted(x)))
                st.download_button("⬇ Apriori Rules CSV", r_exp[['antecedents', 'consequents', 'support', 'confidence', 'lift', 'leverage']].to_csv(index=False).encode(), "apriori_rules.csv", "text/csv")
            else: st.caption("Run Apriori first.")

        with d_c3:
            st.markdown("**FP-Growth Rules**")
            if not fpg_rules.empty:
                f_exp = fpg_rules.copy()
                f_exp['antecedents'] = f_exp['antecedents'].apply(lambda x: ", ".join(sorted(x)))
                f_exp['consequents'] = f_exp['consequents'].apply(lambda x: ", ".join(sorted(x)))
                st.download_button("⬇ FP-Growth Rules CSV", f_exp[['antecedents', 'consequents', 'support', 'confidence', 'lift', 'leverage']].to_csv(index=False).encode(), "fpgrowth_rules.csv", "text/csv")
            else: st.caption("Run FP-Growth first.")

        with d_c4:
            st.markdown("**Business Insights**")
            if not rules_df.empty:
                insights_list = generate_insights(rules_df, top_n=50)
                if insights_list:
                    ins_export_df = insights_to_dataframe(insights_list)
                    st.download_button("⬇ Insights CSV", ins_export_df.to_csv(index=False).encode(), "business_insights.csv", "text/csv")
            else: st.caption("Run Apriori first.")

        st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
        st.subheader("📄 Generated Full Technical Report")

        rep_lines = [
            "=" * 72,
            "SCALABLE ASSOCIATION RULE MINING AND ESTIMATOR",
            "Technical Analysis & Pattern Mining Report",
            "=" * 72,
            "",
            "1. DATASET CHARACTERISTICS",
            "-" * 40,
            f"  Dataset Name         : {active_dataset_label}",
            f"  Raw Ingested Records : {raw_stats['raw_rows']:,}",
            f"  Duplicates Cleaned   : {raw_stats['exact_duplicate_rows']:,}",
            f"  Missing Values       : {raw_stats['total_missing_values']}",
            f"  Cleaned Records      : {len(clean_df):,}",
            f"  Unique Products      : {raw_stats['unique_products']:,}",
            f"  Unique Customers     : {raw_stats['unique_customers']:,}",
            f"  Total Transactions   : {basket_stats['num_transactions']:,}",
            f"  Avg Basket Size      : {basket_stats['avg_basket_size']:.3f} items",
            f"  Development Set (80%): {len(dev_baskets):,} transactions",
            f"  Validation Set  (20%): {len(val_baskets):,} transactions",
            "",
            "2. MINING CONFIGURATION & RESULTS",
            "-" * 40,
            f"  Applied Min Support    : {min_sup}",
            f"  Applied Min Confidence : {min_conf}",
            f"  Apriori Itemsets       : {len(freq_dev):,}",
            f"  Apriori Rules          : {len(rules_df):,}",
            f"  Apriori Execution Time : {t_time:.4f} s",
            f"  FP-Growth Itemsets     : {len(fpg_freq):,}",
            f"  FP-Growth Rules        : {len(fpg_rules):,}",
            f"  FP-Growth Execution Time: {fpg_time:.4f} s",
        ]

        if not rules_df.empty:
            rep_lines += [
                "",
                "3. TOP 10 DISCOVERED RULES (BY LIFT)",
                "-" * 40,
            ]
            for _, r_row in rules_df.nlargest(10, "lift").iterrows():
                ant_txt = ", ".join(sorted(r_row['antecedents']))
                con_txt = ", ".join(sorted(r_row['consequents']))
                rep_lines.append(f"  {ant_txt} → {con_txt}  [sup={r_row['support']:.4f}, conf={r_row['confidence']:.4f}, lift={r_row['lift']:.4f}]")

        rep_lines += [
            "",
            "=" * 72,
            "END OF REPORT — Generated by Scalable Association Rule Mining and Estimator",
            "=" * 72,
        ]

        report_txt = "\n".join(rep_lines)
        st.text_area("Report Preview", value=report_txt, height=360)

        pdf_col, txt_col = st.columns(2)
        with txt_col:
            st.download_button("⬇ Download Full Text Report (.txt)", data=report_txt.encode(), file_name="mining_report.txt", mime="text/plain")

        with pdf_col:
            if st.button("📑 Generate & Download PDF Report", type="primary"):
                try:
                    from fpdf import FPDF
                    pdf = FPDF()
                    pdf.set_auto_page_break(auto=True, margin=15)
                    pdf.add_page()
                    pdf.set_font("Helvetica", "B", 15)
                    pdf.cell(0, 10, "Scalable Association Rule Mining and Estimator", ln=True, align="C")
                    pdf.set_font("Helvetica", "", 10)
                    pdf.cell(0, 6, "Technical Analysis & Pattern Mining Report", ln=True, align="C")
                    pdf.ln(6)
                    for line in rep_lines:
                        safe_line = line.encode('latin-1', 'replace').decode('latin-1')
                        if safe_line.startswith("=") or safe_line.startswith("-"):
                            pdf.set_font("Helvetica", "", 8)
                            pdf.cell(0, 4, safe_line, ln=True)
                        elif safe_line.strip() == "":
                            pdf.ln(3)
                        elif safe_line.strip().startswith(("1.", "2.", "3.")):
                            pdf.set_font("Helvetica", "B", 11)
                            pdf.cell(0, 6, safe_line, ln=True)
                        else:
                            pdf.set_font("Helvetica", "", 9)
                            pdf.multi_cell(0, 5, safe_line)
                    pdf_bytes = pdf.output()
                    st.download_button("⬇ Click to Download Generated PDF", data=bytes(pdf_bytes), file_name="mining_report.pdf", mime="application/pdf")
                    st.success("✅ PDF successfully compiled!")
                except Exception as e:
                    st.error(f"PDF compilation error: {e}")


# =====================================================================
# Application Footer
# =====================================================================
st.markdown(f"""
<div style="margin-top:50px;padding:18px 0;border-top:1px solid {border_card};text-align:center;color:{text_muted};font-size:0.8rem;">
    <strong>Scalable Association Rule Mining and Estimator</strong> &copy; 2026. All rights reserved.
</div>
""", unsafe_allow_html=True)
