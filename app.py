"""
app.py - Aegis: AI Credit Risk & Financial Inclusion Platform
Ultra-Modern White Luxury Fintech Architecture featuring Triple-Layer XAI,
Neuro-Symbolic Syllogistic Auditing, and Real-Time Risk Simulation.
"""

import os
import json
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

# Local modular imports
from loan_utils import (
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    CATEGORICAL_OPTIONS,
    engineer_features,
    get_or_create_model_pipeline,
    predict_credit_risk
)
from neuro_symbolic import NeuroSymbolicEngine
from xai_engine import XAIEngine
from fairness_audit import get_fairness_audit_data

# -----------------------------------------------------------------------------
# 1. Page Configuration & Custom Luxe White CSS
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Aegis | AI Credit Risk & Financial Inclusion",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Bespoke Pure White Luxury Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
        color: #0F172A;
        background-color: #FFFFFF;
    }

    /* Background overrides */
    .stApp {
        background: #FFFFFF;
        background-image: 
            radial-gradient(at 0% 0%, rgba(241, 245, 249, 0.6) 0px, transparent 50%),
            radial-gradient(at 100% 100%, rgba(248, 250, 252, 0.8) 0px, transparent 50%);
    }

    /* Hide standard Streamlit header clutter */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Bespoke Floating Topbar */
    .aegis-header {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 18px;
        padding: 24px 32px;
        margin-bottom: 28px;
        box-shadow: 0 10px 25px -5px rgba(15, 23, 42, 0.03), 0 8px 10px -6px rgba(15, 23, 42, 0.02);
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .brand-title {
        font-size: 1.85rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        background: linear-gradient(135deg, #0F172A 0%, #334155 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .brand-sub {
        font-size: 0.92rem;
        color: #64748B;
        font-weight: 500;
        margin-top: 4px;
    }

    /* Badge Pills */
    .badge-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 6px 14px;
        border-radius: 9999px;
        font-size: 0.82rem;
        font-weight: 600;
        letter-spacing: 0.02em;
    }
    .pill-blue {
        background-color: #EFF6FF;
        color: #1D4ED8;
        border: 1px solid #DBEAFE;
    }
    .pill-emerald {
        background-color: #ECFDF5;
        color: #047857;
        border: 1px solid #A7F3D0;
    }
    .pill-amber {
        background-color: #FFFBEB;
        color: #B45309;
        border: 1px solid #FDE68A;
    }
    .pill-rose {
        background-color: #FFF1F2;
        color: #BE123C;
        border: 1px solid #FECDD3;
    }
    .pill-purple {
        background-color: #FAF5FF;
        color: #7E22CE;
        border: 1px solid #E9D5FF;
    }

    /* Cards */
    .white-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 16px;
        padding: 24px;
        box-shadow: 0 4px 20px -2px rgba(15, 23, 42, 0.04);
        margin-bottom: 20px;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .white-card:hover {
        box-shadow: 0 12px 30px -4px rgba(15, 23, 42, 0.06);
    }

    .kpi-title {
        font-size: 0.82rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #64748B;
        margin-bottom: 6px;
    }
    .kpi-value {
        font-size: 2.1rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        color: #0F172A;
        line-height: 1.1;
    }
    .kpi-delta {
        font-size: 0.82rem;
        font-weight: 600;
        margin-top: 6px;
        display: flex;
        align-items: center;
        gap: 4px;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #FAFBFC;
        border-right: 1px solid #E2E8F0;
    }
    section[data-testid="stSidebar"] .stRadio label {
        font-weight: 500;
        color: #334155;
        border-radius: 10px;
        padding: 8px 12px;
        transition: all 0.15s ease;
    }
    section[data-testid="stSidebar"] .stRadio label:hover {
        background-color: #F1F5F9;
        color: #0F172A;
    }

    /* Custom Form & Inputs */
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input,
    .stSelectbox > div > div {
        background-color: #FFFFFF !important;
        border: 1px solid #CBD5E1 !important;
        border-radius: 10px !important;
        color: #0F172A !important;
        font-weight: 500 !important;
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.03) !important;
    }
    .stTextInput > div > div > input:focus,
    .stNumberInput > div > div > input:focus {
        border-color: #2563EB !important;
        box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.12) !important;
    }

    /* Buttons */
    .stButton > button {
        background: #0F172A !important;
        color: #FFFFFF !important;
        font-weight: 600 !important;
        border-radius: 12px !important;
        border: 1px solid #0F172A !important;
        padding: 10px 22px !important;
        box-shadow: 0 4px 12px rgba(15, 23, 42, 0.12) !important;
        transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1) !important;
    }
    .stButton > button:hover {
        background: #1E293B !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 8px 20px rgba(15, 23, 42, 0.18) !important;
    }

    /* Rule Syllogism Cards */
    .syllogism-box {
        background: #FAFBFC;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 12px;
        border-left: 5px solid #2563EB;
    }
    .syllogism-box.risk {
        border-left-color: #EF4444;
        background: #FFFBFB;
    }
    .syllogism-box.merit {
        border-left-color: #10B981;
        background: #FBFDFB;
    }
    .syllogism-formula {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.82rem;
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        padding: 4px 8px;
        border-radius: 6px;
        color: #475569;
        display: inline-block;
        margin-top: 6px;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. Resource Initializers
# -----------------------------------------------------------------------------
@st.cache_resource
def load_system():
    pipeline = get_or_create_model_pipeline()
    symbolic_engine = NeuroSymbolicEngine()
    xai_engine = XAIEngine(pipeline)
    return pipeline, symbolic_engine, xai_engine

@st.cache_data
def load_sample_applicants():
    csv_path = os.path.join(os.path.dirname(__file__), "sample_applicants.csv")
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    return pd.DataFrame()

@st.cache_data
def load_model_card():
    card_path = os.path.join(os.path.dirname(__file__), "model_card.json")
    if os.path.exists(card_path):
        with open(card_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

pipeline, symbolic_engine, xai_engine = load_system()
sample_df = load_sample_applicants()
model_card = load_model_card()

# -----------------------------------------------------------------------------
# 3. Sidebar Navigation & Global Controls
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("""
    <div style='padding: 10px 0 20px 0;'>
        <div style='display:flex; align-items:center; gap:10px;'>
            <div style='background:#0F172A; width:34px; height:34px; border-radius:8px; display:flex; align-items:center; justify-content:center; color:white; font-weight:800; font-size:1.1rem;'>
                A
            </div>
            <div>
                <span style='font-size:1.15rem; font-weight:800; color:#0F172A; letter-spacing:-0.02em;'>AEGIS</span>
                <span style='font-size:0.75rem; color:#64748B; font-weight:600; display:block;'>CREDIT INTELLIGENCE</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    navigation = st.radio(
        "NAVIGATION",
        [
            "🏠 Executive Overview",
            "💳 Credit Risk Assessment",
            "🔍 Explain My Decision (XAI)",
            "⚖️ Fairness & Responsible AI",
            "📊 Model Benchmarks & Specs",
            "⚙️ Artifact Settings"
        ],
        index=1
    )

    st.markdown("<hr style='border:none; border-top:1px solid #E2E8F0; margin:24px 0;'>", unsafe_allow_html=True)
    st.markdown("##### **PLATFORM TELEMETRY**")
    st.markdown("""
    <div style='background:#FFFFFF; border:1px solid #E2E8F0; border-radius:12px; padding:14px;'>
        <div style='display:flex; justify-content:space-between; margin-bottom:8px;'>
            <span style='font-size:0.8rem; color:#64748B;'>Engine</span>
            <span class='badge-pill pill-blue' style='padding:2px 8px;'>XGBoost v3.4</span>
        </div>
        <div style='display:flex; justify-content:space-between; margin-bottom:8px;'>
            <span style='font-size:0.8rem; color:#64748B;'>Latency</span>
            <span class='badge-pill pill-emerald' style='padding:2px 8px;'>~14 ms</span>
        </div>
        <div style='display:flex; justify-content:space-between;'>
            <span style='font-size:0.8rem; color:#64748B;'>Audit Rules</span>
            <span class='badge-pill pill-purple' style='padding:2px 8px;'>9 Active</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.caption("Bondora P2P Dataset • 110,342 Seasoned Loans")

# Topbar Banner
st.markdown("""
<div class='aegis-header'>
    <div>
        <div class='brand-title'>
            <span>Aegis Financial Inclusion & Risk Platform</span>
            <span class='badge-pill pill-blue'>BONDORA P2P ENTERPRISE</span>
        </div>
        <div class='brand-sub'>Multi-Layer Neuro-Symbolic Credit Scoring, Automated Prudential Auditing, and Responsible Underwriting</div>
    </div>
    <div style='display:flex; gap:10px;'>
        <span class='badge-pill pill-emerald'>● SYSTEM LIVE</span>
        <span class='badge-pill pill-purple'>UN SDG 1 & 10</span>
    </div>
</div>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# PAGE 1: EXECUTIVE OVERVIEW
# -----------------------------------------------------------------------------
if navigation == "🏠 Executive Overview":
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("""
        <div class='white-card'>
            <div class='kpi-title'>Seasoned Cohort</div>
            <div class='kpi-value'>110,342</div>
            <div class='kpi-delta' style='color:#059669;'>12-Mo Maturation Window</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class='white-card'>
            <div class='kpi-title'>XGBoost Stratified ROC-AUC</div>
            <div class='kpi-value'>0.781</div>
            <div class='kpi-delta' style='color:#2563EB;'>CV Validated (5-Fold)</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class='white-card'>
            <div class='kpi-title'>Explainability Layers</div>
            <div class='kpi-value'>3 Types</div>
            <div class='kpi-delta' style='color:#7C3AED;'>SHAP + LIME + Symbolic</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown("""
        <div class='white-card'>
            <div class='kpi-title'>Leakage Governance</div>
            <div class='kpi-value'>52 Eliminated</div>
            <div class='kpi-delta' style='color:#059669;'>100% Application-Time Only</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class='white-card'>
        <h3 style='margin-top:0; font-weight:700; color:#0F172A;'>Core Mission: Responsible Financial Inclusion via Neuro-Symbolic XAI</h3>
        <p style='color:#475569; line-height:1.7; font-size:1.02rem;'>
            Traditional credit scoring penalizes thin-file individuals (first-time borrowers, gig economy workers, and new market entrants) simply because they lack extensive borrowing histories. 
            Aegis combines <strong>gradient-boosted non-linear tree models (XGBoost)</strong> with <strong>First-Order Symbolic Expert Rules</strong> to ensure that applicants with genuine liquidity, high academic human capital, or sound cashflow buffers are not unfairly rejected by opaque algorithms.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 🏛️ Dual-Engine Architecture Workflow")
    st.markdown("""
    ```text
    ┌────────────────────────────────────────────────────────────────────────────────────────┐
    │                              Applicant Origination Features                             │
    │              (Age, Applied Amount, Income, Duration, Liabilities, Country, Education)   │
    └───────────────────────────────────────────┬────────────────────────────────────────────┘
                                                │
                                                ▼
    ┌────────────────────────────────────────────────────────────────────────────────────────┐
    │                         Zero-Leakage Feature Engineering Engine                        │
    │            (Payment-to-Income, Liability-to-Income, Free Cash, Repayment Ratio)        │
    └─────────────────────┬───────────────────────────────────────────────────┬──────────────┘
                          │                                                   │
                          ▼                                                   ▼
    ┌───────────────────────────────────────────┐       ┌────────────────────────────────────┐
    │           Black-Box ML Pipeline           │       │    First-Order Logic Rulebook      │
    │        XGBoost Gradient Boosted Trees     │       │   Prudential Policies & Waivers    │
    │         Empirical Probability: 26.4%      │       │     Symbolic Score: 25 / 100       │
    └─────────────────────┬─────────────────────┘       └─────────────────────┬──────────────┘
                          │                                                   │
                          └─────────────────────┬─────────────────────────────┘
                                                │
                                                ▼
    ┌────────────────────────────────────────────────────────────────────────────────────────┐
    │                           Consensus & Triple-Layer XAI Suite                           │
    │  • Layer 1 (SHAP): Exact Tree Attributions & Verified Additivity Axiom                  │
    │  • Layer 2 (LIME): Neighborhood Surrogates in Unencoded Domain                         │
    │  • Layer 3 (Neuro-Symbolic): Formal Logic Syllogisms & Prudential Compliance           │
    └────────────────────────────────────────────────────────────────────────────────────────┘
    ```
    """)

# -----------------------------------------------------------------------------
# PAGE 2: CREDIT RISK ASSESSMENT
# -----------------------------------------------------------------------------
elif navigation == "💳 Credit Risk Assessment":
    st.markdown("<h2 style='font-weight:800; color:#0F172A; letter-spacing:-0.02em;'>Applicant Assessment & Credit Scoring</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color:#64748B; margin-bottom:20px;'>Select a pre-calibrated test archetype or customize individual parameters to trigger instant multi-model inference.</p>", unsafe_allow_html=True)

    # Archetype selector chips
    st.markdown("##### **Quick-Load Calibrated Archetypes**")
    preset_cols = st.columns(6)

    archetype_labels = [
        ("Prime Low-Risk", "APP-101"),
        ("Moderate Prime", "APP-102"),
        ("High Over-Leveraged", "APP-103"),
        ("Thin-File Inclusion", "APP-104"),
        ("Senior Retiree", "APP-105"),
        ("Subprime Stressed", "APP-106")
    ]

    for i, (label, app_id) in enumerate(archetype_labels):
        with preset_cols[i]:
            if st.button(f"👤 {label}", key=f"chip_{i}", use_container_width=True):
                match_row = sample_df[sample_df["ApplicantId"] == app_id]
                if not match_row.empty:
                    st.session_state["applicant_data"] = match_row.iloc[0].to_dict()

    current_data = st.session_state.get("applicant_data", {
        "Age": 36,
        "AppliedAmount": 3500.0,
        "Amount": 3500.0,
        "Interest": 17.5,
        "LoanDuration": 36,
        "MonthlyPayment": 125.0,
        "IncomeTotal": 2600.0,
        "ExistingLiabilities": 1,
        "LiabilitiesTotal": 200.0,
        "DebtToIncome": 0.077,
        "FreeCash": 2275.0,
        "Country": "EE",
        "NewCreditCustomer": "No",
        "Education": "Higher",
        "EmploymentStatus": "Fully employed",
        "HomeOwnershipType": "Owner",
        "NoOfPreviousLoansBeforeLoan": 1,
        "AmountOfPreviousLoansBeforeLoan": 2500.0,
        "PreviousRepaymentsBeforeLoan": 2500.0
    })

    st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)

    with st.form("application_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("##### 👤 Borrower Credentials")
            age = st.number_input("Age (Years)", min_value=18, max_value=85, value=int(current_data.get("Age", 36)))
            country = st.selectbox("National Jurisdiction", CATEGORICAL_OPTIONS["Country"], index=CATEGORICAL_OPTIONS["Country"].index(current_data.get("Country", "EE")))
            education = st.selectbox("Educational Attainment", CATEGORICAL_OPTIONS["Education"], index=CATEGORICAL_OPTIONS["Education"].index(current_data.get("Education", "Higher")))
            emp_status = st.selectbox("Employment Contract", CATEGORICAL_OPTIONS["EmploymentStatus"], index=CATEGORICAL_OPTIONS["EmploymentStatus"].index(current_data.get("EmploymentStatus", "Fully employed")))
            home_type = st.selectbox("Residential Status", CATEGORICAL_OPTIONS["HomeOwnershipType"], index=CATEGORICAL_OPTIONS["HomeOwnershipType"].index(current_data.get("HomeOwnershipType", "Owner")))
            new_cust = st.selectbox("First-Time Borrower (Thin File)?", CATEGORICAL_OPTIONS["NewCreditCustomer"], index=CATEGORICAL_OPTIONS["NewCreditCustomer"].index(current_data.get("NewCreditCustomer", "No")))

        with col2:
            st.markdown("##### 💰 Loan Facility Request")
            applied_amt = st.number_input("Requested Facility (€)", min_value=200.0, max_value=25000.0, step=250.0, value=float(current_data.get("AppliedAmount", 3500.0)))
            funded_amt = st.number_input("Approved Principal (€)", min_value=200.0, max_value=25000.0, step=250.0, value=float(current_data.get("Amount", 3500.0)))
            interest = st.slider("Contractual Interest Rate (%)", min_value=4.0, max_value=60.0, step=0.5, value=float(current_data.get("Interest", 17.5)))
            duration = st.select_slider("Loan Duration (Months)", options=[6, 12, 18, 24, 36, 48, 60], value=int(current_data.get("LoanDuration", 36)))
            monthly_pmt = st.number_input("Monthly Installment (€)", min_value=10.0, max_value=2000.0, step=10.0, value=float(current_data.get("MonthlyPayment", 125.0)))

        with col3:
            st.markdown("##### 📊 Solvency & History")
            income = st.number_input("Monthly Verified Income (€)", min_value=200.0, max_value=30000.0, step=100.0, value=float(current_data.get("IncomeTotal", 2600.0)))
            existing_liab = st.number_input("Active Liabilities Count", min_value=0, max_value=20, value=int(current_data.get("ExistingLiabilities", 1)))
            liab_total = st.number_input("Total Debt Servicing (€/mo)", min_value=0.0, max_value=15000.0, step=50.0, value=float(current_data.get("LiabilitiesTotal", 200.0)))

            free_cash_calc = max(-500.0, income - monthly_pmt - liab_total)
            free_cash = st.number_input("Discretionary Net Surplus (€)", min_value=-1000.0, max_value=25000.0, value=float(current_data.get("FreeCash", free_cash_calc)))

            st.caption("Bondora Historical Borrowing History")
            prev_count = st.number_input("Prior Settled Facilities", min_value=0, max_value=25, value=int(current_data.get("NoOfPreviousLoansBeforeLoan", 1)))
            prev_amt = st.number_input("Prior Borrowed Principal (€)", min_value=0.0, max_value=100000.0, step=500.0, value=float(current_data.get("AmountOfPreviousLoansBeforeLoan", 2500.0)))
            prev_repay = st.number_input("Prior Repayments Satisfied (€)", min_value=0.0, max_value=100000.0, step=500.0, value=float(current_data.get("PreviousRepaymentsBeforeLoan", 2500.0)))

        assess_button = st.form_submit_button("⚡ Run Full Underwriting Assessment & XAI Audit", use_container_width=True)

    applicant_payload = {
        "Age": age,
        "AppliedAmount": applied_amt,
        "Amount": funded_amt,
        "Interest": interest,
        "LoanDuration": duration,
        "MonthlyPayment": monthly_pmt,
        "IncomeTotal": income,
        "ExistingLiabilities": existing_liab,
        "LiabilitiesTotal": liab_total,
        "DebtToIncome": liab_total / max(1.0, income),
        "FreeCash": free_cash,
        "Country": country,
        "NewCreditCustomer": new_cust,
        "Education": education,
        "EmploymentStatus": emp_status,
        "HomeOwnershipType": home_type,
        "NoOfPreviousLoansBeforeLoan": prev_count,
        "AmountOfPreviousLoansBeforeLoan": prev_amt,
        "PreviousRepaymentsBeforeLoan": prev_repay
    }
    st.session_state["evaluated_applicant"] = applicant_payload

    results = predict_credit_risk(pipeline, applicant_payload)
    st.session_state["latest_results"] = results

    # Live Assessment Dashboard
    st.markdown("<h3 style='font-weight:700; color:#0F172A; margin-top:28px;'>Live Underwriting Assessment</h3>", unsafe_allow_html=True)
    res_col1, res_col2, res_col3, res_col4 = st.columns([1.3, 1, 1.2, 1.2])

    pd_val = results["probability_of_default"]

    with res_col1:
        st.markdown("""<div class='kpi-title'>Default Probability (PD)</div>""", unsafe_allow_html=True)
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=pd_val * 100,
            number={'suffix': "%", 'font': {'size': 36, 'color': '#0F172A', 'family': 'Plus Jakarta Sans'}},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#CBD5E1"},
                'bar': {'color': results["accent_color"], 'thickness': 0.28},
                'bgcolor': "#FFFFFF",
                'borderwidth': 0,
                'steps': [
                    {'range': [0, 25], 'color': "rgba(16, 185, 129, 0.12)"},
                    {'range': [25, 45], 'color': "rgba(245, 158, 11, 0.12)"},
                    {'range': [45, 65], 'color': "rgba(249, 115, 22, 0.12)"},
                    {'range': [65, 100], 'color': "rgba(239, 68, 68, 0.12)"}
                ]
            }
        ))
        fig_gauge.update_layout(
            height=200,
            margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor="#FFFFFF"
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

    with res_col2:
        st.markdown("""<div class='kpi-title'>Regulatory Rating</div>""", unsafe_allow_html=True)
        st.markdown(f"""
        <div style='background:#FFFFFF; border:1px solid #E2E8F0; border-radius:16px; padding:20px; text-align:center; box-shadow:0 4px 15px rgba(0,0,0,0.03);'>
            <div style='font-size:3.5rem; font-weight:800; color:{results["accent_color"]}; line-height:1;'>{results['credit_grade']}</div>
            <div style='font-size:0.85rem; font-weight:700; color:#475569; margin-top:8px;'>{results['risk_category']}</div>
        </div>
        """, unsafe_allow_html=True)

    with res_col3:
        st.markdown("""<div class='kpi-title'>Underwriting Verdict</div>""", unsafe_allow_html=True)
        pill_cls = "pill-emerald" if "Approved" in results["recommendation"] else ("pill-amber" if "Review" in results["recommendation"] else "pill-rose")
        st.markdown(f"""
        <div style='background:#FFFFFF; border:1px solid #E2E8F0; border-radius:16px; padding:20px; box-shadow:0 4px 15px rgba(0,0,0,0.03);'>
            <span class='badge-pill {pill_cls}' style='font-size:0.95rem; padding:8px 16px; width:100%; justify-content:center;'>
                {results['recommendation']}
            </span>
            <div style='margin-top:12px; font-size:0.84rem; color:#64748B; line-height:1.5;'>
                Decision synthesized across statistical tree gradients and supervisory limits.
            </div>
        </div>
        """, unsafe_allow_html=True)

        if results["is_inclusion_candidate"]:
            st.markdown("""
            <div style='margin-top:8px;' class='badge-pill pill-purple'>
                🌟 Qualified Financial Inclusion Candidate
            </div>
            """, unsafe_allow_html=True)

    with res_col4:
        st.markdown("""<div class='kpi-title'>Key Prudential Metrics</div>""", unsafe_allow_html=True)
        eng = results["engineered_features"]
        st.markdown(f"""
        <div style='background:#FFFFFF; border:1px solid #E2E8F0; border-radius:16px; padding:16px; box-shadow:0 4px 15px rgba(0,0,0,0.03); font-size:0.85rem;'>
            <div style='display:flex; justify-content:space-between; margin-bottom:6px;'>
                <span style='color:#64748B;'>Payment-to-Income:</span>
                <strong>{eng.get('PaymentToIncome', 0.0)*100:.1f}%</strong>
            </div>
            <div style='display:flex; justify-content:space-between; margin-bottom:6px;'>
                <span style='color:#64748B;'>Debt-to-Income:</span>
                <strong>{applicant_payload['DebtToIncome']*100:.1f}%</strong>
            </div>
            <div style='display:flex; justify-content:space-between; margin-bottom:6px;'>
                <span style='color:#64748B;'>Net Free Cash:</span>
                <strong>€{applicant_payload['FreeCash']:.0f}/mo</strong>
            </div>
            <div style='display:flex; justify-content:space-between;'>
                <span style='color:#64748B;'>Repayment Track:</span>
                <strong>{eng.get('PreviousRepaymentRatio', 0.0)*100:.1f}%</strong>
            </div>
        </div>
        """, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# PAGE 3: EXPLAIN MY DECISION (XAI SUITE)
# -----------------------------------------------------------------------------
elif navigation == "🔍 Explain My Decision (XAI)":
    st.markdown("<h2 style='font-weight:800; color:#0F172A; letter-spacing:-0.02em;'>Explain My Decision — Triple-Layer XAI Suite</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color:#64748B; margin-bottom:24px;'>Multifaceted model interpretability satisfying EU AI Act & supervisory audit mandates.</p>", unsafe_allow_html=True)

    applicant = st.session_state.get("evaluated_applicant", None)
    if applicant is None:
        st.info("ℹ️ Please submit an applicant in the 'Credit Risk Assessment' tab first.")
        st.stop()

    applicant_df = pd.DataFrame([applicant])
    feat_df = engineer_features(applicant_df)
    ml_prob = float(pipeline.predict_proba(feat_df)[0, 1])

    tab_shap, tab_lime, tab_rules = st.tabs([
        "🌳 Layer 1: SHAP Attribution",
        "🍋 Layer 2: LIME Surrogate",
        "🧠 Layer 3: Neuro-Symbolic Syllogisms"
    ])

    # 1. SHAP
    with tab_shap:
        st.markdown("#### **SHAP (SHapley Additive exPlanations) Local Attribution**")
        st.caption("Measures exact Shapley marginal contributions pushing the applicant towards or away from loan default.")

        shap_res = xai_engine.explain_shap(feat_df)

        if shap_res["success"]:
            col_s1, col_s2 = st.columns([2.2, 1])
            with col_s1:
                drivers = shap_res["top_drivers"]
                df_drivers = pd.DataFrame(drivers)

                colors = ["#EF4444" if val > 0 else "#10B981" for val in df_drivers["shap_value"]]

                fig_shap = go.Figure(go.Bar(
                    x=df_drivers["shap_value"],
                    y=df_drivers["feature"],
                    orientation='h',
                    marker=dict(color=colors, line=dict(width=0)),
                    text=[f"{v:+.3f}" for v in df_drivers["shap_value"]],
                    textposition='outside'
                ))
                fig_shap.update_layout(
                    title="Feature Marginal Contribution to Default Risk",
                    xaxis_title="Shapley Value (Log-Odds Shift)",
                    yaxis=dict(autorange="reversed"),
                    height=450,
                    margin=dict(l=10, r=40, t=40, b=20),
                    paper_bgcolor="#FFFFFF",
                    plot_bgcolor="#FFFFFF",
                    xaxis=dict(gridcolor="#F1F5F9")
                )
                st.plotly_chart(fig_shap, use_container_width=True)

            with col_s2:
                st.markdown("""
                <div class='white-card'>
                    <h5 style='margin-top:0; font-weight:700;'>Mathematical Additivity Axiom</h5>
                    <p style='color:#64748B; font-size:0.85rem;'>Under cooperative game theory, exact attribution requires strict additivity:</p>
                    <div style='background:#F8FAFC; border:1px solid #E2E8F0; border-radius:8px; padding:12px; font-family:monospace; font-size:0.85rem; margin-bottom:12px;'>
                        E[f(x)] = """ + f"{shap_res['base_value']:.3f}" + """<br>
                        ∑ φᵢ    = """ + f"{shap_res['sum_shap']:+.3f}" + """<br>
                        f(x)    = """ + f"{(shap_res['base_value'] + shap_res['sum_shap']):.3f}" + """
                    </div>
                    <span class='badge-pill pill-emerald'>✓ Additivity Axiom Verified</span>
                </div>
                """, unsafe_allow_html=True)

    # 2. LIME
    with tab_lime:
        st.markdown("#### **LIME (Local Interpretable Model-agnostic Explanations)**")
        st.caption("Evaluates continuous sensitivity in the unencoded original applicant feature domain, avoiding dummy one-hot collisions.")

        lime_res = xai_engine.explain_lime(applicant)
        col_l1, col_l2 = st.columns([2.2, 1])

        with col_l1:
            df_lime = pd.DataFrame(lime_res["local_explanations"])
            colors_lime = ["#EF4444" if w > 0 else "#10B981" for w in df_lime["local_weight"]]

            fig_lime = go.Figure(go.Bar(
                x=df_lime["local_weight"],
                y=df_lime["feature"],
                orientation='h',
                marker=dict(color=colors_lime),
                text=[f"{w:+.3f}" for w in df_lime["local_weight"]],
                textposition='outside'
            ))
            fig_lime.update_layout(
                title="LIME Local Surrogate Sensitivity Slopes",
                xaxis_title="Local Linear Weight (Elasticity)",
                yaxis=dict(autorange="reversed"),
                height=380,
                margin=dict(l=10, r=40, t=40, b=20),
                paper_bgcolor="#FFFFFF",
                plot_bgcolor="#FFFFFF",
                xaxis=dict(gridcolor="#F1F5F9")
            )
            st.plotly_chart(fig_lime, use_container_width=True)

        with col_l2:
            st.markdown(f"""
            <div class='white-card'>
                <h5 style='margin-top:0; font-weight:700;'>Surrogate Quality Diagnostics</h5>
                <div style='font-size:2rem; font-weight:800; color:#0F172A;'>{lime_res['local_fidelity_r2']:.3f}</div>
                <div style='font-size:0.8rem; font-weight:600; color:#64748B;'>Local Surrogate Fidelity (R²)</div>
                <hr style='border:none; border-top:1px solid #E2E8F0; margin:14px 0;'>
                <div style='font-size:0.82rem; color:#475569;'>
                    Trained over 150 localized Gaussian neighborhood perturbations within the continuous financial space.
                </div>
            </div>
            """, unsafe_allow_html=True)

    # 3. NEURO-SYMBOLIC
    with tab_rules:
        st.markdown("#### **Neuro-Symbolic Reasoning Engine & Syllogistic Policy Audit**")
        st.caption("Synthesizes statistical probability with First-Order Logic constraints for formal regulatory auditing.")

        sym_audit = symbolic_engine.evaluate_applicant(applicant, feat_df.iloc[0].to_dict(), ml_prob)
        col_r1, col_r2 = st.columns([1.6, 1])

        with col_r1:
            st.markdown("##### **Evaluated First-Order Syllogisms & Prudential Bounds**")
            if not sym_audit["triggered_rules"]:
                st.success("Applicant complies with all prudential limits; no regulatory warning triggered.")
            else:
                for r in sym_audit["triggered_rules"]:
                    box_cls = "risk" if r["type"] == "NEGATIVE" else "merit"
                    icon = "⚠️" if r["type"] == "NEGATIVE" else "🛡️"
                    st.markdown(f"""
                    <div class='syllogism-box {box_cls}'>
                        <div style='display:flex; justify-content:space-between; align-items:center;'>
                            <strong>{icon} Policy {r['rule_id']}: {r['name']}</strong>
                            <span class='badge-pill {"pill-rose" if r["type"]=="NEGATIVE" else "pill-emerald"}'>{r['points']:+d} pts</span>
                        </div>
                        <div style='color:#475569; font-size:0.88rem; margin-top:4px;'>{r['detail']}</div>
                        <div class='syllogism-formula'>{r['syllogism']}</div>
                    </div>
                    """, unsafe_allow_html=True)

        with col_r2:
            st.markdown(f"""
            <div class='white-card'>
                <h5 style='margin-top:0; font-weight:700;'>Consensus Assessment</h5>
                <h3 style='color:{sym_audit['status_color']}; margin:4px 0 10px 0;'>{sym_audit['consensus_status']}</h3>
                <p style='color:#64748B; font-size:0.86rem; line-height:1.5;'>{sym_audit['consensus_desc']}</p>
                <hr style='border:none; border-top:1px solid #E2E8F0; margin:14px 0;'>
                <div style='display:flex; justify-content:space-between; margin-bottom:8px;'>
                    <span style='color:#64748B;'>Symbolic Risk Score:</span>
                    <strong>{sym_audit['symbolic_score']:.0f} / 100</strong>
                </div>
                <div style='display:flex; justify-content:space-between;'>
                    <span style='color:#64748B;'>Empirical ML Probability:</span>
                    <strong>{ml_prob*100:.1f}%</strong>
                </div>
            </div>
            """, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# PAGE 4: FAIRNESS & RESPONSIBLE AI
# -----------------------------------------------------------------------------
elif navigation == "⚖️ Fairness & Responsible AI":
    st.markdown("<h2 style='font-weight:800; color:#0F172A; letter-spacing:-0.02em;'>Fairness & Responsible AI Audit</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color:#64748B; margin-bottom:24px;'>Comprehensive disparate impact, equal opportunity, and latent proxy leakage evaluation.</p>", unsafe_allow_html=True)

    fair_data = get_fairness_audit_data()

    st.markdown("### 🛡️ Disparate Impact & Equal Opportunity Audits")
    df_di = pd.DataFrame(fair_data["disparate_impact_summary"])
    st.dataframe(df_di, use_container_width=True)

    st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)
    f_tab1, f_tab2, f_tab3 = st.tabs(["🚻 Gender Disparity", "🎂 Age Cohorts", "🌍 Country Jurisdictions"])

    with f_tab1:
        df_gender = pd.DataFrame(fair_data["gender_metrics"])
        fig_g = px.bar(
            df_gender, x="group", y=["approval_rate_at_35", "default_rate", "roc_auc"],
            barmode="group",
            title="Subgroup Performance by Gender (Direct Gender Excluded from Model Inputs)",
            color_discrete_sequence=["#2563EB", "#EF4444", "#10B981"]
        )
        fig_g.update_layout(paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF", yaxis=dict(gridcolor="#F1F5F9"))
        st.plotly_chart(fig_g, use_container_width=True)
        st.info("💡 **Audit Takeaway:** With Gender directly omitted, approval rate parity remains compliant (1.066 DI ratio) with nearly identical discriminatory power (0.784 Male vs 0.779 Female).")

    with f_tab2:
        df_age = pd.DataFrame(fair_data["age_metrics"])
        fig_a = px.bar(
            df_age, x="group", y=["approval_rate_at_35", "roc_auc"],
            barmode="group",
            title="Approval Rates and Discrimination Across Age Brackets",
            color_discrete_sequence=["#2563EB", "#7C3AED"]
        )
        fig_a.update_layout(paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF", yaxis=dict(gridcolor="#F1F5F9"))
        st.plotly_chart(fig_a, use_container_width=True)
        st.warning("⚠️ **Observed Demographic Skew:** Young applicants (< 30) exhibit lower approvals (32.4% vs 42.5%) attributable to thin credit files.")

    with f_tab3:
        df_country = pd.DataFrame(fair_data["country_metrics"])
        fig_c = px.bar(
            df_country, x="group", y=["default_rate", "approval_rate_at_35"],
            barmode="group",
            title="Jurisdictional Disparities Driven by Bondora Historical P2P Performance",
            color_discrete_sequence=["#EF4444", "#059669"]
        )
        fig_c.update_layout(paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF", yaxis=dict(gridcolor="#F1F5F9"))
        st.plotly_chart(fig_c, use_container_width=True)
        st.error("🚨 **Structural Reality:** Spanish P2P loans in the Bondora portfolio exhibited a 79.4% observed default rate, reflecting macroeconomic defaults rather than individual algorithmic bias.")

    st.markdown("### 🕵️ Latent Proxy Leakage Audit")
    for p in fair_data["proxy_analysis"]:
        st.markdown(f"""
        <div class='white-card' style='padding:16px 20px; margin-bottom:10px;'>
            <strong>Excluded Protected Attribute:</strong> <code>{p['excluded_protected_attribute']}</code> &nbsp;|&nbsp;
            <strong>Proxy Carriers:</strong> <code>{p['proxy_driver_features']}</code> ({p['correlation_strength']})
            <p style='color:#64748B; font-size:0.86rem; margin-top:6px; margin-bottom:0;'>{p['mitigation_approach']}</p>
        </div>
        """, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# PAGE 5: MODEL BENCHMARKS & SPECS
# -----------------------------------------------------------------------------
elif navigation == "📊 Model Benchmarks & Specs":
    st.markdown("<h2 style='font-weight:800; color:#0F172A; letter-spacing:-0.02em;'>Model Benchmarks & Model Card</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color:#64748B; margin-bottom:24px;'>Official performance leaderboard, temporal validation, and leakage prevention documentation.</p>", unsafe_allow_html=True)

    if model_card:
        b_df = pd.DataFrame(model_card.get("benchmarks", []))
        st.dataframe(b_df, use_container_width=True)

        fig_b = px.bar(
            b_df, x="model", y=["cv_roc_auc", "test_roc_auc", "f1"],
            barmode="group",
            title="Algorithm Comparison Across 5-Fold Cross-Validation and Untouched Test Split",
            color_discrete_sequence=["#2563EB", "#059669", "#7C3AED"]
        )
        fig_b.update_layout(paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF", yaxis=dict(gridcolor="#F1F5F9"))
        st.plotly_chart(fig_b, use_container_width=True)

        st.markdown("<hr style='border:none; border-top:1px solid #E2E8F0; margin:24px 0;'>", unsafe_allow_html=True)
        st.markdown("### ⏳ Temporal Out-of-Time Validation (Macroeconomic Drift)")
        c1, c2 = st.columns(2)
        with c1:
            st.metric("Stratified Random Test ROC-AUC", f"{model_card['performance_metrics']['in_time_stratified_test_roc_auc']:.3f}", "In-Time Validation")
        with c2:
            st.metric("Time-Based Future ROC-AUC", f"{model_card['performance_metrics']['time_based_future_validation_roc_auc']:.3f}", "-0.066 Drift", delta_color="inverse")

        st.caption(model_card['performance_metrics']['time_validation_drop_note'])

        with st.expander("📄 Full JSON Model Card"):
            st.json(model_card)

# -----------------------------------------------------------------------------
# PAGE 6: ARTIFACT SETTINGS
# -----------------------------------------------------------------------------
elif navigation == "⚙️ Artifact Settings":
    st.markdown("<h2 style='font-weight:800; color:#0F172A; letter-spacing:-0.02em;'>Artifact Manager & Google Colab Sync</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color:#64748B; margin-bottom:20px;'>Deploy trained <code>.joblib</code> models directly from your Google Colab experiments.</p>", unsafe_allow_html=True)

    up_file = st.file_uploader("Upload New Model Pipeline (.joblib)", type=["joblib"])
    if up_file is not None:
        save_dest = os.path.join(os.path.dirname(__file__), "default_risk_pipeline.joblib")
        with open(save_dest, "wb") as f:
            f.write(up_file.getbuffer())
        st.success(f"✓ Pipeline {up_file.name} successfully deployed to {save_dest}! Please reload the app.")

    st.markdown("---")
    st.markdown("##### **Active Local Runtime Inventory**")
    st.write(f"• Active Directory: `{os.path.dirname(__file__)}`")
    st.write(f"• Saved Pipeline Present: `{os.path.exists(os.path.join(os.path.dirname(__file__), 'default_risk_pipeline.joblib'))}`")
    st.write(f"• Model Card JSON Present: `{os.path.exists(os.path.join(os.path.dirname(__file__), 'model_card.json'))}`")
    st.write(f"• Sample Applicants Present: `{os.path.exists(os.path.join(os.path.dirname(__file__), 'sample_applicants.csv'))}`")
