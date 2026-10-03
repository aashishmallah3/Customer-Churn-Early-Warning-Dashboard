"""
Customer Churn Early-Warning Dashboard: XGBoost and SHAP with LLM Retention Advice
Built with Streamlit, Plotly, Scikit-Learn, and XGBoost.
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from llm_advisor import generate_retention_advice

# -------------------------------------------------------------
# Streamlit App Configuration
# -------------------------------------------------------------
st.set_page_config(
    page_title="Customer Churn Early-Warning Dashboard",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Executive CSS Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .kpi-card {
        background: linear-gradient(135deg, #F8FAFC 0%, #FFFFFF 100%);
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 18px 20px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .kpi-title {
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #64748B;
        margin-bottom: 4px;
    }
    .kpi-value {
        font-size: 1.8rem;
        font-weight: 800;
        color: #0F172A;
    }
    .kpi-delta {
        font-size: 0.8rem;
        font-weight: 600;
        margin-top: 4px;
    }
    .advice-box {
        background-color: #F0FDF4;
        border-left: 4px solid #16A34A;
        padding: 16px;
        border-radius: 8px;
        margin-top: 12px;
        margin-bottom: 12px;
    }
    .driver-pill {
        display: inline-block;
        background-color: #FEE2E2;
        color: #991B1B;
        font-weight: 600;
        padding: 3px 10px;
        border-radius: 16px;
        font-size: 0.8rem;
        margin: 2px 4px 2px 0px;
    }
    .tag-critical { color: #DC2626; font-weight: bold; }
    .tag-high { color: #EA580C; font-weight: bold; }
    .tag-medium { color: #D97706; font-weight: bold; }
    .tag-low { color: #16A34A; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------
# Data & Model Loader
# -------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    artifact_path = os.path.join(os.path.dirname(__file__), "artifacts", "model_artifacts.pkl")
    if not os.path.exists(artifact_path):
        # Auto-train pipeline if artifact not found
        import pipeline
        pipeline.run_pipeline()
        
    with open(artifact_path, "rb") as f:
        artifacts = pickle.load(f)
    return artifacts

try:
    artifacts = load_artifacts()
    test_results_df = artifacts["test_results_df"]
    full_df = artifacts["full_df"]
    cost_meta = artifacts["cost_analysis"]
    metrics_summary = artifacts["metrics_summary"]
    shap_data = artifacts["shap_data"]
    feature_names = artifacts["feature_names"]
    xgb_model = artifacts["xgb_model"]
    preprocessor = artifacts["preprocessor"]
except Exception as e:
    st.error(f"Error loading model artifacts: {e}. Please ensure pipeline.py has run successfully.")
    st.stop()

# -------------------------------------------------------------
# Session State Initialization
# -------------------------------------------------------------
if "selected_customer_id" not in st.session_state:
    st.session_state["selected_customer_id"] = test_results_df.iloc[0]["customerID"]

# -------------------------------------------------------------
# Sidebar Navigation & Business Controls
# -------------------------------------------------------------
st.sidebar.image("https://img.icons8.com/color/96/satellite-sending-signal.png", width=64)
st.sidebar.title("Radar Telemetry")
st.sidebar.caption("Customer Churn Early-Warning System")

page = st.sidebar.radio(
    "Navigation",
    ["📋 Risk-Ranked Customer List", "🔍 Customer Deep-Dive & AI Advice", "📊 Segment Analytics & Cost Curve"],
    index=0
)

st.sidebar.divider()
st.sidebar.subheader("⚙️ Decision Policy & Cost Setup")

default_opt_threshold = float(cost_meta.get("optimal_threshold", 0.28))
selected_threshold = st.sidebar.slider(
    "Decision Cutoff Threshold (t)",
    min_value=0.05,
    max_value=0.95,
    value=default_opt_threshold,
    step=0.01,
    help="Customers with churn probability >= threshold are flagged for proactive retention."
)

if st.sidebar.button(f"🎯 Reset to Cost-Optimal ({default_opt_threshold:.2f})"):
    selected_threshold = default_opt_threshold
    st.rerun()

col_cost1, col_cost2 = st.sidebar.columns(2)
with col_cost1:
    cost_offer = st.number_input("Offer Cost ($)", value=20.0, step=5.0)
with col_cost2:
    cost_churn = st.number_input("Churn Loss ($)", value=500.0, step=25.0)

st.sidebar.divider()
with st.sidebar.expander("🤖 LLM Retention Advisor Settings", expanded=False):
    llm_provider = st.selectbox("API Provider", ["Built-in Grounded Engine", "Google Gemini", "OpenAI"])
    api_key_input = st.text_input("API Key (Optional)", type="password", help="Leave blank to use the deterministic grounded retention engine.")
    st.caption("The engine generates a 2-sentence summary and 1 targeted retention action ($20) grounded strictly in SHAP drivers.")

# -------------------------------------------------------------
# Real-Time Financial & Cohort Calculations
# -------------------------------------------------------------
test_df = test_results_df.copy()
# Recalculate flags based on dynamic threshold
test_df["Flagged_At_Risk"] = test_df["Churn_Prob"] >= selected_threshold

actual_churners = (test_df["Churn_Binary"] == 1).sum()
cost_do_nothing = actual_churners * cost_churn

tp = int(((test_df["Flagged_At_Risk"]) & (test_df["Churn_Binary"] == 1)).sum())
fp = int(((test_df["Flagged_At_Risk"]) & (test_df["Churn_Binary"] == 0)).sum())
fn = int(((~test_df["Flagged_At_Risk"]) & (test_df["Churn_Binary"] == 1)).sum())
tn = int(((~test_df["Flagged_At_Risk"]) & (test_df["Churn_Binary"] == 0)).sum())

current_total_cost = (fn * cost_churn) + ((tp + fp) * cost_offer)
current_net_savings = cost_do_nothing - current_total_cost

flagged_count = int(test_df["Flagged_At_Risk"].sum())
monthly_revenue_at_risk = test_df[test_df["Flagged_At_Risk"]]["MonthlyCharges"].sum()
annual_revenue_at_risk = monthly_revenue_at_risk * 12

# =============================================================
# PAGE 1: RISK-RANKED CUSTOMER LIST
# =============================================================
if page == "📋 Risk-Ranked Customer List":
    st.markdown('<div class="main-header">📋 Customer Early-Warning Cohort</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="sub-header">Prioritized list of customer accounts ranked by XGBoost predicted churn probability with SHAP attribution. '
        f'Active threshold: <b>{selected_threshold:.2f}</b></div>',
        unsafe_allow_html=True
    )
    
    # KPI Metric Row
    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
    with kpi1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Evaluated Customers</div>
            <div class="kpi-value">{len(test_df):,}</div>
            <div class="kpi-delta" style="color: #64748B;">Test cohort size</div>
        </div>
        """, unsafe_allow_html=True)
    with kpi2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Flagged for Action</div>
            <div class="kpi-value" style="color: #DC2626;">{flagged_count:,}</div>
            <div class="kpi-delta" style="color: #DC2626;">{(flagged_count / len(test_df))*100:.1f}% of cohort</div>
        </div>
        """, unsafe_allow_html=True)
    with kpi3:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Monthly Revenue at Risk</div>
            <div class="kpi-value">${monthly_revenue_at_risk:,.0f}</div>
            <div class="kpi-delta" style="color: #EA580C;">${annual_revenue_at_risk:,.0f} annualized</div>
        </div>
        """, unsafe_allow_html=True)
    with kpi4:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Projected Net Savings</div>
            <div class="kpi-value" style="color: #16A34A;">${current_net_savings:,.0f}</div>
            <div class="kpi-delta" style="color: #16A34A;">vs $0 retention effort</div>
        </div>
        """, unsafe_allow_html=True)
    with kpi5:
        xgb_roc = metrics_summary["xgboost"]["roc_auc"]
        xgb_pr = metrics_summary["xgboost"]["pr_auc"]
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Model Performance</div>
            <div class="kpi-value" style="color: #2563EB;">{xgb_roc:.3f}</div>
            <div class="kpi-delta" style="color: #2563EB;">PR-AUC: {xgb_pr:.3f}</div>
        </div>
        """, unsafe_allow_html=True)
        
    st.write("")
    
    # Filter Section
    with st.expander("🔍 Filter Customer Cohort", expanded=True):
        fcol1, fcol2, fcol3, fcol4 = st.columns([1.5, 1.5, 1.5, 2])
        with fcol1:
            tier_options = ["All"] + list(test_df["Risk_Tier"].unique())
            selected_tier = st.selectbox("Risk Tier", tier_options)
        with fcol2:
            contract_options = ["All"] + list(test_df["Contract"].unique())
            selected_contract = st.selectbox("Contract Type", contract_options)
        with fcol3:
            service_options = ["All"] + list(test_df["InternetService"].unique())
            selected_service = st.selectbox("Internet Service", service_options)
        with fcol4:
            search_query = st.text_input("Search Customer ID", placeholder="e.g. 7590-VHVEG").strip()
            
    filtered_df = test_df.copy()
    if selected_tier != "All":
        filtered_df = filtered_df[filtered_df["Risk_Tier"] == selected_tier]
    if selected_contract != "All":
        filtered_df = filtered_df[filtered_df["Contract"] == selected_contract]
    if selected_service != "All":
        filtered_df = filtered_df[filtered_df["InternetService"] == selected_service]
    if search_query:
        filtered_df = filtered_df[filtered_df["customerID"].str.contains(search_query, case=False, na=False)]
        
    st.markdown(f"**Showing {len(filtered_df):,} accounts matching filters**")
    
    # Display table with formatted columns
    display_cols = [
        "customerID", "Risk_Tier", "Churn_Prob", "MonthlyCharges", "tenure",
        "Contract", "InternetService", "TechSupport", "PaymentMethod",
        "Top_Driver_1", "Top_Driver_2"
    ]
    
    styled_df = filtered_df[display_cols].copy()
    styled_df["Churn_Prob"] = (styled_df["Churn_Prob"] * 100).round(1).astype(str) + "%"
    styled_df["MonthlyCharges"] = "$" + styled_df["MonthlyCharges"].round(2).astype(str)
    styled_df["tenure"] = styled_df["tenure"].astype(str) + " mo"
    
    st.dataframe(
        styled_df,
        column_config={
            "customerID": st.column_config.TextColumn("Customer ID"),
            "Risk_Tier": st.column_config.TextColumn("Risk Tier"),
            "Churn_Prob": st.column_config.TextColumn("Churn Risk"),
            "MonthlyCharges": st.column_config.TextColumn("Monthly Bill"),
            "tenure": st.column_config.TextColumn("Tenure"),
            "Contract": st.column_config.TextColumn("Contract"),
            "InternetService": st.column_config.TextColumn("Internet"),
            "TechSupport": st.column_config.TextColumn("Tech Support"),
            "PaymentMethod": st.column_config.TextColumn("Payment Method"),
            "Top_Driver_1": st.column_config.TextColumn("Primary Churn Driver"),
            "Top_Driver_2": st.column_config.TextColumn("Secondary Churn Driver"),
        },
        use_container_width=True,
        hide_index=True,
        height=400
    )
    
    col_jump, col_export = st.columns([3, 1])
    with col_jump:
        selected_for_deepdive = st.selectbox(
            "Select Customer to Launch Deep-Dive & AI Retention Advice:",
            options=filtered_df["customerID"].tolist() if len(filtered_df) > 0 else test_df["customerID"].head(10).tolist(),
            index=0
        )
        if st.button("🚀 Open Customer Deep-Dive View"):
            st.session_state["selected_customer_id"] = selected_for_deepdive
            st.rerun()
            
    with col_export:
        csv_data = filtered_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Export Cohort (CSV)",
            data=csv_data,
            file_name=f"churn_risk_cohort_t{int(selected_threshold*100)}.csv",
            mime="text/csv",
        )

# =============================================================
# PAGE 2: CUSTOMER DETAIL VIEW (SHAP & AI RETENTION ADVICE)
# =============================================================
elif page == "🔍 Customer Deep-Dive & AI Advice":
    st.markdown('<div class="main-header">🔍 Customer Risk & Retention Deep-Dive</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Individual customer profile, explainable SHAP feature attribution, and grounded retention intelligence.</div>', unsafe_allow_html=True)
    
    # Customer Selector Bar
    cust_col1, cust_col2, cust_col3, cust_col4 = st.columns([2.5, 1, 1, 1])
    with cust_col1:
        current_cid = st.session_state.get("selected_customer_id", test_df.iloc[0]["customerID"])
        # Find index in full test cohort
        id_list = test_df["customerID"].tolist()
        curr_idx = id_list.index(current_cid) if current_cid in id_list else 0
        selected_cid = st.selectbox("Select Customer Account:", options=id_list, index=curr_idx)
        st.session_state["selected_customer_id"] = selected_cid
        
    with cust_col2:
        if st.button("🔥 Highest Risk"):
            st.session_state["selected_customer_id"] = test_df.iloc[0]["customerID"]
            st.rerun()
    with cust_col3:
        if st.button("💰 Top Spender"):
            top_spender_id = test_df.sort_values(by="MonthlyCharges", ascending=False).iloc[0]["customerID"]
            st.session_state["selected_customer_id"] = top_spender_id
            st.rerun()
    with cust_col4:
        if st.button("⚖️ Borderline Risk"):
            # Customer closest to decision threshold
            borderline_id = test_df.iloc[(test_df["Churn_Prob"] - selected_threshold).abs().argsort()[:1]].iloc[0]["customerID"]
            st.session_state["selected_customer_id"] = borderline_id
            st.rerun()
            
    # Retrieve customer row and test index
    customer_row = test_df[test_df["customerID"] == selected_cid].iloc[0]
    test_idx = int(customer_row["test_idx"])
    churn_prob = float(customer_row["Churn_Prob"])
    top_drivers = shap_data["top_drivers_list"][test_idx]
    
    # 2-Column Layout: Left (Profile & Risk Gauge) | Right (AI Retention Advisor)
    col_left, col_right = st.columns([1.1, 1.4])
    
    with col_left:
        st.subheader("👤 Account Profile & Financials")
        
        # Risk Gauge Chart
        gauge_fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=churn_prob * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': f"Predicted Churn Probability<br><span style='font-size:0.8em;color:gray'>Threshold: {selected_threshold*100:.0f}%</span>", 'font': {'size': 18}},
            delta={'reference': selected_threshold * 100, 'increasing': {'color': "red"}, 'decreasing': {'color': "green"}},
            number={'suffix': "%", 'font': {'size': 32, 'color': "#0F172A"}},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': "#DC2626" if churn_prob >= selected_threshold else "#16A34A"},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "#E2E8F0",
                'steps': [
                    {'range': [0, 20], 'color': '#DCFCE7'},
                    {'range': [20, 40], 'color': '#FEF9C3'},
                    {'range': [40, 70], 'color': '#FFEDD5'},
                    {'range': [70, 100], 'color': '#FEE2E2'}
                ],
                'threshold': {
                    'line': {'color': "#0F172A", 'width': 3},
                    'thickness': 0.75,
                    'value': selected_threshold * 100
                }
            }
        ))
        gauge_fig.update_layout(height=240, margin=dict(l=20, r=20, t=50, b=10))
        st.plotly_chart(gauge_fig, use_container_width=True)
        
        # Account Attributes Details
        pcol1, pcol2 = st.columns(2)
        with pcol1:
            st.markdown(f"**Tenure:** `{customer_row['tenure']} months`")
            st.markdown(f"**Contract:** `{customer_row['Contract']}`")
            st.markdown(f"**Monthly Bill:** `${customer_row['MonthlyCharges']:.2f}`")
            st.markdown(f"**Total Charges:** `${customer_row['TotalCharges']:.2f}`")
            st.markdown(f"**Paperless Billing:** `{customer_row['PaperlessBilling']}`")
        with pcol2:
            st.markdown(f"**Internet:** `{customer_row['InternetService']}`")
            st.markdown(f"**Tech Support:** `{customer_row['TechSupport']}`")
            st.markdown(f"**Online Security:** `{customer_row['OnlineSecurity']}`")
            st.markdown(f"**Payment:** `{customer_row['PaymentMethod']}`")
            st.markdown(f"**Senior Citizen:** `{'Yes' if customer_row['SeniorCitizen'] == 1 else 'No'}`")
            
        st.markdown(f"**Actual Historical Status:** `{'Churned (Left)' if customer_row['Churn_Binary'] == 1 else 'Active Customer'}`")
        
    with col_right:
        st.subheader("💡 Grounded AI Retention Advisor")
        st.caption("2-sentence diagnostic + 1 concrete action ($20 budget) grounded in top SHAP drivers.")
        
        # Display Driver Pills
        st.markdown("<b>Algorithmic Churn Drivers:</b>", unsafe_allow_html=True)
        pills_html = "".join([f'<span class="driver-pill">⚠️ {d["feature"]} (+{d["shap_value"]:.2f})</span>' for d in top_drivers[:3]])
        st.markdown(pills_html, unsafe_allow_html=True)
        
        # Generate Advice
        advice = generate_retention_advice(
            customer_row=customer_row.to_dict(),
            top_drivers=top_drivers,
            churn_prob=churn_prob,
            api_key=api_key_input,
            api_provider=llm_provider
        )
        
        st.markdown(f"""
        <div class="advice-box">
            <div style="font-weight: 700; color: #15803D; font-size: 0.95rem; margin-bottom: 6px;">
                📋 Diagnostic Summary:
            </div>
            <div style="color: #1E293B; font-size: 0.92rem; line-height: 1.5; margin-bottom: 12px;">
                {advice['summary']}
            </div>
            <div style="font-weight: 700; color: #15803D; font-size: 0.95rem; margin-bottom: 6px;">
                🎯 Recommended Action (Target Budget: ~${cost_offer:.0f}):
            </div>
            <div style="color: #0F172A; font-weight: 600; font-size: 0.92rem; line-height: 1.5;">
                {advice['action']}
            </div>
            <div style="font-size: 0.75rem; color: #64748B; margin-top: 10px; text-align: right;">
                Source: <i>{advice['source']}</i>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Interactive Retention What-If Simulator
        with st.expander("🧪 What-If Retention Simulation Sandbox", expanded=True):
            st.caption("Test how applying retention interventions changes this customer's predicted churn risk.")
            sim_col1, sim_col2 = st.columns(2)
            with sim_col1:
                sim_contract = st.selectbox(
                    "Simulate Contract Change:",
                    ["No Change", "Upgrade to One year", "Upgrade to Two year"]
                )
            with sim_col2:
                sim_support = st.selectbox(
                    "Simulate Tech Support Addition:",
                    ["No Change", "Add 24/7 Tech Support"]
                )
                
            # Create simulated copy and recalculate
            sim_customer = customer_row.copy()
            if sim_contract == "Upgrade to One year":
                sim_customer["Contract"] = "One year"
            elif sim_contract == "Upgrade to Two year":
                sim_customer["Contract"] = "Two year"
                
            if sim_support == "Add 24/7 Tech Support":
                sim_customer["TechSupport"] = "Yes"
                
            # Build mini-dataframe and transform
            sim_features_df = pd.DataFrame([sim_customer])[artifacts["numeric_features"] + artifacts["categorical_features"]]
            sim_proc = preprocessor.transform(sim_features_df)
            new_prob = float(xgb_model.predict_proba(sim_proc)[0, 1])
            prob_diff = new_prob - churn_prob
            
            s1, s2, s3 = st.columns(3)
            s1.metric("Original Churn Risk", f"{churn_prob*100:.1f}%")
            s2.metric("Simulated Risk Post-Action", f"{new_prob*100:.1f}%", delta=f"{prob_diff*100:.1f}%", delta_color="inverse")
            s3.metric("Retention Status", "SAFE ✅" if new_prob < selected_threshold else "STILL AT RISK ⚠️")
            
    # Full Width SHAP Attribution Waterfall
    st.divider()
    st.subheader("📊 SHAP Feature Attribution Waterfall")
    st.caption("How each specific customer attribute pushes churn risk above or below the baseline population average (Log-Odds space).")
    
    # Extract customer's SHAP values and feature values
    row_shap = shap_data["shap_values"][test_idx]
    clean_feature_names = [f.replace("cat__", "").replace("num__", "").replace("_", " ") for f in feature_names]
    
    # Sort by absolute SHAP impact and take top 12
    top_indices = np.argsort(np.abs(row_shap))[::-1][:12]
    plot_feats = [clean_feature_names[i] for i in top_indices][::-1]
    plot_shaps = [row_shap[i] for i in top_indices][::-1]
    colors = ["#DC2626" if val > 0 else "#16A34A" for val in plot_shaps]
    
    shap_bar_fig = go.Figure(go.Bar(
        x=plot_shaps,
        y=plot_feats,
        orientation='h',
        marker=dict(color=colors),
        text=[f"{val:+.3f}" for val in plot_shaps],
        textposition="outside"
    ))
    
    shap_bar_fig.update_layout(
        title=f"Top Feature Contributions for Customer {selected_cid} (Red = Pushes Risk Up, Green = Lowers Risk)",
        xaxis_title="SHAP Value (Impact on Log-Odds of Churn)",
        yaxis_title="Feature",
        height=420,
        margin=dict(l=150, r=40, t=50, b=40)
    )
    st.plotly_chart(shap_bar_fig, use_container_width=True)

# =============================================================
# PAGE 3: SEGMENT ANALYTICS & COST OPTIMIZATION CURVE
# =============================================================
elif page == "📊 Segment Analytics & Cost Curve":
    st.markdown('<div class="main-header">📊 Segment Analytics & Business Cost Optimization</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Financial threshold optimization balancing intervention expense vs customer lifetime value loss, '
        'plus macro-cohort segment patterns.</div>',
        unsafe_allow_html=True
    )
    
    # Section 1: Business Cost Curve
    st.subheader("💰 Cost Optimization Curve")
    st.markdown(
        f"In traditional ML, practitioners blindly default to a 0.50 cutoff. In telecommunications, false negatives "
        f"(lost customers @ **${cost_churn:,.0f}**) are 25x more catastrophic than false positives "
        f"(retention offers @ **${cost_offer:,.0f}**). Our decision threshold curve finds the mathematically optimal policy cutoff."
    )
    
    cost_df = cost_meta["cost_df"].copy()
    # Recalculate dynamic costs if user modified cost sliders
    cost_df["recalculated_cost"] = (cost_df["fn"] * cost_churn) + ((cost_df["tp"] + cost_df["fp"]) * cost_offer)
    cost_df["recalculated_savings"] = cost_do_nothing - cost_df["recalculated_cost"]
    
    min_cost_idx = cost_df["recalculated_cost"].idxmin()
    dynamic_optimal_threshold = cost_df.loc[min_cost_idx, "threshold"]
    dynamic_min_cost = cost_df.loc[min_cost_idx, "recalculated_cost"]
    dynamic_max_savings = cost_df.loc[min_cost_idx, "recalculated_savings"]
    
    # Cost at 0.50 cutoff
    row_50 = cost_df.iloc[(cost_df["threshold"] - 0.50).abs().argsort()[:1]].iloc[0]
    cost_at_50 = row_50["recalculated_cost"]
    savings_at_50 = row_50["recalculated_savings"]
    incremental_roi = cost_at_50 - dynamic_min_cost
    
    # 3 Comparison Metric Cards
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Do Nothing Baseline Cost", f"${cost_do_nothing:,.0f}", help="Total cost if all churners leave with zero outreach.")
    c2.metric("Default 0.50 Cutoff Cost", f"${cost_at_50:,.0f}", f"Net Savings: ${savings_at_50:,.0f}")
    c3.metric(f"Optimal Cutoff Cost (t={dynamic_optimal_threshold:.2f})", f"${dynamic_min_cost:,.0f}", f"Max Savings: ${dynamic_max_savings:,.0f}", delta_color="normal")
    c4.metric("Incremental Profit vs 0.50", f"+${incremental_roi:,.0f}", "Pure business gain", delta_color="normal")
    
    # Interactive Cost Plotly Line Chart
    cost_fig = go.Figure()
    cost_fig.add_trace(go.Scatter(
        x=cost_df["threshold"],
        y=cost_df["recalculated_cost"],
        mode="lines",
        name="Total Business Cost ($)",
        line=dict(color="#2563EB", width=3)
    ))
    
    # Add benchmark horizontal line for doing nothing
    cost_fig.add_hline(
        y=cost_do_nothing,
        line_dash="dot",
        line_color="#DC2626",
        annotation_text=f"Do Nothing ($ {cost_do_nothing:,.0f})",
        annotation_position="bottom right"
    )
    
    # Add vertical line for default 0.50
    cost_fig.add_vline(
        x=0.50,
        line_dash="dash",
        line_color="#64748B",
        annotation_text=f"Default 0.50 ($ {cost_at_50:,.0f})",
        annotation_position="top right"
    )
    
    # Add vertical line for optimal threshold
    cost_fig.add_vline(
        x=dynamic_optimal_threshold,
        line_dash="solid",
        line_color="#16A34A",
        annotation_text=f"Optimal {dynamic_optimal_threshold:.2f} ($ {dynamic_min_cost:,.0f})",
        annotation_position="top left"
    )
    
    cost_fig.update_layout(
        title="Business Cost as a Function of Churn Probability Threshold",
        xaxis_title="Classification Threshold (t)",
        yaxis_title="Total Cost ($)",
        height=400,
        margin=dict(l=40, r=40, t=50, b=40),
        hovermode="x unified"
    )
    st.plotly_chart(cost_fig, use_container_width=True)
    
    st.divider()
    # Section 2: Segment Patterns (EDA)
    st.subheader("📈 Macro Segment Churn Patterns")
    
    seg1, seg2 = st.columns(2)
    with seg1:
        # Churn by Contract Type
        contract_churn = full_df.groupby("Contract")["Churn_Binary"].agg(["count", "mean"]).reset_index()
        contract_churn["Churn_Rate"] = contract_churn["mean"] * 100
        
        fig_contract = px.bar(
            contract_churn,
            x="Contract",
            y="Churn_Rate",
            color="Contract",
            text=contract_churn["Churn_Rate"].apply(lambda x: f"{x:.1f}%"),
            title="Churn Rate by Contract Type (Month-to-Month vs Annual)",
            color_discrete_sequence=["#EF4444", "#3B82F6", "#10B981"]
        )
        fig_contract.update_layout(showlegend=False, yaxis_title="Churn Rate (%)", height=320)
        st.plotly_chart(fig_contract, use_container_width=True)
        
    with seg2:
        # Churn by Internet Service & Tech Support
        service_churn = full_df.groupby(["InternetService", "TechSupport"])["Churn_Binary"].mean().reset_index()
        service_churn["Churn_Rate"] = service_churn["Churn_Binary"] * 100
        
        fig_service = px.bar(
            service_churn,
            x="InternetService",
            y="Churn_Rate",
            color="TechSupport",
            barmode="group",
            title="Impact of Tech Support on Churn by Internet Type",
            color_discrete_sequence=["#EF4444", "#10B981", "#94A3B8"]
        )
        fig_service.update_layout(yaxis_title="Churn Rate (%)", height=320)
        st.plotly_chart(fig_service, use_container_width=True)
        
    seg3, seg4 = st.columns(2)
    with seg3:
        # Churn vs Tenure Distribution
        fig_tenure = px.histogram(
            full_df,
            x="tenure",
            color="Churn",
            barmode="overlay",
            nbins=36,
            title="Tenure Distribution by Churn Status (Early Cliff Effect)",
            color_discrete_map={"Yes": "#EF4444", "No": "#10B981"},
            opacity=0.75
        )
        fig_tenure.update_layout(xaxis_title="Tenure (Months)", yaxis_title="Customer Count", height=320)
        st.plotly_chart(fig_tenure, use_container_width=True)
        
    with seg4:
        # Global Feature Importance (SHAP)
        global_imp = shap_data["global_importance"].head(10)
        fig_imp = px.bar(
            global_imp,
            x="importance",
            y="feature",
            orientation="h",
            title="Top 10 Global Churn Drivers (Mean |SHAP| across Test Cohort)",
            color="importance",
            color_continuous_scale="Reds"
        )
        fig_imp.update_layout(yaxis=dict(autorange="reversed"), xaxis_title="Mean |SHAP Value|", height=320, coloraxis_showscale=False)
        st.plotly_chart(fig_imp, use_container_width=True)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #94A3B8; font-size: 0.85rem;'>"
    "Customer Churn Early-Warning Dashboard • Built with Scikit-Learn, XGBoost, SHAP & Streamlit • Enterprise AI Retention Architecture"
    "</div>",
    unsafe_allow_html=True
)
