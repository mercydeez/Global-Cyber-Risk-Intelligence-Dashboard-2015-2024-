"""
Global Cyber Risk Intelligence Dashboard (2015–2024)
=====================================================
Enterprise-grade Streamlit analytics platform for C-Suite stakeholders.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.model_selection import learning_curve, train_test_split
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================
st.set_page_config(
    page_title="Global Cyber Risk Intelligence Dashboard",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# FIXED PROFESSIONAL STYLING - HIGH CONTRAST, READABLE
# =============================================================================
st.markdown("""
<style>
    /* Base palette */
    :root {
        --primary: #1f4ed8;
        --secondary: #0f766e;
        --accent: #f59e0b;
        --risk: #dc2626;
        --bg: #f8fafc;
        --white: #ffffff;
        --text: #0f172a;
    }

    /* Global page styling */
    .main {
        background-color: var(--bg);
        color: var(--text);
    }

    /* Typography */
    h1 { 
        font-size: 32px; 
        margin-bottom: 8px;
        color: #1e40af !important;
    }
    h2 { 
        font-size: 24px; 
        margin-bottom: 6px;
        color: #1e3a8a !important;
    }
    h3 { 
        font-size: 20px; 
        margin-bottom: 4px;
        color: #1e293b !important;
    }

    /* Vertical spacing */
    .block-container { 
        padding-top: 1rem;
        padding-bottom: 2rem;
        max-width: 1400px;
    }
    .stPlotlyChart { margin: 12px 0 28px 0; }
    .stMarkdown { margin: 8px 0 16px 0; }
    
    /* Section headers */
    .element-container:has(h2) {
        margin-top: 32px;
        margin-bottom: 16px;
    }

    /* ===== FIXED SIDEBAR STYLING - DARK BACKGROUND, WHITE TEXT ===== */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e293b 0%, #334155 100%) !important;
    }
    
    section[data-testid="stSidebar"] > div {
        background: transparent !important;
        border-right: 3px solid #475569 !important;
    }
    
    /* Sidebar text - ALL WHITE */
    section[data-testid="stSidebar"] .stMarkdown,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] div {
        color: #f1f5f9 !important;
    }
    
    /* Sidebar headers - BRIGHT WHITE */
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: #ffffff !important;
        font-weight: 700 !important;
    }
    
    /* Sidebar input labels */
    section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] {
        color: #f8fafc !important;
        font-weight: 600 !important;
    }
    
    /* Sidebar metric values */
    section[data-testid="stSidebar"] [data-testid="stMetricValue"] {
        color: #fbbf24 !important;
        font-weight: 700 !important;
    }
    
    section[data-testid="stSidebar"] [data-testid="stMetricLabel"] {
        color: #e5e7eb !important;
    }

    /* Sidebar info boxes */
    section[data-testid="stSidebar"] .stAlert {
        background-color: rgba(59, 130, 246, 0.2) !important;
        border-left: 4px solid #3b82f6 !important;
    }
    
    section[data-testid="stSidebar"] .stAlert p {
        color: #e0e7ff !important;
    }

    /* ===== FIXED METRIC CARDS - GRADIENT WITH DARK TEXT ===== */
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%) !important;
        border-radius: 12px !important;
        padding: 24px !important;
        border: 2px solid #93c5fd !important;
        min-height: 120px !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1) !important;
    }
    
    div[data-testid="stMetric"] label {
        color: #1e40af !important;
        font-weight: 700 !important;
        font-size: 15px !important;
    }
    
    [data-testid="stMetricValue"] {
        font-size: 32px !important;
        color: #1e3a8a !important;
        font-weight: 800 !important;
    }
    
    [data-testid="stMetricDelta"] {
        color: #475569 !important;
        font-weight: 600 !important;
    }

    /* ===== TABS STYLING ===== */
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
        background: linear-gradient(135deg, #eef2ff 0%, #e0e7ff 100%);
        padding: 12px;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 10px 18px;
        font-weight: 600;
        color: #1e293b;
        border: 2px solid #e2e8f0;
        transition: all 0.2s ease;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%);
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
        color: #ffffff !important;
        border: 2px solid #1e40af;
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.4);
    }

    /* Dataframe styling */
    .dataframe { 
        font-size: 14px;
        color: #1e293b !important;
    }
    .stDataFrame div[role="table"] { 
        margin-bottom: 16px;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# DATA LOADING & PREPROCESSING
# =============================================================================
@st.cache_data
def load_and_prepare_data():
    """Load CSV and prepare data for analysis."""
    df = pd.read_csv("Global_Cybersecurity_Threats_2015-2024.csv")
    
    df.columns = [
        'Country', 'Year', 'Attack Type', 'Industry',
        'Financial Loss', 'Affected Users', 'Attack Source',
        'Security Vulnerability', 'Defense Mechanism', 'Resolution Time'
    ]
    
    region_map = {
        'USA': 'North America',
        'Canada': 'North America',
        'UK': 'Europe',
        'Germany': 'Europe',
        'France': 'Europe',
        'Russia': 'Europe',
        'China': 'Asia',
        'India': 'Asia',
        'Japan': 'Asia',
        'Australia': 'Oceania',
        'Brazil': 'South America'   
    }
    df['Region'] = df['Country'].map(region_map).fillna('Other')
    
    country_name_map = {
        'UK': 'United Kingdom',
        'USA': 'United States'
    }
    df['Country_Name'] = df['Country'].replace(country_name_map)
    
    df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
    df['Financial Loss'] = pd.to_numeric(df['Financial Loss'], errors='coerce')
    df['Affected Users'] = pd.to_numeric(df['Affected Users'], errors='coerce')
    df['Resolution Time'] = pd.to_numeric(df['Resolution Time'], errors='coerce')
    
    df = df.dropna(subset=['Year', 'Financial Loss', 'Region'])
    
    return df

df_original = load_and_prepare_data()

# =============================================================================
# SIDEBAR: FILTERS & CONTROLS
# =============================================================================
st.sidebar.title("🎛️ Dashboard Controls")
st.sidebar.markdown("---")

st.sidebar.subheader("📅 Time Period")
year_range = st.sidebar.slider(
    "Select Year Range",
    min_value=int(df_original["Year"].min()),
    max_value=int(df_original["Year"].max()),
    value=(int(df_original["Year"].min()), int(df_original["Year"].max())),
    help="Filter data by year range"
)

st.sidebar.subheader("🌍 Geographic Filter")
available_regions = sorted(df_original["Region"].unique())
selected_regions = st.sidebar.multiselect(
    "Select Regions",
    options=available_regions,
    default=available_regions,
    help="Choose one or more regions to analyze"
)

st.sidebar.subheader("🏢 Industry Filter")
available_industries = sorted(df_original["Industry"].unique())
_default_top_industries = (
    df_original.groupby("Industry")["Financial Loss"].sum().sort_values(ascending=False).head(6).index.tolist()
    if len(available_industries) > 6 else available_industries
)
selected_industries = st.sidebar.multiselect(
    "Select Industries",
    options=available_industries,
    default=_default_top_industries,
    help="Choose one or more industries to analyze"
)

st.sidebar.markdown("---")
st.sidebar.info(
    "💡 **Tip:** Use filters to drill down into specific regions, "
    "time periods, or industries for focused analysis."
)

df = df_original[
    (df_original["Year"].between(year_range[0], year_range[1])) &
    (df_original["Region"].isin(selected_regions if selected_regions else df_original["Region"])) &
    (df_original["Industry"].isin(selected_industries if selected_industries else df_original["Industry"]))
].copy()

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Current Selection")
st.sidebar.metric("Total Incidents", f"{len(df):,}")
st.sidebar.metric("Total Loss", f"${df['Financial Loss'].sum():,.1f}M")

if df.empty:
    st.warning("No data for the current filters. Adjust selections in the sidebar to view insights.")
    st.stop()

# =============================================================================
# HEADER & EXECUTIVE SUMMARY
# =============================================================================
st.title("🔐 Global Cyber Risk Intelligence Dashboard (2015–2024)")
st.markdown(
    "**Executive-grade analytics platform** providing strategic visibility into cyber threats, "
    "financial exposure, defense effectiveness, and predictive risk assessment."
)
st.markdown("---")

# =============================================================================
# KEY PERFORMANCE INDICATORS (KPIs)
# =============================================================================
kpi1, kpi2, kpi3, kpi4 = st.columns(4)

with kpi1:
    total_loss = df['Financial Loss'].sum()
    st.metric(
        label="💰 Total Financial Loss",
        value=f"${total_loss:,.1f}M",
        delta="Million USD",
        help="Cumulative financial impact across all incidents"
    )

with kpi2:
    avg_resolution = df['Resolution Time'].mean()
    st.metric(
        label="⏱️ Avg Resolution Time",
        value=f"{avg_resolution:.1f} hrs",
        delta="Hours",
        help="Average time to resolve cyber incidents"
    )

with kpi3:
    total_users = df['Affected Users'].sum()
    st.metric(
        label="👥 Total Affected Users",
        value=f"{total_users:,.0f}",
        delta="Cumulative",
        help="Total number of users impacted by incidents"
    )

with kpi4:
    country_count = df["Country"].nunique()
    st.metric(
        label="🌍 Countries Impacted",
        value=f"{country_count}",
        delta="Unique",
        help="Number of distinct countries affected"
    )

st.markdown("---")

# =============================================================================
# TABBED INTERFACE FOR DETAILED ANALYSIS
# =============================================================================
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📋 Executive Overview",
    "🎯 Threat & Risk Analysis",
    "💵 Financial & Industry Impact",
    "🗺️ Geospatial Intelligence",
    "🛡️ Defense & Controls",
    "🤖 ML Evaluation & Insights"
])

# =============================================================================
# TAB 1: EXECUTIVE OVERVIEW
# =============================================================================
with tab1:
    st.header("Executive Overview: One-Glance Risk Assessment")
    st.markdown("_Strategic view of cyber risk landscape for boardroom decision-making_")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("💰 Financial Loss by Region")
        st.markdown("_Color-coded regional exposure analysis_")
        
        region_loss = df.groupby("Region")["Financial Loss"].sum().reset_index().sort_values("Financial Loss", ascending=False)
        fig_reg = px.bar(
            region_loss,
            x="Region",
            y="Financial Loss",
            color="Region",
            color_discrete_sequence=["#1f4ed8", "#0f766e", "#f59e0b", "#dc2626", "#6366f1", "#0ea5a4", "#fb923c"],
            title=None,
        )
        fig_reg.update_layout(margin=dict(l=10,r=10,t=10,b=10), yaxis_title="Loss ($M)")
        st.plotly_chart(fig_reg, use_container_width=True)
        st.caption(f"🔴 Highest: {region_loss.iloc[0]['Region']} (${region_loss.iloc[0]['Financial Loss']:.1f}M)")
    
    with col2:
        st.subheader("📈 Year-over-Year Financial Loss Trend")
        st.markdown("_Temporal evolution of financial impact_")
        
        yearly_loss = df.groupby("Year")["Financial Loss"].sum().reset_index()
        fig_year = px.line(
            yearly_loss, x="Year", y="Financial Loss",
            markers=True,
            color_discrete_sequence=["#1f4ed8"],
        )
        fig_year.update_layout(margin=dict(l=10,r=10,t=10,b=10), yaxis_title="Loss ($M)")
        st.plotly_chart(fig_year, use_container_width=True)
        
        if len(yearly_loss) > 1 and yearly_loss['Financial Loss'].iloc[0] not in [0, np.nan]:
            base = yearly_loss['Financial Loss'].iloc[0]
            trend = ((yearly_loss['Financial Loss'].iloc[-1] - base) / base * 100) if base != 0 else np.nan
            if not np.isnan(trend):
                st.caption(f"📊 Overall trend: {trend:+.1f}% change from start to end")
            else:
                st.caption("📊 Overall trend: N/A (insufficient baseline)")
    
    st.markdown("---")
    
    st.subheader("🎯 Financial Loss by Attack Type (Waterfall Analysis)")
    st.markdown(
        "_Identifies which threat vectors contribute most to financial exposure. "
        "Focus defense investments on top contributors._"
    )
    
    attack_loss = df.groupby("Attack Type")["Financial Loss"].sum().reset_index().sort_values("Financial Loss", ascending=False)
    attack_loss['Percentage'] = (attack_loss['Financial Loss'] / attack_loss['Financial Loss'].sum() * 100).round(1)
    
    # Create waterfall chart
    total_loss = attack_loss['Financial Loss'].sum()
    
    fig_attack = go.Figure(go.Waterfall(
        name="Financial Loss",
        orientation="v",
        measure=["relative"] * len(attack_loss) + ["total"],
        x=list(attack_loss['Attack Type']) + ["Total"],
        textposition="outside",
        text=[f"${x:.1f}M" for x in attack_loss['Financial Loss']] + [f"${total_loss:.1f}M"],
        y=list(attack_loss['Financial Loss']) + [total_loss],
        connector={"line": {"color": "rgb(63, 63, 63)"}},
        increasing={"marker": {"color": "#60a5fa"}},
        decreasing={"marker": {"color": "#ef4444"}},
        totals={"marker": {"color": "#10b981"}}
    ))
    
    fig_attack.update_layout(
        margin=dict(l=10, r=10, t=10, b=10),
        yaxis_title="Loss ($M)",
        showlegend=False
    )
    st.plotly_chart(fig_attack, use_container_width=True)
    
    if not attack_loss.empty and attack_loss['Financial Loss'].sum() > 0:
        top_threat = attack_loss.iloc[0]
        st.caption(
            f"🔴 Top threat: **{top_threat['Attack Type']}** accounts for "
            f"**${top_threat['Financial Loss']:.1f}M** ({top_threat['Percentage']:.1f}% of total loss)"
        )
        
        # Show breakdown for all attack types
        breakdown_text = " | ".join([f"{row['Attack Type']}: ${row['Financial Loss']:.1f}M ({row['Percentage']:.1f}%)" 
                                     for _, row in attack_loss.head(3).iterrows()])
        st.caption(f"📊 Top 3: {breakdown_text}")
    else:
        st.caption("Top threat share N/A (no financial loss recorded in selection)")
    
    st.markdown("---")
    
    st.subheader("🏢 Industry Vulnerability Drill-Down")
    st.markdown("_Sectoral exposure ranked by financial impact and incident frequency_")
    
    industry_metrics = df.groupby("Industry").agg({
        'Financial Loss': 'sum',
        'Country': 'count',
        'Affected Users': 'sum',
        'Resolution Time': 'mean'
    }).rename(columns={
        'Country': 'Incident Count',
        'Affected Users': 'Total Users Affected',
        'Resolution Time': 'Avg Resolution (hrs)'
    }).round(2)
    
    industry_metrics = industry_metrics.sort_values('Financial Loss', ascending=False)
    industry_metrics['Financial Loss'] = industry_metrics['Financial Loss'].round(2)
    
    st.dataframe(
        industry_metrics,
        use_container_width=True,
        height=300
    )
    st.caption("💡 **Action:** Industries with high loss + high incident count require priority risk mitigation")

# =============================================================================
# TAB 2: THREAT ANALYSIS
# =============================================================================
with tab2:
    st.header("Threat Analysis: Attack Patterns & Emerging Risks")
    st.markdown("_Identify dominant threats, track evolution, and apply Pareto prioritization_")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Pareto Analysis: Attack Types")
        st.markdown(
            "_80/20 rule: Focus on top threats driving majority of financial loss_"
        )
        
        attack_loss = df.groupby("Attack Type")["Financial Loss"].sum().reset_index().sort_values("Financial Loss", ascending=False)
        attack_loss["Cum %"] = (attack_loss["Financial Loss"].cumsum() / attack_loss["Financial Loss"].sum() * 100)
        
        fig_pareto = make_subplots(specs=[[{"secondary_y": True}]])
        fig_pareto.add_trace(
            go.Bar(x=attack_loss['Attack Type'], y=attack_loss['Financial Loss'], name='Loss ($M)',
                   marker_color="#60a5fa"), secondary_y=False
        )
        fig_pareto.add_trace(
            go.Scatter(x=attack_loss['Attack Type'], y=attack_loss['Cum %'], name='Cumulative %',
                       mode='lines+markers', marker_color="#fbbf24", line=dict(width=3)), secondary_y=True
        )
        fig_pareto.update_yaxes(title_text="Loss ($M)", secondary_y=False)
        fig_pareto.update_yaxes(title_text="Cumulative %", range=[0,100], secondary_y=True)
        fig_pareto.update_layout(margin=dict(l=10,r=10,t=10,b=10))
        st.plotly_chart(fig_pareto, use_container_width=True)
        
        top_3_pct = attack_loss['Cum %'].iloc[min(2, len(attack_loss)-1)]
        if len(attack_loss) >= 2:
            st.caption(
                f"🎯 **Pareto Insight:** Top 3 attack types account for {top_3_pct:.1f}% of total loss"
            )
        st.caption("🔵 Blue bars = loss; 🟡 Orange line = cumulative share (80/20 focus)")
    
    with col2:
        st.subheader("📈 Attack Type Evolution Over Time")
        st.markdown("_Multi-line chart tracking threat trends_")
        
        attack_time = df.groupby(["Year", "Attack Type"]).size().reset_index(name='Incidents')
        top_attacks = df.groupby("Attack Type")["Financial Loss"].sum().nlargest(5).index
        attack_time_filtered = attack_time[attack_time['Attack Type'].isin(top_attacks)]
        fig_lines = px.line(
            attack_time_filtered, x='Year', y='Incidents', color='Attack Type',
            markers=True, color_discrete_sequence=px.colors.qualitative.Set2
        )
        fig_lines.update_layout(margin=dict(l=10,r=10,t=10,b=10))
        st.plotly_chart(fig_lines, use_container_width=True)
        st.caption("📊 Each colored line = different attack type (top 5 by loss)")
    
    st.markdown("---")
    
    st.subheader("🔍 Severity Clustering: Affected Users vs Financial Loss")
    st.markdown(
        "_Scatter plot reveals severity patterns. Outliers = high-impact incidents._"
    )
    
    scatter_data = df[['Affected Users', 'Financial Loss', 'Attack Type']].dropna()
    
    fig_scatter = px.scatter(
        scatter_data, x="Affected Users", y="Financial Loss", color="Attack Type",
        color_discrete_sequence=px.colors.qualitative.Set3, hover_data=['Attack Type']
    )
    fig_scatter.update_layout(margin=dict(l=10,r=10,t=10,b=10), yaxis_title="Loss ($M)")
    st.plotly_chart(fig_scatter, use_container_width=True)
    
    corr = scatter_data['Affected Users'].corr(scatter_data['Financial Loss']) if len(scatter_data) > 2 else np.nan
    if not np.isnan(corr):
        st.caption(
            f"📊 **Correlation:** {corr:.3f} | "
            f"**Interpretation:** {'Strong positive' if corr > 0.7 else 'Moderate positive' if corr > 0.4 else 'Weak'} "
            f"relationship between user count and financial impact"
        )
    else:
        st.caption("📊 Correlation: N/A (insufficient data)")
    
    st.markdown("---")
    
    st.subheader("🎯 RFM Risk Segmentation")
    st.markdown(
        "**RFM Methodology (Recency-Frequency-Monetary):**\n"
        "- **Recency:** Most recent year attack type appeared\n"
        "- **Frequency:** Number of incidents\n"
        "- **Monetary:** Total financial loss\n\n"
        "**Executive Action:** Threats with high R+F+M scores require immediate defense investment"
    )
    
    rfm = df.groupby("Attack Type").agg({
        'Year': 'max',
        'Country': 'count',
        'Financial Loss': 'sum'
    }).rename(columns={
        'Year': 'Most Recent Year',
        'Country': 'Frequency (Incidents)',
        'Financial Loss': 'Monetary (Total $M)'
    }).round(2)
    
    # ===== FIX: Safe normalization with division by zero check =====
    def safe_normalize(series):
        """Normalize series to 0-100 scale, handling edge cases."""
        min_val = series.min()
        max_val = series.max()
        if max_val == min_val:  # All values are the same
            return pd.Series([50.0] * len(series), index=series.index)  # Return neutral score
        return ((series - min_val) / (max_val - min_val) * 100).round(1)
    
    rfm['Recency Score'] = safe_normalize(rfm['Most Recent Year'])
    rfm['Frequency Score'] = safe_normalize(rfm['Frequency (Incidents)'])
    rfm['Monetary Score'] = safe_normalize(rfm['Monetary (Total $M)'])
    
    rfm['RFM Composite'] = ((rfm['Recency Score'] + rfm['Frequency Score'] + rfm['Monetary Score']) / 3).round(1)
    
    rfm = rfm.sort_values('RFM Composite', ascending=False)
    
    st.dataframe(rfm, use_container_width=True, height=300)
    st.caption("🎯 **Priority Action:** Attack types with RFM Composite > 70 are high-priority threats")

# =============================================================================
# TAB 3: INDUSTRY & FINANCIAL IMPACT
# =============================================================================
with tab3:
    st.header("Industry & Financial Impact: Sectoral Exposure Analysis")
    st.markdown("_Understand business sector vulnerabilities for strategic risk mitigation_")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🏭 Financial Loss by Industry")
        st.markdown("_Color-coded sectoral exposure ranking_")
        
        industry_loss = df.groupby("Industry")["Financial Loss"].sum().reset_index().sort_values("Financial Loss", ascending=False)
        fig_ind = px.bar(
            industry_loss, x='Industry', y='Financial Loss', color='Industry',
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        fig_ind.update_layout(margin=dict(l=10,r=10,t=10,b=10), yaxis_title="Loss ($M)")
        st.plotly_chart(fig_ind, use_container_width=True)
        if not industry_loss.empty:
            st.caption(f"🔴 Highest risk sector: {industry_loss.iloc[0]['Industry']} (${industry_loss.iloc[0]['Financial Loss']:.1f}M)")
    
    with col2:
        st.subheader("⏱️ Resolution Time vs Financial Loss")
        st.markdown("_Does faster resolution reduce financial impact?_")
        
        scatter_data = df[['Resolution Time', 'Financial Loss', 'Industry']].dropna()
        
        # ===== FIX: Check for sufficient data before plotting =====
        if len(scatter_data) > 0:
            fig_sc2 = px.scatter(
                scatter_data, x='Resolution Time', y='Financial Loss', color='Industry',
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            fig_sc2.update_layout(margin=dict(l=10,r=10,t=10,b=10), yaxis_title="Loss ($M)")
            st.plotly_chart(fig_sc2, use_container_width=True)
            
            if len(scatter_data) > 2:
                corr = scatter_data['Resolution Time'].corr(scatter_data['Financial Loss'])
                col_a, col_b = st.columns(2)
                col_a.metric("Correlation", f"{corr:.3f}")
                col_b.metric(
                    "Interpretation",
                    f"{'Positive' if corr > 0 else 'Negative'} relationship"
                )
            else:
                st.caption("⚠️ Insufficient data for correlation analysis")
        else:
            st.warning("No data available for this visualization")
    
    st.markdown("---")
    
    st.markdown("##### Regional Summary")
    region_summary = df.groupby("Region").agg({
        'Financial Loss': ['sum', 'mean'],
        'Affected Users': 'sum',
        'Year': 'count',
        'Resolution Time': 'mean'
    }).round(2)
    region_summary.columns = ['Total Loss ($M)', 'Avg Loss ($M)', 'Total Users', 'Incidents', 'Avg Resolution (hrs)']
    region_summary = region_summary.sort_values('Total Loss ($M)', ascending=False)
    
    st.dataframe(region_summary, use_container_width=True, height=200)
    
    st.markdown("##### Country-Level Detail")
    geo_analysis = df.groupby(["Region", "Country"]).agg({
        'Financial Loss': ['sum', 'mean'],
        'Affected Users': 'sum',
        'Year': 'count'
    }).round(2)
    geo_analysis.columns = ['Total Loss ($M)', 'Avg Loss ($M)', 'Total Users', 'Incidents']
    geo_analysis = geo_analysis.sort_values('Total Loss ($M)', ascending=False)
    
    st.dataframe(geo_analysis, use_container_width=True, height=300)

# =============================================================================
# TAB 4: GEOSPATIAL INTELLIGENCE
# =============================================================================
with tab4:
    st.header("Geospatial Intelligence: Country & Regional Risk")
    st.markdown("_Identify geographic hotspots to inform regional investment_")
    
    geo1, geo2 = st.columns(2)
    
    with geo1:
        st.subheader("🗺️ Choropleth: Financial Loss by Country")
        country_loss = df.groupby('Country_Name')['Financial Loss'].sum().reset_index()
        fig_choro = px.choropleth(
            country_loss,
            locations='Country_Name',
            locationmode='country names',
            color='Financial Loss',
            color_continuous_scale='Blues',
            title=None
        )
        fig_choro.update_layout(margin=dict(l=10,r=10,t=10,b=10))
        st.plotly_chart(fig_choro, use_container_width=True)
    
    with geo2:
        st.subheader("🌐 Bubble Map: Country Exposure")
        country_metrics = df.groupby('Country').agg({
            'Financial Loss': 'sum',
            'Affected Users': 'sum'
        }).reset_index()
        
        fig_bubble = px.scatter(
            country_metrics,
            x='Country',
            y='Financial Loss',
            size='Affected Users',
            color='Financial Loss',
            color_continuous_scale='Reds',
            hover_data=['Affected Users']
        )
        fig_bubble.update_layout(margin=dict(l=10,r=10,t=10,b=10), yaxis_title="Loss ($M)")
        st.plotly_chart(fig_bubble, use_container_width=True)
        st.caption("🔵 Bubble size = affected users | Color intensity = financial loss")
    
    st.markdown("---")
    
    st.subheader("📊 Regional Incident Frequency")
    region_incidents = df.groupby('Region').size().reset_index(name='Incidents').sort_values('Incidents', ascending=False)
    fig_region_bar = px.bar(
        region_incidents,
        x='Region',
        y='Incidents',
        color='Region',
        color_discrete_sequence=px.colors.qualitative.Bold
    )
    fig_region_bar.update_layout(margin=dict(l=10,r=10,t=10,b=10))
    st.plotly_chart(fig_region_bar, use_container_width=True)

# =============================================================================
# TAB 5: DEFENSE MECHANISMS & CONTROLS
# =============================================================================
with tab5:
    st.header("Defense Mechanisms & Security Controls Effectiveness")
    st.markdown("_Evaluate which security measures provide best ROI and fastest resolution_")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🛡️ Defense Mechanism Effectiveness")
        st.markdown("_Resolution time by defense type (lower is better)_")
        
        defense_eff = df.groupby('Defense Mechanism').agg({
            'Resolution Time': 'mean',
            'Financial Loss': 'mean'
        }).round(2).reset_index().sort_values('Resolution Time')
        
        fig_def = px.bar(
            defense_eff,
            x='Defense Mechanism',
            y='Resolution Time',
            color='Financial Loss',
            color_continuous_scale='RdYlGn_r'
        )
        fig_def.update_layout(margin=dict(l=10,r=10,t=10,b=10), yaxis_title="Avg Resolution (hrs)")
        st.plotly_chart(fig_def, use_container_width=True)
        
        if not defense_eff.empty:
            st.caption(f"🟢 Best performer: {defense_eff.iloc[0]['Defense Mechanism']} ({defense_eff.iloc[0]['Resolution Time']:.1f} hrs avg)")
    
    with col2:
        st.subheader("🔒 Security Vulnerability Distribution")
        st.markdown("_Pie chart: root cause analysis_")
        
        vuln_dist = df.groupby('Security Vulnerability').size().reset_index(name='Count')
        fig_pie = px.pie(
            vuln_dist,
            values='Count',
            names='Security Vulnerability',
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        fig_pie.update_layout(margin=dict(l=10,r=10,t=10,b=10))
        st.plotly_chart(fig_pie, use_container_width=True)
    
    st.markdown("---")
    
    st.subheader("📊 Attack Source Analysis")
    source_metrics = df.groupby('Attack Source').agg({
        'Financial Loss': ['sum', 'mean'],
        'Resolution Time': 'mean',
        'Country': 'count'
    }).round(2)
    source_metrics.columns = ['Total Loss ($M)', 'Avg Loss ($M)', 'Avg Resolution (hrs)', 'Incidents']
    source_metrics = source_metrics.sort_values('Total Loss ($M)', ascending=False)
    
    st.dataframe(source_metrics, use_container_width=True, height=250)
    st.caption("💡 **Actionable Insight:** Attack sources with high loss and long resolution need priority security controls")

# =============================================================================
# TAB 6: MACHINE LEARNING EVALUATION & INSIGHTS
# =============================================================================
with tab6:
    st.header("Machine Learning: Predictive Risk Assessment")
    st.markdown("_Random Forest model evaluation for threat prediction_")
    
    # ===== FIX: Check for sufficient data before ML =====
    if len(df) < 10:
        st.error("⚠️ Insufficient data for machine learning analysis. At least 10 records required.")
        st.stop()
    
    # Prepare features for ML
    ml_data = df[['Attack Type', 'Industry', 'Region', 'Attack Source', 
                  'Security Vulnerability', 'Defense Mechanism', 'Financial Loss']].copy()
    
    # Create binary target (high risk = above median financial loss)
    median_loss = ml_data['Financial Loss'].median()
    ml_data['High_Risk'] = (ml_data['Financial Loss'] > median_loss).astype(int)
    
    # Encode categorical features
    from sklearn.preprocessing import LabelEncoder
    
    cat_cols = ['Attack Type', 'Industry', 'Region', 'Attack Source', 
                'Security Vulnerability', 'Defense Mechanism']
    
    le_dict = {}
    for col in cat_cols:
        le = LabelEncoder()
        ml_data[col + '_encoded'] = le.fit_transform(ml_data[col].astype(str))
        le_dict[col] = le
    
    # Prepare X and y
    feature_cols = [col + '_encoded' for col in cat_cols]
    X = ml_data[feature_cols]
    y = ml_data['High_Risk']
    
    # ===== FIX: Check for class balance =====
    if y.nunique() < 2:
        st.error("⚠️ Insufficient class diversity for classification. All records have same risk level.")
        st.stop()
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    # Train Random Forest
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
    rf_model.fit(X_train, y_train)
    
    # Predictions
    y_pred = rf_model.predict(X_test)
    y_pred_proba = rf_model.predict_proba(X_test)[:, 1]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 ROC Curve")
        st.markdown("_Model discrimination ability (AUC closer to 1.0 is better)_")
        
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'ROC (AUC = {roc_auc:.3f})',
                                     line=dict(color='#1f4ed8', width=3)))
        fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random Classifier',
                                     line=dict(color='gray', dash='dash')))
        fig_roc.update_layout(
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            margin=dict(l=10,r=10,t=10,b=10)
        )
        st.plotly_chart(fig_roc, use_container_width=True)
        st.metric("Model AUC Score", f"{roc_auc:.3f}")
    
    with col2:
        st.subheader("🎯 Confusion Matrix")
        st.markdown("_Classification accuracy breakdown_")
        
        cm = confusion_matrix(y_test, y_pred)
        fig_cm = px.imshow(
            cm,
            labels=dict(x="Predicted", y="Actual", color="Count"),
            x=['Low Risk', 'High Risk'],
            y=['Low Risk', 'High Risk'],
            color_continuous_scale='Blues',
            text_auto=True
        )
        fig_cm.update_layout(margin=dict(l=10,r=10,t=10,b=10))
        st.plotly_chart(fig_cm, use_container_width=True)
        
        accuracy = (cm[0,0] + cm[1,1]) / cm.sum()
        st.metric("Model Accuracy", f"{accuracy:.1%}")
    
    st.markdown("---")
    
    st.subheader("🔍 Feature Importance")
    st.markdown("_Which factors most predict high-risk incidents?_")
    
    feature_importance = pd.DataFrame({
        'Feature': cat_cols,
        'Importance': rf_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    fig_imp = px.bar(
        feature_importance,
        x='Importance',
        y='Feature',
        orientation='h',
        color='Importance',
        color_continuous_scale='Viridis'
    )
    fig_imp.update_layout(margin=dict(l=10,r=10,t=10,b=10))
    st.plotly_chart(fig_imp, use_container_width=True)
    st.caption(f"🎯 **Top predictor:** {feature_importance.iloc[0]['Feature']} (importance: {feature_importance.iloc[0]['Importance']:.3f})")
    
    st.markdown("---")
    
    st.subheader("📊 Learning Curve Analysis")
    st.markdown("_Model performance vs training data size_")
    
    # ===== FIX: Use unfitted estimator for learning_curve =====
    try:
        rf_estimator = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
        
        train_sizes, train_scores, val_scores = learning_curve(
            rf_estimator, X, y, cv=min(5, len(X) // 10),  # Adjust CV folds based on data size
            n_jobs=-1,
            train_sizes=np.linspace(0.1, 1.0, 10),
            random_state=42,
            scoring='accuracy'
        )
        
        train_mean = np.mean(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        
        fig_lc = go.Figure()
        fig_lc.add_trace(go.Scatter(x=train_sizes, y=train_mean, mode='lines+markers',
                                    name='Training Score', line=dict(color='#1f4ed8')))
        fig_lc.add_trace(go.Scatter(x=train_sizes, y=val_mean, mode='lines+markers',
                                    name='Validation Score', line=dict(color='#f59e0b')))
        fig_lc.update_layout(
            xaxis_title='Training Set Size',
            yaxis_title='Accuracy Score',
            margin=dict(l=10,r=10,t=10,b=10)
        )
        st.plotly_chart(fig_lc, use_container_width=True)
        st.caption("💡 **Insight:** Converging lines indicate good model generalization")
    except Exception as e:
        st.warning(f"⚠️ Learning curve could not be generated: {str(e)}")

# =============================================================================
# FOOTER
# =============================================================================
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #64748b; font-size: 13px;'>"
    "🔐 Global Cyber Risk Intelligence Dashboard | "
    f"Data Period: 2015–2024 | Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')} UTC | "
    "Confidential – Executive Use Only"
    "</div>",
    unsafe_allow_html=True
)