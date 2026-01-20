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
from sklearn.preprocessing import OrdinalEncoder
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
# UTILITY FUNCTIONS
# =============================================================================
def format_currency(value):
    """Format currency values compactly. Dataset uses $M, so convert accordingly."""
    if pd.isna(value) or np.isinf(value):
        return "$0"
    
    if value == 0:
        return "$0"
    
    abs_val = abs(value)
    sign = "-" if value < 0 else ""
    
    if abs_val >= 1_000:
        return f"{sign}${abs_val / 1_000:.1f}B"
    elif abs_val >= 1:
        return f"{sign}${abs_val:.1f}M"
    elif abs_val >= 0.001:
        return f"{sign}${abs_val * 1_000:.1f}K"
    else:
        return f"{sign}${abs_val * 1_000_000:.0f}"

def format_number(value):
    """Format large numbers compactly (K, M, B)."""
    if pd.isna(value) or np.isinf(value):
        return "0"
    
    if value == 0:
        return "0"
    
    abs_val = abs(value)
    sign = "-" if value < 0 else ""
    
    if abs_val >= 1_000_000_000:
        return f"{sign}{abs_val / 1_000_000_000:.1f}B"
    elif abs_val >= 1_000_000:
        return f"{sign}{abs_val / 1_000_000:.1f}M"
    elif abs_val >= 1_000:
        return f"{sign}{abs_val / 1_000:.1f}K"
    else:
        return f"{sign}{abs_val:.0f}"

def safe_normalize(series):
    """Normalize series to 0-100 scale, handling edge cases."""
    min_val = series.min()
    max_val = series.max()
    if max_val == min_val:
        return pd.Series([50.0] * len(series), index=series.index)
    return ((series - min_val) / (max_val - min_val) * 100).round(1)

# =============================================================================
# DATA LOADING & PREPROCESSING
# =============================================================================
@st.cache_data
def load_and_prepare_data():
    """Load CSV and prepare data for analysis."""
    try:
        df = pd.read_csv("Global_Cybersecurity_Threats_2015-2024.csv")
    except FileNotFoundError:
        st.error("❌ CSV file not found. Ensure 'Global_Cybersecurity_Threats_2015-2024.csv' exists.")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error loading CSV: {str(e)}")
        st.stop()
    
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
st.sidebar.metric("Total Loss", format_currency(df['Financial Loss'].sum()))

st.sidebar.markdown("---")
csv_export = df.to_csv(index=False).encode('utf-8')
st.sidebar.download_button(
    label="📥 Download Filtered Data",
    data=csv_export,
    file_name=f"cyber_risk_report_{datetime.now().strftime('%Y%m%d')}.csv",
    mime="text/csv",
    help="Export current filtered dataset as CSV"
)

if df.empty:
    st.warning("No data for the current filters. Adjust selections in the sidebar to view insights.")
    st.stop()

# =============================================================================
# EXECUTIVE WHITE THEME STYLING
# =============================================================================
st.markdown("""
<style>
    :root {
        --primary: #1f4ed8;
        --secondary: #0f766e;
        --accent: #f59e0b;
        --risk: #dc2626;
        --bg: #ffffff;
        --white: #ffffff;
        --text: #0f172a;
    }

    .main {
        background-color: #ffffff;
        color: var(--text);
    }

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

    .block-container { 
        padding-top: 1rem;
        padding-bottom: 2rem;
        max-width: 1400px;
    }
    .stPlotlyChart { margin: 12px 0 28px 0; }
    .stMarkdown { margin: 8px 0 16px 0; }
    
    .element-container:has(h2) {
        margin-top: 32px;
        margin-bottom: 16px;
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f1f5f9 0%, #e2e8f0 100%) !important;
    }
    
    section[data-testid="stSidebar"] > div {
        background: transparent !important;
        border-right: 3px solid #cbd5e1 !important;
    }
    
    section[data-testid="stSidebar"] .stMarkdown,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] div {
        color: #334155 !important;
    }
    
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: #1e293b !important;
        font-weight: 700 !important;
    }
    
    section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] {
        color: #475569 !important;
        font-weight: 600 !important;
    }
    
    section[data-testid="stSidebar"] [data-testid="stMetricValue"] {
        color: #1e40af !important;
        font-weight: 700 !important;
    }
    
    section[data-testid="stSidebar"] [data-testid="stMetricLabel"] {
        color: #64748b !important;
    }

    section[data-testid="stSidebar"] .stAlert {
        background-color: rgba(59, 130, 246, 0.1) !important;
        border-left: 4px solid #3b82f6 !important;
    }
    
    section[data-testid="stSidebar"] .stAlert p {
        color: #1e40af !important;
    }

    div[data-testid="stMetric"] {
        background: #ffffff !important;
        border-radius: 12px !important;
        padding: 24px !important;
        border: 2px solid #e5e7eb !important;
        min-height: 140px !important;
        height: 140px !important;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08) !important;
        display: flex !important;
        flex-direction: column !important;
        justify-content: space-between !important;
    }
    
    div[data-testid="stMetric"] label {
        color: #1e40af !important;
        font-weight: 700 !important;
        font-size: 15px !important;
        line-height: 1.4 !important;
        min-height: 42px !important;
        display: flex !important;
        align-items: center !important;
    }
    
    [data-testid="stMetricValue"] {
        font-size: 32px !important;
        color: #1e3a8a !important;
        font-weight: 800 !important;
        line-height: 1.2 !important;
        min-height: 38px !important;
        display: flex !important;
        align-items: center !important;
    }
    
    [data-testid="stMetricDelta"] {
        color: #475569 !important;
        font-weight: 600 !important;
        font-size: 13px !important;
        min-height: 20px !important;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
        background: #f8fafc;
        padding: 12px;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 10px 18px;
        font-weight: 600;
        color: #475569;
        border: 2px solid #e2e8f0;
        transition: all 0.2s ease;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: #f1f5f9;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
        color: #ffffff !important;
        border: 2px solid #1e40af;
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.4);
    }

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
        value=format_currency(total_loss),
        help="Dataset values are in USD Millions ($M), displayed in compact K/M/B format."
    )

with kpi2:
    avg_resolution = df['Resolution Time'].mean()
    avg_resolution_display = f"{avg_resolution:.1f} hrs" if not pd.isna(avg_resolution) else "N/A"
    st.metric(
        label="⏱️ Avg Resolution Time",
        value=avg_resolution_display,
        delta="Hours",
        help="Average time to resolve cyber incidents"
    )

with kpi3:
    total_users = df['Affected Users'].sum()
    total_users_display = format_number(total_users) if not pd.isna(total_users) else "0"
    st.metric(
        label="👥 Total Affected Users",
        value=total_users_display,
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

st.subheader("💡 Executive Intelligence Snapshot")

if not df.empty and df['Financial Loss'].sum() > 0:
    try:
        attack_grouped = df.groupby('Attack Type')['Financial Loss'].sum()
        if not attack_grouped.empty and attack_grouped.sum() > 0:
            top_attack = attack_grouped.idxmax()
            top_attack_pct = (attack_grouped.max() / attack_grouped.sum() * 100)
        else:
            top_attack = "N/A"
            top_attack_pct = 0
        
        industry_grouped = df.groupby('Industry')['Financial Loss'].sum()
        top_industry = industry_grouped.idxmax() if not industry_grouped.empty else "N/A"
        
        country_grouped = df.groupby('Country')['Financial Loss'].sum()
        top_country = country_grouped.idxmax() if not country_grouped.empty else "N/A"

        if df['Year'].nunique() > 1:
            yearly_totals = df.groupby('Year')['Financial Loss'].sum().sort_index()
            if len(yearly_totals) > 0 and yearly_totals.iloc[0] != 0:
                yoy_change = ((yearly_totals.iloc[-1] - yearly_totals.iloc[0]) / yearly_totals.iloc[0] * 100)
                yoy_text = f" Financial losses {'increased' if yoy_change > 0 else 'decreased'} by {abs(yoy_change):.1f}% over the period."
            else:
                yoy_text = ""
        else:
            yoy_text = ""

        insight_text = f"""
**Key Findings:**
- 🎯 **{top_attack}** is the dominant threat, accounting for **{top_attack_pct:.1f}%** of total financial damage
- 🏢 **{top_industry}** sector faces highest exposure
- 🌍 **{top_country}** leads in total incident impact
{yoy_text}
"""
        st.info(insight_text)
    except Exception:
        st.info("Executive intelligence snapshot unavailable due to data constraints.")
else:
    st.info("No data available for intelligence snapshot.")
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
        if not region_loss.empty and region_loss['Financial Loss'].sum() > 0:
            region_loss['Loss_Formatted'] = region_loss['Financial Loss'].apply(format_currency)
            fig_reg = px.bar(
                region_loss,
                x="Region",
                y="Financial Loss",
                color="Region",
                color_discrete_sequence=["#1f4ed8", "#0f766e", "#f59e0b", "#dc2626", "#6366f1", "#0ea5a4", "#fb923c"],
                title=None,
                template="plotly_white",
                custom_data=['Loss_Formatted']
            )
            fig_reg.update_traces(hovertemplate='<b>%{x}</b><br>Loss: %{customdata[0]}<extra></extra>')
            fig_reg.update_layout(margin=dict(l=10,r=10,t=10,b=10), yaxis_title="Loss (USD)")
            st.plotly_chart(fig_reg, use_container_width=True)
            st.caption(f"🔴 Highest: {region_loss.iloc[0]['Region']} ({format_currency(region_loss.iloc[0]['Financial Loss'])})")
        else:
            st.warning("No regional data available")
    
    with col2:
        st.subheader("📈 Year-over-Year Financial Loss Trend")
        st.markdown("_Temporal evolution of financial impact_")
        
        yearly_loss = df.groupby("Year")["Financial Loss"].sum().reset_index()
        if not yearly_loss.empty and yearly_loss['Financial Loss'].sum() > 0:
            yearly_loss['Loss_Formatted'] = yearly_loss['Financial Loss'].apply(format_currency)
            fig_year = px.line(
                yearly_loss, x="Year", y="Financial Loss",
                markers=True,
                color_discrete_sequence=["#1f4ed8"],
                template="plotly_white",
                custom_data=['Loss_Formatted']
            )
            fig_year.update_traces(hovertemplate='<b>Year %{x}</b><br>Loss: %{customdata[0]}<extra></extra>')
            fig_year.update_layout(margin=dict(l=10,r=10,t=10,b=10), yaxis_title="Loss (USD)")
            st.plotly_chart(fig_year, use_container_width=True)
            
            if len(yearly_loss) > 1 and yearly_loss['Financial Loss'].iloc[0] not in [0, np.nan]:
                base = yearly_loss['Financial Loss'].iloc[0]
                trend = ((yearly_loss['Financial Loss'].iloc[-1] - base) / base * 100) if base != 0 else np.nan
                if not np.isnan(trend):
                    st.caption(f"📊 Overall trend: {trend:+.1f}% change from start to end")
                else:
                    st.caption("📊 Overall trend: N/A (insufficient baseline)")
        else:
            st.warning("No yearly data available")
    
    st.markdown("---")
    
    st.subheader("🎯 Financial Loss by Attack Type (Waterfall Analysis)")
    st.markdown(
        "_Identifies which threat vectors contribute most to financial exposure. "
        "Focus defense investments on top contributors._"
    )
    
    attack_loss = df.groupby("Attack Type")["Financial Loss"].sum().reset_index().sort_values("Financial Loss", ascending=False)
    
    if not attack_loss.empty and attack_loss['Financial Loss'].sum() > 0:
        attack_loss['Percentage'] = (attack_loss['Financial Loss'] / attack_loss['Financial Loss'].sum() * 100).round(1)
        
        total_loss = attack_loss['Financial Loss'].sum()
        
        fig_attack = go.Figure(go.Waterfall(
            name="Financial Loss",
            orientation="v",
            measure=["relative"] * len(attack_loss) + ["total"],
            x=list(attack_loss['Attack Type']) + ["Total"],
            textposition="outside",
            text=[f"{format_currency(x)}\n({x/total_loss*100:.1f}%)" for x in attack_loss['Financial Loss']] + [format_currency(total_loss)],
            y=list(attack_loss['Financial Loss']) + [total_loss],
            connector={"line": {"color": "rgb(63, 63, 63)"}},
            increasing={"marker": {"color": "#60a5fa"}},
            decreasing={"marker": {"color": "#ef4444"}},
            totals={"marker": {"color": "#10b981"}}
        ))
        
        fig_attack.update_layout(
            margin=dict(l=10, r=10, t=10, b=10),
            yaxis_title="Loss (USD)",
            showlegend=False,
            template="plotly_white"
        )
        st.plotly_chart(fig_attack, use_container_width=True)
        
        top_threat = attack_loss.iloc[0]
        st.caption(
            f"🔴 Top threat: **{top_threat['Attack Type']}** accounts for "
            f"**{format_currency(top_threat['Financial Loss'])}** ({top_threat['Percentage']:.1f}% of total loss)"
        )
        
        breakdown_text = " | ".join([f"{row['Attack Type']}: {format_currency(row['Financial Loss'])} ({row['Percentage']:.1f}%)" 
                                     for _, row in attack_loss.head(3).iterrows()])
        st.caption(f"📊 Top 3: {breakdown_text}")
    else:
        st.warning("No attack type data available for waterfall analysis")
    
    st.markdown("---")
    
    st.subheader("🏢 Industry Vulnerability Drill-Down")
    st.markdown("_Sectoral exposure ranked by financial impact and incident frequency_")
    
    if not df.empty:
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
        industry_display = industry_metrics.copy()
        industry_display['Financial Loss'] = industry_metrics['Financial Loss'].apply(format_currency)
        industry_display = industry_display.rename(columns={'Financial Loss': 'Total Loss'})
        
        st.dataframe(
            industry_display,
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
        
        if not attack_loss.empty and attack_loss['Financial Loss'].sum() > 0:
            total_loss = attack_loss['Financial Loss'].sum()
            attack_loss["Cum %"] = (attack_loss["Financial Loss"].cumsum() / total_loss * 100)
            attack_loss['Loss_Formatted'] = attack_loss['Financial Loss'].apply(format_currency)
            
            fig_pareto = make_subplots(specs=[[{"secondary_y": True}]])
            fig_pareto.add_trace(
                go.Bar(x=attack_loss['Attack Type'], y=attack_loss['Financial Loss'], name='Loss',
                       marker_color="#60a5fa",
                       customdata=attack_loss['Loss_Formatted'],
                       hovertemplate='<b>%{x}</b><br>Loss: %{customdata}<extra></extra>'), secondary_y=False
            )
            fig_pareto.add_trace(
                go.Scatter(x=attack_loss['Attack Type'], y=attack_loss['Cum %'], name='Cumulative %',
                           mode='lines+markers', marker_color="#fbbf24", line=dict(width=3),
                           hovertemplate='<b>%{x}</b><br>Cumulative: %{y:.1f}%<extra></extra>'), secondary_y=True
            )
            fig_pareto.update_yaxes(title_text="Loss (USD)", secondary_y=False)
            fig_pareto.update_yaxes(title_text="Cumulative %", range=[0,100], secondary_y=True)
            fig_pareto.update_layout(margin=dict(l=10,r=10,t=10,b=10), template="plotly_white")
            st.plotly_chart(fig_pareto, use_container_width=True)
            
            top_3_pct = attack_loss['Cum %'].iloc[min(2, len(attack_loss)-1)] if len(attack_loss) > 2 else 0
            if len(attack_loss) >= 2:
                st.caption(
                    f"🎯 **Pareto Insight:** Top 3 attack types account for {top_3_pct:.1f}% of total loss"
                )
            st.caption("🔵 Blue bars = loss; 🟡 Orange line = cumulative share (80/20 focus)")
        else:
            st.warning("No attack type data available for Pareto analysis")
    
    with col2:
        st.subheader("📈 Attack Type Evolution Over Time")
        st.markdown("_Multi-line chart tracking threat trends_")
        
        attack_time = df.groupby(["Year", "Attack Type"]).size().reset_index(name='Incidents')
        top_attacks = df.groupby("Attack Type")["Financial Loss"].sum().nlargest(5).index
        attack_time_filtered = attack_time[attack_time['Attack Type'].isin(top_attacks)]
        
        if not attack_time_filtered.empty:
            fig_lines = px.line(
                attack_time_filtered, x='Year', y='Incidents', color='Attack Type',
                markers=True, color_discrete_sequence=px.colors.qualitative.Set2,
                template="plotly_white"
            )
            fig_lines.update_layout(margin=dict(l=10,r=10,t=10,b=10))
            st.plotly_chart(fig_lines, use_container_width=True)
            st.caption("📊 Each colored line = different attack type (top 5 by loss)")
        else:
            st.warning("No attack evolution data available")
    
    st.markdown("---")
    
    st.subheader("🔍 Severity Clustering: Affected Users vs Financial Loss")
    st.markdown(
        "_Scatter plot reveals severity patterns. Outliers = high-impact incidents._"
    )
    
    scatter_data = df[['Affected Users', 'Financial Loss', 'Attack Type']].dropna()
    
    if not scatter_data.empty and len(scatter_data) > 0 and scatter_data['Financial Loss'].sum() > 0:
        scatter_data = scatter_data.copy()
        scatter_data['Loss_Formatted'] = scatter_data['Financial Loss'].apply(format_currency)
        fig_scatter = px.scatter(
            scatter_data, x="Affected Users", y="Financial Loss", color="Attack Type",
            color_discrete_sequence=px.colors.qualitative.Set3,
            template="plotly_white",
            custom_data=['Attack Type', 'Loss_Formatted']
        )
        fig_scatter.update_traces(hovertemplate='<b>%{customdata[0]}</b><br>Users: %{x:,.0f}<br>Loss: %{customdata[1]}<extra></extra>')
        fig_scatter.update_layout(margin=dict(l=10,r=10,t=10,b=10), yaxis_title="Loss (USD)")
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
    else:
        st.warning("No data available for severity clustering visualization")
    
    st.markdown("---")
    
    st.subheader("🎯 RFM Risk Segmentation")
    st.markdown(
        "**RFM Methodology (Recency-Frequency-Monetary):**\n"
        "- **Recency:** Most recent year attack type appeared\n"
        "- **Frequency:** Number of incidents\n"
        "- **Monetary:** Total financial loss\n\n"
        "**Executive Action:** Threats with high R+F+M scores require immediate defense investment"
    )
    
    if not df.empty:
        rfm = df.groupby("Attack Type").agg({
            'Year': 'max',
            'Country': 'count',
            'Financial Loss': 'sum'
        }).rename(columns={
            'Year': 'Most Recent Year',
            'Country': 'Frequency (Incidents)',
            'Financial Loss': 'Monetary (Total $M)'
        }).round(2)
        
        rfm['Recency Score'] = safe_normalize(rfm['Most Recent Year'])
        rfm['Frequency Score'] = safe_normalize(rfm['Frequency (Incidents)'])
        rfm['Monetary Score'] = safe_normalize(rfm['Monetary (Total $M)'])
        
        rfm['RFM Composite'] = ((rfm['Recency Score'] + rfm['Frequency Score'] + rfm['Monetary Score']) / 3).round(1)
        
        loss_by_attack = df.groupby('Attack Type')['Financial Loss'].sum()
        loss_mean = loss_by_attack.mean()
        loss_std = loss_by_attack.std()
        
        if pd.notna(loss_std) and loss_std > 0:
            rfm['Anomaly'] = rfm['Monetary (Total $M)'].apply(
                lambda x: '🚨' if x > (loss_mean + 2 * loss_std) else ''
            )
        else:
            rfm['Anomaly'] = ''
        
        rfm['Monetary Display'] = rfm['Monetary (Total $M)'].apply(format_currency)
        
        rfm = rfm.sort_values('RFM Composite', ascending=False)
        
        rfm_display = rfm[['Most Recent Year', 'Frequency (Incidents)', 'Monetary Display', 
                           'Recency Score', 'Frequency Score', 'Monetary Score', 'RFM Composite', 'Anomaly']]
        
        st.dataframe(rfm_display, use_container_width=True, height=300)
        st.caption("🎯 **Priority Action:** Attack types with RFM Composite > 70 are high-priority threats")
    else:
        st.warning("No data available for RFM analysis")

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
        if not industry_loss.empty and industry_loss['Financial Loss'].sum() > 0:
            industry_loss['Loss_Formatted'] = industry_loss['Financial Loss'].apply(format_currency)
            fig_ind = px.bar(
                industry_loss, x='Industry', y='Financial Loss', color='Industry',
                color_discrete_sequence=px.colors.qualitative.Set2,
                template="plotly_white",
                custom_data=['Loss_Formatted']
            )
            fig_ind.update_traces(hovertemplate='<b>%{x}</b><br>Loss: %{customdata[0]}<extra></extra>')
            fig_ind.update_layout(margin=dict(l=10,r=10,t=10,b=10), yaxis_title="Loss (USD)")
            st.plotly_chart(fig_ind, use_container_width=True)
            st.caption(f"🔴 Highest risk sector: {industry_loss.iloc[0]['Industry']} ({format_currency(industry_loss.iloc[0]['Financial Loss'])})")
        else:
            st.warning("No industry data available")
    
    with col2:
        st.subheader("⏱️ Resolution Time vs Financial Loss")
        st.markdown("_Does faster resolution reduce financial impact?_")
        
        scatter_data = df[['Resolution Time', 'Financial Loss', 'Industry']].dropna().copy()
        
        if len(scatter_data) > 0 and scatter_data['Financial Loss'].sum() > 0:
            scatter_data['Loss_Formatted'] = scatter_data['Financial Loss'].apply(format_currency)
            fig_sc2 = px.scatter(
                scatter_data, x='Resolution Time', y='Financial Loss', color='Industry',
                color_discrete_sequence=px.colors.qualitative.Set3,
                template="plotly_white",
                custom_data=['Industry', 'Loss_Formatted']
            )
            fig_sc2.update_traces(hovertemplate='<b>%{customdata[0]}</b><br>Resolution: %{x:.1f} hrs<br>Loss: %{customdata[1]}<extra></extra>')
            fig_sc2.update_layout(margin=dict(l=10,r=10,t=10,b=10), yaxis_title="Loss (USD)")
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
    if not df.empty:
        region_summary = df.groupby("Region").agg({
            'Financial Loss': ['sum', 'mean'],
            'Affected Users': 'sum',
            'Year': 'count',
            'Resolution Time': 'mean'
        }).round(2)
        region_summary.columns = ['Total Loss', 'Avg Loss', 'Total Users', 'Incidents', 'Avg Resolution (hrs)']
        region_summary = region_summary.sort_values('Total Loss', ascending=False)
        region_display = region_summary.copy()
        region_display['Total Loss'] = region_summary['Total Loss'].apply(format_currency)
        region_display['Avg Loss'] = region_summary['Avg Loss'].apply(format_currency)
        
        st.dataframe(region_display, use_container_width=True, height=200)
    
    st.markdown("##### Country-Level Detail")
    if not df.empty:
        geo_analysis = df.groupby(["Region", "Country"]).agg({
            'Financial Loss': ['sum', 'mean'],
            'Affected Users': 'sum',
            'Year': 'count'
        }).round(2)
        geo_analysis.columns = ['Total Loss', 'Avg Loss', 'Total Users', 'Incidents']
        geo_analysis = geo_analysis.sort_values('Total Loss', ascending=False)
        geo_display = geo_analysis.copy()
        geo_display['Total Loss'] = geo_analysis['Total Loss'].apply(format_currency)
        geo_display['Avg Loss'] = geo_analysis['Avg Loss'].apply(format_currency)
        
        st.dataframe(geo_display, use_container_width=True, height=300)

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
        
        if not country_loss.empty and country_loss['Financial Loss'].sum() > 0 and len(country_loss) >= 3:
            try:
                country_loss_copy = country_loss.copy()
                country_loss_copy['Loss_Formatted'] = country_loss_copy['Financial Loss'].apply(format_currency)
                fig_choro = px.choropleth(
                    country_loss_copy,
                    locations='Country_Name',
                    locationmode='country names',
                    color='Financial Loss',
                    color_continuous_scale='Blues',
                    title=None,
                    template="plotly_white",
                    custom_data=['Loss_Formatted']
                )
                fig_choro.update_traces(hovertemplate='<b>%{location}</b><br>Loss: %{customdata[0]}<extra></extra>')
                fig_choro.update_layout(margin=dict(l=10,r=10,t=10,b=10))
                st.plotly_chart(fig_choro, use_container_width=True)
            except Exception:
                st.warning("⚠️ Choropleth map unavailable. Showing bar chart fallback.")
                country_loss_fallback = country_loss.copy()
                country_loss_fallback['Loss_Formatted'] = country_loss_fallback['Financial Loss'].apply(format_currency)
                top_countries = country_loss_fallback.nlargest(10, 'Financial Loss')
                fig_fallback = px.bar(
                    top_countries, x='Country_Name', y='Financial Loss',
                    color='Financial Loss', color_continuous_scale='Blues',
                    template="plotly_white", custom_data=['Loss_Formatted']
                )
                fig_fallback.update_traces(hovertemplate='<b>%{x}</b><br>Loss: %{customdata[0]}<extra></extra>')
                fig_fallback.update_layout(margin=dict(l=10,r=10,t=10,b=10), xaxis_title="Country", yaxis_title="Loss (USD)")
                st.plotly_chart(fig_fallback, use_container_width=True)
        else:
            st.warning("⚠️ Insufficient data for choropleth. Showing bar chart fallback.")
            if not country_loss.empty:
                country_loss_fallback = country_loss.copy()
                country_loss_fallback['Loss_Formatted'] = country_loss_fallback['Financial Loss'].apply(format_currency)
                top_countries = country_loss_fallback.nlargest(10, 'Financial Loss')
                if not top_countries.empty:
                    fig_fallback = px.bar(
                        top_countries, x='Country_Name', y='Financial Loss',
                        color='Financial Loss', color_continuous_scale='Blues',
                        template="plotly_white", custom_data=['Loss_Formatted']
                    )
                    fig_fallback.update_traces(hovertemplate='<b>%{x}</b><br>Loss: %{customdata[0]}<extra></extra>')
                    fig_fallback.update_layout(margin=dict(l=10,r=10,t=10,b=10), xaxis_title="Country", yaxis_title="Loss (USD)")
                    st.plotly_chart(fig_fallback, use_container_width=True)
    
    with geo2:
        st.subheader("🌐 Bubble Map: Country Exposure")
        country_metrics = df.groupby('Country').agg({
            'Financial Loss': 'sum',
            'Affected Users': 'sum'
        }).reset_index()
        
        if not country_metrics.empty and country_metrics['Financial Loss'].sum() > 0:
            country_metrics['Loss_Formatted'] = country_metrics['Financial Loss'].apply(format_currency)
            fig_bubble = px.scatter(
                country_metrics,
                x='Country',
                y='Financial Loss',
                size='Affected Users',
                color='Financial Loss',
                color_continuous_scale='Reds',
                template="plotly_white",
                custom_data=['Loss_Formatted', 'Affected Users']
            )
            fig_bubble.update_traces(hovertemplate='<b>%{x}</b><br>Loss: %{customdata[0]}<br>Users: %{customdata[1]:,.0f}<extra></extra>')
            fig_bubble.update_layout(margin=dict(l=10,r=10,t=10,b=10), yaxis_title="Loss (USD)")
            st.plotly_chart(fig_bubble, use_container_width=True)
            st.caption("🔵 Bubble size = affected users | Color intensity = financial loss")
        else:
            st.warning("No country data available for bubble map")
    
    st.markdown("---")
    
    st.subheader("📊 Regional Incident Frequency")
    region_incidents = df.groupby('Region').size().reset_index(name='Incidents').sort_values('Incidents', ascending=False)
    if not region_incidents.empty and region_incidents['Incidents'].sum() > 0:
        fig_region_bar = px.bar(
            region_incidents,
            x='Region',
            y='Incidents',
            color='Region',
            color_discrete_sequence=px.colors.qualitative.Bold,
            template="plotly_white"
        )
        fig_region_bar.update_layout(margin=dict(l=10,r=10,t=10,b=10))
        st.plotly_chart(fig_region_bar, use_container_width=True)
    else:
        st.warning("No regional incident data available")

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
        }).round(2).reset_index()
        defense_eff = defense_eff.dropna(subset=['Resolution Time']).sort_values('Resolution Time')
        
        if not defense_eff.empty and defense_eff['Resolution Time'].notna().any():
            defense_eff['Loss_Formatted'] = defense_eff['Financial Loss'].apply(format_currency)
            fig_def = px.bar(
                defense_eff,
                x='Defense Mechanism',
                y='Resolution Time',
                color='Financial Loss',
                color_continuous_scale='RdYlGn_r',
                template="plotly_white",
                custom_data=['Loss_Formatted']
            )
            fig_def.update_traces(hovertemplate='<b>%{x}</b><br>Resolution: %{y:.1f} hrs<br>Avg Loss: %{customdata[0]}<extra></extra>')
            fig_def.update_layout(margin=dict(l=10,r=10,t=10,b=10), yaxis_title="Avg Resolution (hrs)")
            st.plotly_chart(fig_def, use_container_width=True)
            
            st.caption(f"🟢 Best performer: {defense_eff.iloc[0]['Defense Mechanism']} ({defense_eff.iloc[0]['Resolution Time']:.1f} hrs avg)")
        else:
            st.warning("No defense mechanism data available")
    
    with col2:
        st.subheader("🔒 Security Vulnerability Distribution")
        st.markdown("_Pie chart: root cause analysis_")
        
        vuln_dist = df.groupby('Security Vulnerability').size().reset_index(name='Count')
        if not vuln_dist.empty and vuln_dist['Count'].sum() > 0:
            fig_pie = px.pie(
                vuln_dist,
                values='Count',
                names='Security Vulnerability',
                color_discrete_sequence=px.colors.qualitative.Pastel,
                template="plotly_white"
            )
            fig_pie.update_layout(margin=dict(l=10,r=10,t=10,b=10))
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.warning("No vulnerability data available")
    
    st.markdown("---")
    
    st.subheader("📊 Attack Source Analysis")
    if not df.empty:
        source_metrics = df.groupby('Attack Source').agg({
            'Financial Loss': ['sum', 'mean'],
            'Resolution Time': 'mean',
            'Country': 'count'
        }).round(2)
        source_metrics.columns = ['Total Loss', 'Avg Loss', 'Avg Resolution (hrs)', 'Incidents']
        source_metrics = source_metrics.sort_values('Total Loss', ascending=False)
        source_display = source_metrics.copy()
        source_display['Total Loss'] = source_metrics['Total Loss'].apply(format_currency)
        source_display['Avg Loss'] = source_metrics['Avg Loss'].apply(format_currency)
        
        st.dataframe(source_display, use_container_width=True, height=250)
        st.caption("💡 **Actionable Insight:** Attack sources with high loss and long resolution need priority security controls")

# =============================================================================
# TAB 6: MACHINE LEARNING EVALUATION & INSIGHTS
# =============================================================================
with tab6:
    st.header("Machine Learning: Predictive Risk Assessment")
    st.markdown("_Random Forest model evaluation for threat prediction_")
    
    ml_data = df[['Attack Type', 'Industry', 'Region', 'Attack Source', 
                  'Security Vulnerability', 'Defense Mechanism', 'Financial Loss']].copy()
    
    if len(ml_data) < 10:
        st.error("⚠️ Insufficient data for ML analysis. At least 10 records required.")
        st.stop()
    
    percentile_75 = ml_data['Financial Loss'].quantile(0.75)
    ml_data['High_Risk'] = (ml_data['Financial Loss'] > percentile_75).astype(int)
    
    cat_cols = ['Attack Type', 'Industry', 'Region', 'Attack Source', 
                'Security Vulnerability', 'Defense Mechanism']
    
    encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    ml_data_cat = ml_data[cat_cols].astype(str)
    X_encoded = encoder.fit_transform(ml_data_cat)
    X = pd.DataFrame(X_encoded, columns=[f"{col}_encoded" for col in cat_cols], index=ml_data.index)
    y = ml_data['High_Risk']
    
    if y.nunique() < 2:
        st.error("⚠️ Insufficient class diversity. All records have same risk level.")
        st.stop()
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
    rf_model.fit(X_train, y_train)
    
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
            margin=dict(l=10,r=10,t=10,b=10),
            template="plotly_white"
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
            text_auto=True,
            template="plotly_white"
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
        color_continuous_scale='Viridis',
        template="plotly_white"
    )
    fig_imp.update_layout(margin=dict(l=10,r=10,t=10,b=10))
    st.plotly_chart(fig_imp, use_container_width=True)
    st.caption(f"🎯 **Top predictor:** {feature_importance.iloc[0]['Feature']} (importance: {feature_importance.iloc[0]['Importance']:.3f})")
    
    st.markdown("---")
    
    st.subheader("📊 Learning Curve Analysis")
    st.markdown("_Model performance vs training data size_")
    
    try:
        rf_estimator = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
        cv_folds = min(5, max(2, len(X) // 10))
        
        train_sizes, train_scores, val_scores = learning_curve(
            rf_estimator, X, y, cv=cv_folds,
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
            margin=dict(l=10,r=10,t=10,b=10),
            template="plotly_white"
        )
        st.plotly_chart(fig_lc, use_container_width=True)
        st.caption("💡 **Insight:** Converging lines indicate good model generalization")
    except Exception as e:
        st.warning(f"⚠️ Learning curve generation failed: {str(e)}")

# =============================================================================
# FOOTER
# =============================================================================
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #64748b; font-size: 13px; line-height: 1.6;'>"
    "🔐 <strong>Global Cyber Risk Intelligence Dashboard</strong> | Data Period: 2015–2024<br>"
    "Made with <a href='https://github.com/mercydeez' target='_blank' style='color: #ef4444; text-decoration: none; font-size: 16px;'>❤️</a> by <strong>Atharva Soundankar</strong>"
    "</div>",
    unsafe_allow_html=True
)