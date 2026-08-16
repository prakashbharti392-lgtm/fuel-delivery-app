"""
dashboard.py
Fuel Delivery App - Analytics Dashboard (matplotlib/seaborn version - no plotly needed)

Run karne ke liye:
    streamlit run dashboard.py
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ---- Page Setup ----
st.set_page_config(
    page_title="Fuel Delivery Analytics",
    page_icon="⛽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---- Custom CSS (professional look, matches app ka pink/orange theme) ----
st.markdown("""
<style>
    .main { background-color: #f4f6f8; }

    .dashboard-header {
        background: linear-gradient(135deg, #ff6a00, #ee0979);
        padding: 28px 32px;
        border-radius: 14px;
        margin-bottom: 24px;
    }
    .dashboard-header h1 {
        color: white; margin: 0; font-size: 30px; font-weight: 700;
    }
    .dashboard-header p {
        color: rgba(255,255,255,0.9); margin: 6px 0 0; font-size: 15px;
    }

    div[data-testid="stMetric"] {
        background: white;
        padding: 18px 20px;
        border-radius: 12px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.06);
        border-left: 4px solid #ee0979;
    }
    div[data-testid="stMetricLabel"] { font-weight: 600; color: #666; }
    div[data-testid="stMetricValue"] { color: #1a1a1a; font-weight: 700; }

    h2, h3 { color: #1a1a1a; font-weight: 700; }

    section[data-testid="stSidebar"] { background-color: #1a1a2e; }
    section[data-testid="stSidebar"] * { color: white !important; }
</style>
""", unsafe_allow_html=True)

sns.set_style("whitegrid")
PALETTE = ["#ee0979", "#ff6a00", "#f77f00", "#c9184a", "#f4a261"]

# ---- Data Load karo ----
df = pd.read_csv("data/fuel_delivery_data.csv")

# ==============================
# SIDEBAR - Filters
# ==============================
st.sidebar.title("⛽ FuelMitra")
st.sidebar.markdown("### Filters")

time_filter = st.sidebar.multiselect(
    "Time of Day", options=df["time_of_day"].unique(), default=list(df["time_of_day"].unique())
)
traffic_filter = st.sidebar.multiselect(
    "Traffic Level", options=df["traffic_level"].unique(), default=list(df["traffic_level"].unique())
)
weather_filter = st.sidebar.multiselect(
    "Weather", options=df["weather"].unique(), default=list(df["weather"].unique())
)

st.sidebar.markdown("---")
st.sidebar.caption("Highway Fuel Delivery — Analytics Dashboard")
st.sidebar.caption("Final Year Project · Data Science")

df_filtered = df[
    df["time_of_day"].isin(time_filter) &
    df["traffic_level"].isin(traffic_filter) &
    df["weather"].isin(weather_filter)
]

# ==============================
# HEADER
# ==============================
st.markdown("""
<div class="dashboard-header">
    <h1>⛽ Fuel Delivery — Analytics Dashboard</h1>
    <p>Highway fuel delivery requests ka business insights aur performance overview</p>
</div>
""", unsafe_allow_html=True)

if df_filtered.empty:
    st.warning("Koi data nahi mila in filters ke sath. Sidebar se filters change karo.")
    st.stop()

# ==============================
# SECTION 1: KPI Cards
# ==============================
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Requests", f"{len(df_filtered):,}")
with col2:
    st.metric("Avg ETA", f"{df_filtered['eta_minutes'].mean():.1f} min")
with col3:
    st.metric("Avg Distance", f"{df_filtered['distance_km'].mean():.1f} km")
with col4:
    st.metric("Avg Fuel / Order", f"{df_filtered['fuel_quantity'].mean():.1f} L")

st.write("")

# ==============================
# SECTION 2: Time of Day Analysis
# ==============================
st.subheader("🕐 Peak Hours Analysis")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Requests kis time zyada aate hain**")
    time_counts = df_filtered["time_of_day"].value_counts()
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(x=time_counts.index, y=time_counts.values, ax=ax, palette=PALETTE)
    ax.set_xlabel("")
    ax.set_ylabel("Requests")
    st.pyplot(fig)

with col2:
    st.markdown("**Time of Day ke hisaab se Average ETA**")
    avg_eta_time = df_filtered.groupby("time_of_day")["eta_minutes"].mean()
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(x=avg_eta_time.index, y=avg_eta_time.values, ax=ax, palette=PALETTE)
    ax.set_xlabel("")
    ax.set_ylabel("Avg ETA (min)")
    st.pyplot(fig)

# ==============================
# SECTION 3: Traffic & Weather Impact
# ==============================
st.subheader("🚦 Traffic & Weather Impact on ETA")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Traffic Level vs Avg ETA**")
    avg_eta_traffic = (
        df_filtered.groupby("traffic_level")["eta_minutes"]
        .mean().reindex(["Low", "Medium", "High"]).dropna()
    )
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(x=avg_eta_traffic.index, y=avg_eta_traffic.values, ax=ax, palette=PALETTE)
    ax.set_xlabel("")
    ax.set_ylabel("Avg ETA (min)")
    st.pyplot(fig)

with col2:
    st.markdown("**Weather ke hisaab se Average ETA**")
    avg_eta_weather = df_filtered.groupby("weather")["eta_minutes"].mean()
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(x=avg_eta_weather.index, y=avg_eta_weather.values, ax=ax, palette=PALETTE)
    ax.set_xlabel("")
    ax.set_ylabel("Avg ETA (min)")
    st.pyplot(fig)

# ==============================
# SECTION 4: Distance vs ETA
# ==============================
st.subheader("📏 Distance vs ETA Relationship")

fig, ax = plt.subplots(figsize=(10, 5))
sns.scatterplot(data=df_filtered, x="distance_km", y="eta_minutes", hue="traffic_level",
                 palette=PALETTE, alpha=0.7, ax=ax)
ax.set_xlabel("Distance (km)")
ax.set_ylabel("ETA (minutes)")
st.pyplot(fig)

# ==============================
# SECTION 5: Fuel Quantity Distribution
# ==============================
st.subheader("🛢️ Fuel Quantity Distribution")

col1, col2 = st.columns([2, 1])

with col1:
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(df_filtered["fuel_quantity"], bins=20, kde=True, ax=ax, color="#ff6a00")
    ax.set_xlabel("Fuel Quantity (litres)")
    st.pyplot(fig)

with col2:
    st.markdown("#### Quick Stats")
    st.write(f"**Minimum:** {df_filtered['fuel_quantity'].min()} L")
    st.write(f"**Maximum:** {df_filtered['fuel_quantity'].max()} L")
    st.write(f"**Median:** {df_filtered['fuel_quantity'].median()} L")
    st.write(f"**Avg ETA:** {df_filtered['eta_minutes'].mean():.1f} min")
    st.write(f"**Avg Distance:** {df_filtered['distance_km'].mean():.1f} km")

# ==============================
# SECTION 6: Raw Data
# ==============================
with st.expander("📋 Raw Data dekhna hai to yahan click karo"):
    st.dataframe(df_filtered, use_container_width=True)