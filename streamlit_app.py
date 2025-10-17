import streamlit as st
import pandas as pd
from datetime import datetime

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Travel Ticket Analysis",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- DATA LOADING ---
@st.cache_data
def load_data():
    """Loads the travel tickets dataset."""
    df = pd.read_csv('EDA_cleaned.csv')
    
    # Convert date columns to datetime
    df['Created'] = pd.to_datetime(df['Created'])
    df['DepartureTime'] = pd.to_datetime(df['DepartureTime'])
    
    # Calculate days until departure
    df['DaysUntilDeparture'] = (df['DepartureTime'] - df['Created']).dt.days
    
    # Create price categories
    df['PriceCategory'] = pd.cut(df['Price'], 
                               bins=[0, 1000000, 5000000, 10000000, float('inf')],
                               labels=['<1M', '1M-5M', '5M-10M', '>10M'])
    
    # Convert gender to string for better display
    df['Gender'] = df['Male'].map({1: 'Male', 0: 'Female'})
    
    return df

df = load_data()

# --- APP TITLE AND DESCRIPTION ---
st.title("✈️ Travel Tickets Data Analysis")
st.markdown("""
This application performs exploratory data analysis (EDA) on the Travel Tickets dataset.
Use the filters in the sidebar to explore different aspects of ticket sales and customer behavior.
""")

# --- SIDEBAR FOR FILTERS ---
st.sidebar.header("Filter Your Data")

# Filter for vehicle type
vehicle_types = st.sidebar.multiselect(
    "Select Vehicle Type",
    options=df["Vehicle"].unique(),
    default=df["Vehicle"].unique()
)

# Filter for trip reason
trip_reasons = st.sidebar.multiselect(
    "Select Trip Reason",
    options=df["TripReason"].unique(),
    default=df["TripReason"].unique()
)

# Filter for domestic/international
domestic_filter = st.sidebar.multiselect(
    "Select Trip Type",
    options=df["Domestic"].unique(),
    default=df["Domestic"].unique()
)

# Filter for cancellation status
cancel_filter = st.sidebar.multiselect(
    "Select Cancel Status",
    options=df["Cancel"].unique(),
    default=df["Cancel"].unique()
)

# Filter for price range
min_price, max_price = int(df["Price"].min()), int(df["Price"].max())
price_range = st.sidebar.slider(
    "Select Price Range",
    min_value=min_price,
    max_value=max_price,
    value=(min_price, max_price),
)

# Filter for gender
gender_filter = st.sidebar.multiselect(
    "Select Gender",
    options=df["Gender"].unique(),
    default=df["Gender"].unique()
)

# --- FILTERING THE DATAFRAME ---
df_selection = df.copy()

# Apply filters
if vehicle_types:
    df_selection = df_selection[df_selection["Vehicle"].isin(vehicle_types)]
if trip_reasons:
    df_selection = df_selection[df_selection["TripReason"].isin(trip_reasons)]
if domestic_filter:
    df_selection = df_selection[df_selection["Domestic"].isin(domestic_filter)]
if cancel_filter:
    df_selection = df_selection[df_selection["Cancel"].isin(cancel_filter)]
if gender_filter:
    df_selection = df_selection[df_selection["Gender"].isin(gender_filter)]

# Apply price filter
df_selection = df_selection[
    (df_selection["Price"] >= price_range[0]) &
    (df_selection["Price"] <= price_range[1])
]

# Display error message if no data is selected
if df_selection.empty:
    st.warning("No data available for the selected filters. Please adjust your selection.")
    st.stop()

# --- MAIN PAGE CONTENT ---
st.subheader("📊 Key Metrics")

# --- DISPLAY KEY METRICS ---
col1, col2, col3, col4 = st.columns(4)

with col1:
    total_tickets = df_selection.shape[0]
    st.metric(label="Total Tickets", value=f"{total_tickets:,}")

with col2:
    avg_price = round(df_selection["Price"].mean(), 2)
    st.metric(label="Average Price", value=f"${avg_price:,.2f}")

with col3:
    cancellation_rate = (df_selection["Cancel"].sum() / len(df_selection)) * 100
    st.metric(label="Cancellation Rate", value=f"{cancellation_rate:.1f}%")

with col4:
    avg_discount = round(df_selection["CouponDiscount"].mean(), 2)
    st.metric(label="Average Discount", value=f"${avg_discount:,.2f}")

st.markdown("---")

# --- VISUALIZATIONS ---
st.subheader("📈 Visualizations")

# Arrange charts in columns
viz_col1, viz_col2 = st.columns(2)

with viz_col1:
    # Ticket count by vehicle type
    st.subheader("Ticket Count by Vehicle Type")
    vehicle_counts = df_selection["Vehicle"].value_counts()
    st.bar_chart(vehicle_counts)

with viz_col2:
    # Average price by vehicle type
    st.subheader("Average Price by Vehicle Type")
    avg_price_vehicle = df_selection.groupby("Vehicle")["Price"].mean()
    st.bar_chart(avg_price_vehicle)

# Second row of visualizations
viz_col3, viz_col4 = st.columns(2)

with viz_col3:
    # Average price by trip reason
    st.subheader("Average Price by Trip Reason")
    avg_price_reason = df_selection.groupby("TripReason")["Price"].mean()
    st.bar_chart(avg_price_reason)

with viz_col4:
    # Cancellation rate by vehicle type
    st.subheader("Cancellation Rate by Vehicle Type")
    cancel_by_vehicle = df_selection.groupby("Vehicle")["Cancel"].mean() * 100
    st.bar_chart(cancel_by_vehicle)

# Third row
viz_col5, viz_col6 = st.columns(2)

with viz_col5:
    # Tickets by gender
    st.subheader("Tickets by Gender")
    gender_counts = df_selection["Gender"].value_counts()
    st.bar_chart(gender_counts)

with viz_col6:
    # Average days until departure by vehicle type
    st.subheader("Average Days Until Departure")
    avg_days = df_selection.groupby("Vehicle")["DaysUntilDeparture"].mean()
    st.bar_chart(avg_days)

# Price distribution
st.subheader("Price Distribution")
price_bins = st.slider("Number of bins", 10, 100, 50)
st.bar_chart(df_selection["Price"].value_counts(bins=price_bins).sort_index())

# --- DATA TABLES ---
st.subheader("📋 Data Summary Tables")

# Summary table by vehicle type
st.write("**Summary by Vehicle Type:**")
summary_table = df_selection.groupby("Vehicle").agg({
    'Price': ['count', 'mean', 'median'],
    'Cancel': 'mean',
    'DaysUntilDeparture': 'mean'
}).round(2)
st.dataframe(summary_table)

# Top routes
st.write("**Top 10 Routes:**")
route_counts = df_selection.groupby(['From', 'To']).size().sort_values(ascending=False).head(10)
st.dataframe(route_counts.reset_index(name='Ticket Count'))

# --- DISPLAY RAW DATA ---
with st.expander("View Raw Data"):
    st.dataframe(df_selection)
    st.markdown(f"**Data Dimensions:** {df_selection.shape[0]} rows, {df_selection.shape[1]} columns")

st.markdown("---")
st.write("Data Source: Travel Tickets Dataset")
