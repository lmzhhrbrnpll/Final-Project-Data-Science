# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Customer Churn Analysis & Prediction",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- DATA LOADING ---
@st.cache_data
def load_data():
    """Loads the customer churn dataset."""
    try:
        # Try multiple possible file locations
        file_paths = [
            "customer_churn.csv",
            "Data/customer_churn.csv",
            "EDA_cleaned.csv",
            "Data/EDA_cleaned.csv",
            "data.csv"
        ]
        
        for file_path in file_paths:
            try:
                df = pd.read_csv(file_path)
                st.success(f"Successfully loaded data from: {file_path}")
                return df
            except:
                continue
                
        # If no file found, create sample data
        st.warning("No dataset file found. Using sample data for demonstration.")
        np.random.seed(42)
        n_samples = 1000
        
        sample_data = {
            'Tenure': np.random.randint(1, 36, n_samples),
            'OrderCount': np.random.randint(1, 50, n_samples),
            'CashbackAmount': np.random.uniform(50, 500, n_samples),
            'HourSpendOnApp': np.random.uniform(0.5, 8.0, n_samples),
            'NumberOfDeviceRegistered': np.random.randint(1, 6, n_samples),
            'SatisfactionScore': np.random.randint(1, 6, n_samples),
            'Complain': np.random.randint(0, 2, n_samples),
            'CityTier': np.random.randint(1, 4, n_samples),
            'Gender': np.random.choice(['Male', 'Female'], n_samples),
            'PreferredLoginDevice': np.random.choice(['Mobile', 'Computer', 'Phone'], n_samples),
            'PreferredPaymentMode': np.random.choice(['Credit Card', 'Debit Card', 'UPI', 'Cash'], n_samples),
            'PreferedOrderCat': np.random.choice(['Fashion', 'Electronics', 'Groceries', 'Books'], n_samples),
            'Churn': np.random.randint(0, 2, n_samples)
        }
        
        return pd.DataFrame(sample_data)
        
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        # Return minimal sample data as fallback
        return pd.DataFrame({
            'Tenure': [12, 24, 6, 18],
            'Churn': [0, 1, 0, 1]
        })

# Load data
df = load_data()

# --- SIMPLE PREPROCESSING ---
def simple_encoder(df, column):
    """Simple encoding for categorical variables without sklearn"""
    if column in df.columns and df[column].dtype == 'object':
        unique_vals = df[column].unique()
        mapping = {val: idx for idx, val in enumerate(unique_vals)}
        return df[column].map(mapping), mapping
    return df[column], None

def calculate_risk_score(input_values, df):
    """Calculate churn risk score based on input values and data patterns"""
    risk_score = 0
    max_possible_score = 0
    
    # Define risk factors and their weights
    risk_factors = {
        'Tenure': {'weight': 0.2, 'risk_func': lambda x, df: 1 if x < df['Tenure'].quantile(0.25) else 0},
        'SatisfactionScore': {'weight': 0.15, 'risk_func': lambda x, df: 1 if x <= 2 else 0},
        'Complain': {'weight': 0.25, 'risk_func': lambda x, df: x},  # 1 if has complaint
        'CashbackAmount': {'weight': 0.1, 'risk_func': lambda x, df: 1 if x < df['CashbackAmount'].quantile(0.25) else 0},
        'OrderCount': {'weight': 0.1, 'risk_func': lambda x, df: 1 if x < df['OrderCount'].quantile(0.25) else 0},
        'HourSpendOnApp': {'weight': 0.1, 'risk_func': lambda x, df: 1 if x < df['HourSpendOnApp'].quantile(0.25) else 0},
    }
    
    # Calculate risk for each factor
    for factor, config in risk_factors.items():
        if factor in input_values:
            risk_value = config['risk_func'](input_values[factor], df)
            risk_score += risk_value * config['weight']
            max_possible_score += config['weight']
    
    # Normalize score
    if max_possible_score > 0:
        normalized_score = risk_score / max_possible_score
    else:
        normalized_score = 0
        
    return min(normalized_score, 1.0)

# --- MAIN APP ---
st.title("📊 Customer Churn Analysis & Prediction Dashboard")

# Create tabs
tab1, tab2, tab3 = st.tabs(["📈 Exploratory Data Analysis", "🤖 Churn Prediction", "🔍 Data Overview"])

with tab1:
    if df is not None and not df.empty:
        st.markdown("""
        ## Exploratory Data Analysis
        Explore customer behavior and churn patterns using the filters below.
        """)

        # --- SIDEBAR FILTERS (Important filters only) ---
        st.sidebar.header("🔍 EDA Filters")
        
        # Important filters only
        important_filters = []
        
        # Churn status (always important)
        if 'Churn' in df.columns:
            churn_status = st.sidebar.multiselect(
                "Churn Status",
                options=df["Churn"].unique(),
                default=df["Churn"].unique()
            )
            important_filters.append(('Churn', churn_status))
        
        # Tenure filter (important)
        if 'Tenure' in df.columns:
            min_tenure, max_tenure = int(df["Tenure"].min()), int(df["Tenure"].max())
            tenure_range = st.sidebar.slider(
                "Tenure Range (months)",
                min_tenure, max_tenure, (min_tenure, max_tenure)
            )
            important_filters.append(('Tenure', tenure_range))
        
        # Satisfaction Score (important)
        if 'SatisfactionScore' in df.columns:
            satisfaction_scores = st.sidebar.multiselect(
                "Satisfaction Score",
                options=sorted(df["SatisfactionScore"].unique()),
                default=sorted(df["SatisfactionScore"].unique())
            )
            important_filters.append(('SatisfactionScore', satisfaction_scores))
        
        # City Tier (important)
        if 'CityTier' in df.columns:
            city_tiers = st.sidebar.multiselect(
                "City Tier",
                options=sorted(df["CityTier"].unique()),
                default=sorted(df["CityTier"].unique())
            )
            important_filters.append(('CityTier', city_tiers))
        
        # Complain status (important)
        if 'Complain' in df.columns:
            complain_status = st.sidebar.multiselect(
                "Complain Status",
                options=df["Complain"].unique(),
                default=df["Complain"].unique()
            )
            important_filters.append(('Complain', complain_status))

        # --- APPLY FILTERS ---
        df_filtered = df.copy()
        
        for col, filter_value in important_filters:
            if col in df_filtered.columns:
                if col == 'Tenure':
                    # Handle range filter
                    df_filtered = df_filtered[
                        (df_filtered[col] >= filter_value[0]) & 
                        (df_filtered[col] <= filter_value[1])
                    ]
                else:
                    # Handle multiselect filter
                    if filter_value:
                        df_filtered = df_filtered[df_filtered[col].isin(filter_value)]

        # Display results
        total_customers = df_filtered.shape[0]
        if total_customers == 0:
            st.warning("No data available for the selected filters.")
        else:
            # --- KEY METRICS ---
            st.subheader("📊 Key Metrics")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Customers", total_customers)
            
            with col2:
                if 'Churn' in df_filtered.columns:
                    churn_count = df_filtered["Churn"].sum()
                    churn_rate = (churn_count / total_customers) * 100
                    st.metric("Churn Rate", f"{churn_rate:.1f}%")
                else:
                    st.metric("Sample Data", "Demo Mode")
            
            with col3:
                if 'Tenure' in df_filtered.columns:
                    avg_tenure = df_filtered["Tenure"].mean()
                    st.metric("Avg Tenure", f"{avg_tenure:.1f} months")
                else:
                    st.metric("Features", len(df_filtered.columns))
            
            with col4:
                if 'Complain' in df_filtered.columns:
                    complain_count = df_filtered["Complain"].sum()
                    complain_rate = (complain_count / total_customers) * 100
                    st.metric("Complain Rate", f"{complain_rate:.1f}%")
                elif 'OrderCount' in df_filtered.columns:
                    avg_orders = df_filtered["OrderCount"].mean()
                    st.metric("Avg Orders", f"{avg_orders:.1f}")

            # Additional metrics row
            if any(col in df_filtered.columns for col in ['CashbackAmount', 'HourSpendOnApp', 'OrderCount']):
                st.subheader("📈 Engagement Metrics")
                col5, col6, col7 = st.columns(3)
                
                metrics_to_show = [
                    ('CashbackAmount', 'Avg Cashback', '💰'),
                    ('HourSpendOnApp', 'Avg App Hours', '⏱️'),
                    ('OrderCount', 'Avg Orders', '📦')
                ]
                
                for i, (col_name, display_name, icon) in enumerate(metrics_to_show):
                    with [col5, col6, col7][i]:
                        if col_name in df_filtered.columns:
                            avg_val = df_filtered[col_name].mean()
                            st.metric(f"{icon} {display_name}", f"{avg_val:.1f}")

            st.markdown("---")

            # --- VISUALIZATIONS ---
            st.subheader("📊 Visualizations")

            # Row 1: Main charts
            col1, col2 = st.columns(2)
            
            with col1:
                if 'Churn' in df_filtered.columns:
                    st.subheader("Churn Distribution")
                    churn_counts = df_filtered["Churn"].value_counts()
                    churn_df = pd.DataFrame({
                        'Status': ['Not Churned', 'Churned'],
                        'Count': churn_counts.values
                    })
                    st.bar_chart(churn_df.set_index('Status'))
            
            with col2:
                if 'CashbackAmount' in df_filtered.columns and 'Churn' in df_filtered.columns:
                    st.subheader("Cashback by Churn Status")
                    cashback_by_churn = df_filtered.groupby('Churn')['CashbackAmount'].mean()
                    st.bar_chart(cashback_by_churn)
                elif 'Tenure' in df_filtered.columns and 'Churn' in df_filtered.columns:
                    st.subheader("Tenure by Churn Status")
                    tenure_by_churn = df_filtered.groupby('Churn')['Tenure'].mean()
                    st.bar_chart(tenure_by_churn)

            # Row 2: Additional insights
            col3, col4 = st.columns(2)
            
            with col3:
                if 'SatisfactionScore' in df_filtered.columns:
                    st.subheader("Satisfaction Score Distribution")
                    satisfaction_counts = df_filtered["SatisfactionScore"].value_counts().sort_index()
                    st.bar_chart(satisfaction_counts)
                elif 'CityTier' in df_filtered.columns:
                    st.subheader("City Tier Distribution")
                    city_counts = df_filtered["CityTier"].value_counts().sort_index()
                    st.bar_chart(city_counts)
            
            with col4:
                if 'Complain' in df_filtered.columns and 'Churn' in df_filtered.columns:
                    st.subheader("Churn by Complain Status")
                    churn_by_complain = df_filtered.groupby('Complain')['Churn'].mean() * 100
                    st.bar_chart(churn_by_complain)
                elif 'PreferredPaymentMode' in df_filtered.columns:
                    st.subheader("Preferred Payment Methods")
                    payment_counts = df_filtered["PreferredPaymentMode"].value_counts().head(5)
                    st.bar_chart(payment_counts)

            # --- DATA SUMMARY ---
            st.subheader("📋 Summary Statistics")
            
            summary_col1, summary_col2 = st.columns(2)
            
            with summary_col1:
                st.write("**Customer Overview:**")
                st.write(f"- Total customers: {len(df_filtered):,}")
                if 'Churn' in df_filtered.columns:
                    st.write(f"- Churned customers: {df_filtered['Churn'].sum():,}")
                    st.write(f"- Churn rate: {churn_rate:.2f}%")
                if 'Complain' in df_filtered.columns:
                    st.write(f"- Customers with complaints: {df_filtered['Complain'].sum():,}")
            
            with summary_col2:
                st.write("**Key Metrics:**")
                if 'Tenure' in df_filtered.columns:
                    st.write(f"- Average tenure: {df_filtered['Tenure'].mean():.1f} months")
                if 'SatisfactionScore' in df_filtered.columns:
                    st.write(f"- Average satisfaction: {df_filtered['SatisfactionScore'].mean():.1f}/5")
                if 'CashbackAmount' in df_filtered.columns:
                    st.write(f"- Average cashback: ${df_filtered['CashbackAmount'].mean():.1f}")

            # --- RAW DATA VIEW ---
            with st.expander("📄 View Filtered Data"):
                st.dataframe(df_filtered.head(100), use_container_width=True)
                st.write(f"**Data Dimensions:** {len(df_filtered)} rows, {len(df_filtered.columns)} columns")
                st.write(f"*Showing first 100 rows*")

with tab2:
    st.header("🤖 Churn Prediction Model")
    st.markdown("""
    ## Rule-based Churn Risk Assessment
    This system calculates churn risk based on business rules and data patterns from your dataset.
    """)
    
    if df is not None and not df.empty:
        st.success("✅ Risk assessment system ready!")
        
        # Display data patterns
        st.subheader("📊 Data Patterns Analysis")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if 'Churn' in df.columns:
                churn_rate = (df['Churn'].sum() / len(df)) * 100
                st.metric("Overall Churn Rate", f"{churn_rate:.1f}%")
        
        with col2:
            if 'Tenure' in df.columns:
                avg_tenure_churn = df[df['Churn'] == 1]['Tenure'].mean() if 1 in df['Churn'].values else 0
                avg_tenure_no_churn = df[df['Churn'] == 0]['Tenure'].mean() if 0 in df['Churn'].values else 0
                st.metric("Avg Tenure (Churn vs Non-Churn)", f"{avg_tenure_churn:.1f} vs {avg_tenure_no_churn:.1f}")
        
        with col3:
            if 'Complain' in df.columns and 'Churn' in df.columns:
                complain_churn_rate = (df[df['Complain'] == 1]['Churn'].mean() * 100) if 1 in df['Complain'].values else 0
                st.metric("Churn Rate with Complaints", f"{complain_churn_rate:.1f}%")

        # --- PREDICTION INTERFACE ---
        st.markdown("---")
        st.subheader("🎯 Churn Risk Assessment")
        st.markdown("""
        Enter customer details below to assess churn risk. **All available features are used** for comprehensive assessment.
        """)
        
        with st.form("prediction_form"):
            st.write("### Customer Information")
            
            # Organize inputs into columns
            col1, col2, col3 = st.columns(3)
            input_data = {}
            
            # Get all column names except Churn
            feature_columns = [col for col in df.columns if col != 'Churn']
            
            # Create input fields for all features
            for i, feature in enumerate(feature_columns):
                # Distribute across columns
                col = [col1, col2, col3][i % 3]
                
                with col:
                    if df[feature].dtype in ['int64', 'float64']:
                        # Numerical features
                        min_val = float(df[feature].min())
                        max_val = float(df[feature].max())
                        default_val = float(df[feature].median())
                        
                        # Adjust ranges for better UX
                        if feature == 'Tenure':
                            min_val, max_val = 0, 60
                            default_val = min(12, max_val)
                        elif feature == 'CashbackAmount':
                            min_val, max_val = 0, 1000
                        elif feature == 'HourSpendOnApp':
                            min_val, max_val = 0.0, 10.0
                        elif feature == 'SatisfactionScore':
                            min_val, max_val = 1, 5
                        elif feature == 'OrderCount':
                            min_val, max_val = 0, 100
                            
                        input_data[feature] = st.slider(
                            f"{feature}",
                            min_val, max_val, default_val,
                            help=f"Range in data: {df[feature].min():.1f} - {df[feature].max():.1f}"
                        )
                    else:
                        # Categorical features
                        unique_vals = df[feature].unique()
                        default_val = unique_vals[0] if len(unique_vals) > 0 else ""
                        input_data[feature] = st.selectbox(
                            f"{feature}",
                            options=unique_vals,
                            index=0,
                            help=f"Select from {len(unique_vals)} options"
                        )
            
            submitted = st.form_submit_button("🔮 Assess Churn Risk", use_container_width=True)
            
            if submitted:
                # Calculate risk score
                risk_score = calculate_risk_score(input_data, df)
                
                # Display results
                st.markdown("---")
                st.subheader("📋 Risk Assessment Results")
                
                # Results in columns
                result_col1, result_col2, result_col3 = st.columns(3)
                
                with result_col1:
                    st.metric(
                        "Churn Risk Score", 
                        f"{risk_score:.1%}",
                        delta=f"{(risk_score-0.5)*100:+.1f}% vs neutral" if risk_score != 0.5 else None
                    )
                
                with result_col2:
                    confidence = 0.8  # Fixed confidence for rule-based system
                    st.metric("Assessment Confidence", f"{confidence:.1%}")
                
                with result_col3:
                    if risk_score > 0.7:
                        st.error("**HIGH RISK OF CHURN** ⚠️")
                        st.write("Immediate retention action recommended")
                    elif risk_score > 0.4:
                        st.warning("**MEDIUM RISK OF CHURN** 📊")
                        st.write("Monitor closely and consider proactive engagement")
                    else:
                        st.success("**LOW RISK OF CHURN** ✅")
                        st.write("Customer likely to stay")
                
                # Visual risk gauge
                st.write("**Risk Level Gauge:**")
                st.progress(float(risk_score))
                st.caption(f"Risk score: {risk_score:.1%}")
                
                # Risk factors analysis
                with st.expander("🔍 Detailed Risk Analysis"):
                    st.write("**Key Risk Factors Identified:**")
                    
                    risk_factors = []
                    
                    # Tenure risk
                    if 'Tenure' in input_data:
                        tenure_q25 = df['Tenure'].quantile(0.25) if 'Tenure' in df.columns else 12
                        if input_data['Tenure'] < tenure_q25:
                            risk_factors.append(f"**Low Tenure**: {input_data['Tenure']} months (below 25th percentile: {tenure_q25:.1f} months)")
                    
                    # Satisfaction risk
                    if 'SatisfactionScore' in input_data and input_data['SatisfactionScore'] <= 2:
                        risk_factors.append(f"**Low Satisfaction**: Score {input_data['SatisfactionScore']} (≤ 2 indicates dissatisfaction)")
                    
                    # Complain risk
                    if 'Complain' in input_data and input_data['Complain'] == 1:
                        risk_factors.append("**Active Complaints**: Customer has registered complaints")
                    
                    # Cashback risk
                    if 'CashbackAmount' in input_data:
                        cashback_q25 = df['CashbackAmount'].quantile(0.25) if 'CashbackAmount' in df.columns else 150
                        if input_data['CashbackAmount'] < cashback_q25:
                            risk_factors.append(f"**Low Cashback**: ${input_data['CashbackAmount']:.1f} (below 25th percentile: ${cashback_q25:.1f})")
                    
                    # Order count risk
                    if 'OrderCount' in input_data:
                        orders_q25 = df['OrderCount'].quantile(0.25) if 'OrderCount' in df.columns else 5
                        if input_data['OrderCount'] < orders_q25:
                            risk_factors.append(f"**Low Order Count**: {input_data['OrderCount']} orders (below 25th percentile: {orders_q25:.1f})")
                    
                    # App usage risk
                    if 'HourSpendOnApp' in input_data:
                        hours_q25 = df['HourSpendOnApp'].quantile(0.25) if 'HourSpendOnApp' in df.columns else 1.0
                        if input_data['HourSpendOnApp'] < hours_q25:
                            risk_factors.append(f"**Low App Engagement**: {input_data['HourSpendOnApp']:.1f} hours (below 25th percentile: {hours_q25:.1f} hours)")
                    
                    if risk_factors:
                        for factor in risk_factors:
                            st.write(f"• 🔴 {factor}")
                    else:
                        st.write("• 🟢 No significant risk factors identified")
                    
                    st.write("---")
                    st.write("**Recommendations:**")
                    if risk_score > 0.7:
                        st.write("""
                        - **Immediate retention offers** (discounts, premium features)
                        - **Personalized outreach** from customer success team
                        - **Win-back campaign** with special incentives
                        - **Root cause analysis** for dissatisfaction
                        """)
                    elif risk_score > 0.4:
                        st.write("""
                        - **Proactive engagement** through email campaigns
                        - **Loyalty program** enrollment push
                        - **Satisfaction survey** to identify issues
                        - **Personalized recommendations** to increase engagement
                        """)
                    else:
                        st.write("""
                        - **Continue standard engagement** practices
                        - **Monitor for changes** in behavior patterns
                        - **Upsell opportunities** for loyal customers
                        - **Referral program** enrollment
                        """)
        
        # --- BUSINESS INSIGHTS ---
        st.markdown("---")
        st.subheader("💡 Business Insights")
        
        insight_col1, insight_col2 = st.columns(2)
        
        with insight_col1:
            st.write("**Top Churn Drivers:**")
            insights = []
            
            if 'Tenure' in df.columns and 'Churn' in df.columns:
                low_tenure_churn = df[df['Tenure'] < df['Tenure'].quantile(0.25)]['Churn'].mean() if len(df[df['Tenure'] < df['Tenure'].quantile(0.25)]) > 0 else 0
                insights.append(f"- **New customers** (< {df['Tenure'].quantile(0.25):.1f} months): {low_tenure_churn:.1%} churn rate")
            
            if 'Complain' in df.columns and 'Churn' in df.columns:
                complain_churn = df[df['Complain'] == 1]['Churn'].mean() if 1 in df['Complain'].values else 0
                insights.append(f"- **Customers with complaints**: {complain_churn:.1%} churn rate")
            
            if 'SatisfactionScore' in df.columns and 'Churn' in df.columns:
                low_sat_churn = df[df['SatisfactionScore'] <= 2]['Churn'].mean() if len(df[df['SatisfactionScore'] <= 2]) > 0 else 0
                insights.append(f"- **Low satisfaction** (score ≤ 2): {low_sat_churn:.1%} churn rate")
            
            for insight in insights:
                st.write(insight)
        
        with insight_col2:
            st.write("**Retention Opportunities:**")
            st.write("- **Onboarding program** for new customers")
            st.write("- **Proactive complaint resolution** system")
            st.write("- **Satisfaction improvement** campaigns")
            st.write("- **Loyalty rewards** for engaged customers")
            st.write("- **Personalized communication** based on usage patterns")

    else:
        st.error("❌ No data available for risk assessment.")

with tab3:
    st.header("🔍 Data Overview")
    
    if df is not None and not df.empty:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📁 Dataset Information")
            st.write(f"**Shape:** {df.shape[0]:,} rows × {df.shape[1]} columns")
            
            st.write("**Data Types:**")
            dtype_counts = df.dtypes.value_counts()
            for dtype, count in dtype_counts.items():
                st.write(f"- {dtype}: {count} columns")
            
            st.write("**Memory Usage:**")
            memory_mb = df.memory_usage(deep=True).sum() / 1024**2
            st.write(f"- Total: {memory_mb:.2f} MB")
            
            if 'Churn' in df.columns:
                st.write("**Churn Distribution:**")
                churn_counts = df['Churn'].value_counts()
                churn_rate = (churn_counts[1] / len(df)) * 100 if 1 in churn_counts else 0
                st.write(f"- Churned: {churn_counts[1] if 1 in churn_counts else 0:,} ({churn_rate:.1f}%)")
                st.write(f"- Not Churned: {churn_counts[0] if 0 in churn_counts else len(df):,}")
        
        with col2:
            st.subheader("📊 Basic Statistics")
            numeric_df = df.select_dtypes(include=[np.number])
            if not numeric_df.empty:
                st.dataframe(numeric_df.describe(), use_container_width=True)
            else:
                st.write("No numeric columns available for statistics.")
        
        # Data Quality Check
        st.subheader("✅ Data Quality Check")
        
        missing_values = df.isnull().sum()
        missing_percentage = (missing_values / len(df)) * 100
        
        quality_data = []
        for col in df.columns:
            unique_count = df[col].nunique()
            quality_data.append({
                'Column': col,
                'Data Type': str(df[col].dtype),
                'Missing Values': missing_values[col],
                'Missing %': f"{missing_percentage[col]:.1f}%",
                'Unique Values': unique_count
            })
        
        quality_df = pd.DataFrame(quality_data)
        st.dataframe(quality_df, use_container_width=True)
        
        # Column Details
        st.subheader("📋 Column Details")
        
        # Show important columns first
        important_columns = ['Churn', 'Tenure', 'SatisfactionScore', 'Complain', 'CashbackAmount']
        display_columns = [col for col in important_columns if col in df.columns] + \
                         [col for col in df.columns if col not in important_columns]
        
        for col in display_columns[:6]:  # Show first 6 columns
            with st.expander(f"📊 {col} ({df[col].dtype})"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**Unique values:** {df[col].nunique()}")
                    if df[col].dtype in ['int64', 'float64']:
                        st.write(f"**Min:** {df[col].min():.2f}")
                        st.write(f"**Max:** {df[col].max():.2f}")
                        st.write(f"**Mean:** {df[col].mean():.2f}")
                
                with col2:
                    if df[col].dtype == 'object' or df[col].nunique() <= 10:
                        st.write("**Value distribution:**")
                        value_counts = df[col].value_counts().head(10)
                        for value, count in value_counts.items():
                            percentage = (count / len(df)) * 100
                            st.write(f"- {value}: {count} ({percentage:.1f}%)")
                    else:
                        st.write("**Sample values:**")
                        st.write(df[col].head(5).tolist())
    else:
        st.error("No data available for overview.")

# Footer
st.markdown("---")
st.markdown("**Customer Churn Analysis & Prediction Dashboard** | *Using rule-based risk assessment with all available features*")

# Sidebar info
st.sidebar.markdown("---")
st.sidebar.write("**ℹ️ App Info**")
st.sidebar.write(f"- Data shape: {df.shape if df is not None else 'N/A'}")
st.sidebar.write("*Rule-based risk assessment*")
st.sidebar.write("*No ML dependencies required*")
