# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix
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
            'CustomerID': range(1, n_samples + 1),
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

# --- DATA FOR EDA (with important filters only) ---
df_eda = df.copy()

# --- DATA FOR PREDICTION (all features) ---
@st.cache_data
def prepare_prediction_data(df):
    """Prepare data for prediction model with all features."""
    df_pred = df.copy()
    
    # Remove CustomerID if exists
    if 'CustomerID' in df_pred.columns:
        df_pred = df_pred.drop('CustomerID', axis=1)
    
    # Handle missing values
    numeric_columns = df_pred.select_dtypes(include=[np.number]).columns
    categorical_columns = df_pred.select_dtypes(include=['object']).columns
    
    # Fill numerical missing values with median
    for col in numeric_columns:
        if df_pred[col].isnull().sum() > 0:
            df_pred[col].fillna(df_pred[col].median(), inplace=True)
    
    # Fill categorical missing values with mode
    for col in categorical_columns:
        if df_pred[col].isnull().sum() > 0:
            df_pred[col].fillna(df_pred[col].mode()[0], inplace=True)
    
    return df_pred

df_pred = prepare_prediction_data(df)

# --- MODEL TRAINING FUNCTION ---
@st.cache_resource
def train_churn_model(df_pred):
    """Train Random Forest model for churn prediction."""
    if df_pred is None or 'Churn' not in df_pred.columns:
        return None
    
    try:
        # Encode categorical variables
        df_ml = df_pred.copy()
        categorical_cols = df_ml.select_dtypes(include=['object']).columns
        label_encoders = {}
        
        for col in categorical_cols:
            le = LabelEncoder()
            df_ml[col] = le.fit_transform(df_ml[col].astype(str))
            label_encoders[col] = le
        
        # Prepare features and target
        X = df_ml.drop('Churn', axis=1)
        y = df_ml['Churn']
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Hyperparameter tuning
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [10, 15, None],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        }
        
        rf = RandomForestClassifier(random_state=42, class_weight='balanced')
        grid_search = GridSearchCV(
            rf, param_grid, cv=3, scoring='roc_auc', n_jobs=-1, verbose=0
        )
        
        grid_search.fit(X_train_scaled, y_train)
        
        best_model = grid_search.best_estimator_
        
        # Make predictions
        y_pred = best_model.predict(X_test_scaled)
        y_pred_proba = best_model.predict_proba(X_test_scaled)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc_score = roc_auc_score(y_test, y_pred_proba)
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc_score': auc_score
        }
        
        return {
            'model': best_model,
            'scaler': scaler,
            'label_encoders': label_encoders,
            'feature_names': X_train.columns.tolist(),
            'metrics': metrics,
            'best_params': grid_search.best_params_,
            'X_test': X_test,
            'y_test': y_test,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }
    
    except Exception as e:
        st.error(f"Error training model: {str(e)}")
        return None

# --- MAIN APP ---
st.title("📊 Customer Churn Analysis & Prediction Dashboard")

# Create tabs
tab1, tab2, tab3 = st.tabs(["📈 Exploratory Data Analysis", "🤖 Churn Prediction Model", "🔍 Data Overview"])

with tab1:
    if df_eda is not None and not df_eda.empty:
        st.markdown("""
        ## Exploratory Data Analysis
        Explore customer behavior and churn patterns using the filters below.
        """)

        # --- SIDEBAR FILTERS (Important filters only) ---
        st.sidebar.header("🔍 EDA Filters")
        
        # Important filters only
        important_filters = []
        
        # Churn status (always important)
        if 'Churn' in df_eda.columns:
            churn_status = st.sidebar.multiselect(
                "Churn Status",
                options=df_eda["Churn"].unique(),
                default=df_eda["Churn"].unique()
            )
            important_filters.append(('Churn', churn_status))
        
        # Tenure filter (important)
        if 'Tenure' in df_eda.columns:
            min_tenure, max_tenure = int(df_eda["Tenure"].min()), int(df_eda["Tenure"].max())
            tenure_range = st.sidebar.slider(
                "Tenure Range (months)",
                min_tenure, max_tenure, (min_tenure, max_tenure)
            )
            important_filters.append(('Tenure', tenure_range))
        
        # Satisfaction Score (important)
        if 'SatisfactionScore' in df_eda.columns:
            satisfaction_scores = st.sidebar.multiselect(
                "Satisfaction Score",
                options=sorted(df_eda["SatisfactionScore"].unique()),
                default=sorted(df_eda["SatisfactionScore"].unique())
            )
            important_filters.append(('SatisfactionScore', satisfaction_scores))
        
        # City Tier (important)
        if 'CityTier' in df_eda.columns:
            city_tiers = st.sidebar.multiselect(
                "City Tier",
                options=sorted(df_eda["CityTier"].unique()),
                default=sorted(df_eda["CityTier"].unique())
            )
            important_filters.append(('CityTier', city_tiers))
        
        # Complain status (important)
        if 'Complain' in df_eda.columns:
            complain_status = st.sidebar.multiselect(
                "Complain Status",
                options=df_eda["Complain"].unique(),
                default=df_eda["Complain"].unique()
            )
            important_filters.append(('Complain', complain_status))

        # --- APPLY FILTERS ---
        df_filtered = df_eda.copy()
        
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
                st.dataframe(df_filtered, use_container_width=True)
                st.write(f"**Data Dimensions:** {len(df_filtered)} rows, {len(df_filtered.columns)} columns")

with tab2:
    st.header("🤖 Churn Prediction Model")
    st.markdown("""
    This section uses a **Random Forest Classifier** trained on all available features to predict customer churn.
    """)
    
    if df_pred is not None and 'Churn' in df_pred.columns:
        # Train or load model
        with st.spinner("Training machine learning model..."):
            model_result = train_churn_model(df_pred)
        
        if model_result is not None:
            st.success("✅ Model training completed!")
            
            # Display model performance
            st.subheader("📊 Model Performance")
            
            metrics = model_result['metrics']
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Accuracy", f"{metrics['accuracy']:.3f}")
            with col2:
                st.metric("AUC Score", f"{metrics['auc_score']:.3f}")
            with col3:
                st.metric("Precision", f"{metrics['precision']:.3f}")
            with col4:
                st.metric("Recall", f"{metrics['recall']:.3f}")
            
            # Additional metrics
            col5, col6, col7, col8 = st.columns(4)
            with col5:
                st.metric("F1-Score", f"{metrics['f1_score']:.3f}")
            with col6:
                st.metric("Training Samples", len(df_pred) - len(model_result['X_test']))
            with col7:
                st.metric("Test Samples", len(model_result['X_test']))
            with col8:
                st.metric("Features Used", len(model_result['feature_names']))
            
            # Best parameters
            with st.expander("🎯 View Best Hyperparameters"):
                st.json(model_result['best_params'])
            
            # Confusion Matrix
            st.subheader("📈 Confusion Matrix")
            cm = confusion_matrix(model_result['y_test'], model_result['y_pred'])
            cm_df = pd.DataFrame(
                cm, 
                index=['Actual Not Churn', 'Actual Churn'],
                columns=['Predicted Not Churn', 'Predicted Churn']
            )
            st.dataframe(cm_df.style.background_gradient(cmap='Blues'), use_container_width=True)
            
            # Feature Importance
            st.subheader("🔍 Feature Importance")
            feature_importance = pd.DataFrame({
                'Feature': model_result['feature_names'],
                'Importance': model_result['model'].feature_importances_
            }).sort_values('Importance', ascending=False)
            
            # Display top features
            st.bar_chart(feature_importance.head(10).set_index('Feature'))
            
            with st.expander("📋 View All Feature Importances"):
                st.dataframe(feature_importance, use_container_width=True)
            
            # --- PREDICTION INTERFACE ---
            st.markdown("---")
            st.subheader("🎯 Churn Risk Assessment")
            st.markdown("""
            Enter customer details below to predict churn probability. **All features are used** for accurate prediction.
            """)
            
            with st.form("prediction_form"):
                st.write("### Customer Information")
                
                # Organize inputs into columns
                col1, col2, col3 = st.columns(3)
                input_data = {}
                
                # Get all feature names for prediction
                feature_names = model_result['feature_names']
                
                # Create input fields for all features
                for i, feature in enumerate(feature_names):
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
                            elif feature == 'CashbackAmount':
                                min_val, max_val = 0, 1000
                            elif feature == 'HourSpendOnApp':
                                min_val, max_val = 0.0, 10.0
                            
                            input_data[feature] = st.slider(
                                f"{feature}",
                                min_val, max_val, default_val,
                                help=f"Range: {min_val} - {max_val}"
                            )
                        else:
                            # Categorical features
                            unique_vals = df[feature].unique()
                            input_data[feature] = st.selectbox(
                                f"{feature}",
                                options=unique_vals,
                                help=f"Select from {len(unique_vals)} options"
                            )
                
                submitted = st.form_submit_button("🔮 Predict Churn Risk", use_container_width=True)
                
                if submitted:
                    # Prepare input data
                    input_df = pd.DataFrame([input_data])
                    
                    # Encode categorical variables
                    for col in input_df.columns:
                        if col in model_result['label_encoders']:
                            le = model_result['label_encoders'][col]
                            if input_df[col].iloc[0] in le.classes_:
                                input_df[col] = le.transform(input_df[col])
                            else:
                                # Use most frequent class as fallback
                                input_df[col] = le.transform([le.classes_[0]])
                    
                    # Ensure correct column order and scale
                    input_df = input_df[feature_names]
                    input_scaled = model_result['scaler'].transform(input_df)
                    
                    # Make prediction
                    churn_probability = model_result['model'].predict_proba(input_scaled)[0, 1]
                    churn_prediction = model_result['model'].predict(input_scaled)[0]
                    
                    # Display results
                    st.markdown("---")
                    st.subheader("📋 Prediction Results")
                    
                    # Results in columns
                    result_col1, result_col2, result_col3 = st.columns(3)
                    
                    with result_col1:
                        st.metric(
                            "Churn Probability", 
                            f"{churn_probability:.1%}",
                            delta=f"{(churn_probability-0.5)*100:+.1f}% vs baseline" if churn_probability != 0.5 else None
                        )
                    
                    with result_col2:
                        confidence = max(churn_probability, 1-churn_probability)
                        st.metric("Prediction Confidence", f"{confidence:.1%}")
                    
                    with result_col3:
                        if churn_prediction == 1:
                            st.error("**HIGH RISK OF CHURN** ⚠️")
                            st.write("Immediate retention action recommended")
                        else:
                            st.success("**LOW RISK OF CHURN** ✅")
                            st.write("Customer likely to stay")
                    
                    # Visual probability gauge
                    st.write("**Risk Level Gauge:**")
                    st.progress(float(churn_probability))
                    st.caption(f"Churn probability: {churn_probability:.1%}")
                    
                    # Risk factors analysis
                    with st.expander("🔍 Detailed Risk Analysis"):
                        st.write("**Key Contributing Factors:**")
                        
                        # Get feature contributions
                        feature_contributions = []
                        for feature in feature_names:
                            importance = feature_importance[feature_importance['Feature'] == feature]['Importance'].values[0]
                            value = input_data[feature]
                            avg_value = df[feature].mean() if df[feature].dtype in ['int64', 'float64'] else None
                            
                            if avg_value is not None:
                                if (feature in ['Tenure', 'SatisfactionScore', 'CashbackAmount'] and value < avg_value) or \
                                   (feature in ['Complain'] and value > 0):
                                    feature_contributions.append((feature, importance, value, avg_value))
                        
                        # Sort by importance and show top 5
                        feature_contributions.sort(key=lambda x: x[1], reverse=True)
                        
                        for feature, importance, value, avg_value in feature_contributions[:5]:
                            st.write(f"- **{feature}**: {value} (avg: {avg_value:.1f}) - Impact: {importance:.3f}")
        
        else:
            st.error("❌ Model training failed. Please check your data.")
    else:
        st.error("❌ No suitable data available for model training. Ensure dataset contains 'Churn' column.")

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
st.markdown("**Customer Churn Analysis & Prediction Dashboard** | *Using Random Forest with all features for prediction*")

# Sidebar info
st.sidebar.markdown("---")
st.sidebar.write("**ℹ️ App Info**")
st.sidebar.write(f"- Data shape: {df.shape if df is not None else 'N/A'}")
st.sidebar.write(f"- EDA filters: {len(important_filters) if 'important_filters' in locals() else 0} active")
st.sidebar.write("*For prediction, all features are used*")
