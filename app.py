import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import plotly.express as px
import plotly.graph_objects as go

# Page configuration
st.set_page_config(
    page_title="MW Asia Auto Forecasting App",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {padding: 2rem;}
    .stSelectbox, .stMultiSelect {margin-bottom: 1rem;}
    .title-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 5px;
        margin-bottom: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown("""
    <div class='title-container'>
        <h1 style='text-align: center; color: #1f77b4;'>MW Asia Auto Forecasting App</h1>
    </div>
""", unsafe_allow_html=True)

# File upload section
uploaded_file = st.file_uploader(
    "Upload your sales data (Excel/CSV)",
    type=['csv', 'xlsx', 'xls'],
    help="Upload your sales data file in CSV or Excel format"
)

# Main application logic
if uploaded_file is not None:
    try:
        # Load data
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        
        # Data preprocessing
        df['Year'] = df['Year'].astype(int)
        df['Period'] = df['Period'].astype(int)
        df['Year_Period'] = df['Year'].astype(str) + '-P' + df['Period'].astype(str).str.zfill(2)
        
        # Create columns for filters
        col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
        
        with col1:
            st.markdown("### Year Selection")
            years = sorted(df['Year'].unique())
            selected_years = st.multiselect(
                'Select Years',
                years,
                default=years,
                help="Select one or multiple years"
            )
            
        with col2:
            st.markdown("### Period Selection")
            periods = sorted(df['Period'].unique())
            selected_periods = st.multiselect(
                'Select Periods (1-13)',
                periods,
                default=periods,
                help="Select one or multiple periods"
            )
        
        with col3:
            st.markdown("### Data Type Selection")
            data_types = sorted(df['Data_Type'].unique())
            selected_data_type = st.selectbox(
                'Select Data Type',
                data_types,
                help="Choose which data type to display"
            )
            
        with col4:
            st.markdown("### View Selection")
            aggregation_options = [
                'GRD_code_Consolidated',
                'Description_Consolidated',
                'Category',
                'Sub_Category',
                'Segment',
                'Brand',
                'Product Range',
                'Channel_1'
            ]
            aggregation_level = st.selectbox(
                'Select View Level',
                aggregation_options,
                help="Choose how to aggregate the data"
            )

        # Filter data
        filtered_df = df[
            (df['Year'].isin(selected_years)) & 
            (df['Period'].isin(selected_periods)) &
            (df['Data_Type'] == selected_data_type)
        ]

        # Create pivot table
        if not filtered_df.empty:
            pivot_df = pd.pivot_table(
                filtered_df,
                values='Value',
                index=aggregation_level,
                columns='Year_Period',
                aggfunc='sum',
                fill_value=0
            )

            # Sort columns by Year-Period
            pivot_df = pivot_df.reindex(sorted(pivot_df.columns), axis=1)
            pivot_df['Total'] = pivot_df.sum(axis=1)
            pivot_df = pivot_df.sort_values('Total', ascending=False)

            # Display the pivot table
            st.markdown("### Forecast Data Table")
            st.markdown(f"**Showing data for {selected_data_type} aggregated by {aggregation_level}**")
            
            formatted_df = pivot_df.style.format("{:,.0f}")
            st.dataframe(formatted_df, use_container_width=True)

            # Forecasting section
            st.markdown("### Forecasting Settings")
            forecast_col1, forecast_col2, forecast_col3 = st.columns(3)
            
            with forecast_col1:
                forecast_model = st.selectbox(
                    "Select Forecasting Model",
                    ["Simple Moving Average", "Exponential Smoothing", "Holt-Winters", "SARIMA"]
                )
            
            with forecast_col2:
                future_periods = st.number_input(
                    "Number of Periods to Forecast",
                    min_value=1,
                    max_value=26,
                    value=13
                )
                
            with forecast_col3:
                if forecast_model == "Simple Moving Average":
                    window_size = st.slider("Window Size", 2, 12, 3)
                    model_params = {'window_size': window_size}
                elif forecast_model == "Exponential Smoothing":
                    alpha = st.slider("Smoothing Factor (α)", 0.0, 1.0, 0.2)
                    model_params = {'alpha': alpha}
                else:
                    model_params = {}

            # Additional features can be added here

    except Exception as e:
        st.error(f"Error processing data: {str(e)}")
else:
    st.info("Please upload a file to begin analysis")
