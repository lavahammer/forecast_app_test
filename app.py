import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import warnings

# Suppress statsmodels warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(page_title="MW Asia Auto Forecasting App", layout="wide")

# Custom title with specific color
st.markdown("""
    <h1 style='text-align: center; color: #0000A0;'>MW Asia Auto Forecasting App</h1>
    """, unsafe_allow_html=True)

def calculate_sma_forecast(series, window_size, periods):
    """
    Simple Moving Average forecast
    - Window Size: Number of periods to average (larger window = smoother forecast)
    - Best for stable data with minimal trend/seasonality
    """
    if 'Total' in series:
        series = series.drop('Total')
        
    values = series.values
    non_zero_values = values[values != 0]
    
    if len(non_zero_values) >= window_size:
        ma = pd.Series(values).rolling(window=window_size).mean()
        forecast_value = ma.dropna().iloc[-1]
    else:
        forecast_value = non_zero_values.mean() if len(non_zero_values) > 0 else values.mean()
    
    return np.array([forecast_value] * periods)

def calculate_ema_forecast(series, alpha, periods):
    """
    Exponential Moving Average forecast
    - Alpha (α): Smoothing factor between 0 and 1
        - Higher α (closer to 1) = More weight to recent data
        - Lower α (closer to 0) = More weight to historical data
    - Best for data with gradual trends
    """
    if 'Total' in series:
        series = series.drop('Total')
        
    values = series.values
    non_zero_values = values[values != 0]
    
    if len(non_zero_values) >= 13:
        try:
            model = ExponentialSmoothing(
                values,
                seasonal_periods=13
            )
            fitted = model.fit(
                smoothing_level=alpha,
                optimized=False
            )
            forecast = fitted.forecast(periods)
            return forecast
        except Exception as e:
            return np.array([non_zero_values.mean()] * periods)
    else:
        return np.array([non_zero_values.mean() if len(non_zero_values) > 0 else values.mean()] * periods)

def calculate_holtwinters_forecast(series, periods):
    """
    Holt-Winters forecast with Triple Exponential Smoothing
    - Automatically optimizes three components:
        1. Level (α): Base value smoothing
        2. Trend (β): Trend component smoothing
        3. Seasonal (γ): Seasonal pattern smoothing
    - Best for data with both trend and seasonality
    - Uses 13-period seasonality for this application
    """
    if 'Total' in series:
        series = series.drop('Total')
    
    values = series.values
    non_zero_values = values[values != 0]
    
    if len(non_zero_values) >= 2 * 13:
        try:
            model = ExponentialSmoothing(
                values,
                seasonal_periods=13,
                trend='add',
                seasonal='add'
            )
            fitted = model.fit(
                optimized=True,
                method='L-BFGS-B'
            )
            forecast = fitted.forecast(periods)
            return forecast
        except Exception as e:
            return np.array([non_zero_values.mean()] * periods)
    else:
        return np.array([non_zero_values.mean() if len(non_zero_values) > 0 else values.mean()] * periods)

# File upload
uploaded_file = st.file_uploader("Upload your sales data (Excel/CSV)", type=['csv', 'xlsx', 'xls'])

if uploaded_file is not None:
    try:
        # Load data
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
            
        # Basic preprocessing
        df['Year'] = df['Year'].astype(int)
        df['Period'] = df['Period'].astype(int)
        df['Year_Period'] = df['Year'].astype(str) + '-P' + df['Period'].astype(str).str.zfill(2)
        
        # Create filters
        col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
        
        with col1:
            st.markdown("### Year Selection")
            years = sorted(df['Year'].unique())
            selected_years = st.multiselect('Select Years', years, default=years)
        
        with col2:
            st.markdown("### Period Selection")
            periods = sorted(df['Period'].unique())
            selected_periods = st.multiselect('Select Periods (1-13)', periods, default=periods)
        
        with col3:
            st.markdown("### Data Type Selection")
            data_types = sorted(df['Data_Type'].unique())
            selected_data_type = st.selectbox('Select Data Type', data_types)
        
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
            aggregation_level = st.selectbox('Select View Level', aggregation_options)

        # Filter data
        filtered_df = df[
            (df['Year'].isin(selected_years)) & 
            (df['Period'].isin(selected_periods)) &
            (df['Data_Type'] == selected_data_type)
        ]

        # Create pivot table
        pivot_df = pd.pivot_table(
            filtered_df,
            values='Value',
            index=aggregation_level,
            columns='Year_Period',
            aggfunc='sum',
            fill_value=0
        )

        # Sort columns
        pivot_df = pivot_df.reindex(sorted(pivot_df.columns), axis=1)

        # Forecasting Settings
        st.markdown("### Forecasting Settings")
        
        with st.expander("📊 Model Information", expanded=False):
            st.markdown("""
            **Simple Moving Average (SMA)**
            - Uses the average of previous periods
            - Window Size: Larger window = smoother forecast, less responsive to recent changes
            - Best for: Stable data with minimal trend or seasonality
            
            **Exponential Moving Average (EMA)**
            - Weighted average with more weight to recent data
            - Alpha (α): Higher = more weight to recent data, lower = more weight to historical data
            - Best for: Data with gradual trends
            
            **Holt-Winters**
            - Triple exponential smoothing with trend and seasonality
            - Automatically optimizes level, trend, and seasonal components
            - Best for: Data with both trend and seasonal patterns
            - Uses 13-period seasonality for this application
            """)
        
        f_col1, f_col2, f_col3 = st.columns(3)
        
        with f_col1:
            forecast_model = st.selectbox(
                "Select Forecasting Model",
                ["Simple Moving Average", "Exponential Smoothing", "Holt-Winters"]
            )
        
        with f_col2:
            future_periods = st.number_input(
                "Number of Periods to Forecast",
                min_value=1,
                max_value=26,
                value=13
            )
            
        with f_col3:
            if forecast_model == "Simple Moving Average":
                window_size = st.slider(
                    "Window Size",
                    min_value=2,
                    max_value=12,
                    value=3,
                    help="Number of periods to average. Larger window = smoother forecast"
                )
            elif forecast_model == "Exponential Smoothing":
                alpha = st.slider(
                    "Smoothing Factor (α)",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.2,
                    help="Weight given to recent data. Higher = more weight to recent data"
                )

        if st.button("Generate Forecast"):
            with st.spinner("Generating forecasts..."):
                # Generate future period labels
                future_period_labels = []
                last_year, last_period = map(int, pivot_df.columns[-1].split('-P'))
                
                for i in range(future_periods):
                    next_period = last_period + i + 1
                    year = last_year + (next_period - 1) // 13
                    period = ((next_period - 1) % 13) + 1
                    future_period_labels.append(f"{year}-P{str(period).zfill(2)}")

                # Generate forecasts for each row
                for idx in pivot_df.index:
                    series = pivot_df.loc[idx]
                    try:
                        if forecast_model == "Simple Moving Average":
                            forecast = calculate_sma_forecast(series, window_size, future_periods)
                        elif forecast_model == "Exponential Smoothing":
                            forecast = calculate_ema_forecast(series, alpha, future_periods)
                        else:  # Holt-Winters
                            forecast = calculate_holtwinters_forecast(series, future_periods)
                        
                        # Add forecasts to pivot table
                        for i, period in enumerate(future_period_labels):
                            pivot_df.loc[idx, period] = forecast[i]
                    except Exception as e:
                        st.warning(f"Could not generate forecast for {idx}: {str(e)}")

            st.success("Forecast generated successfully!")

        # Add total column and sort
        pivot_df['Total'] = pivot_df.sum(axis=1)
        pivot_df = pivot_df.sort_values('Total', ascending=False)

        # Display the pivot table
        st.markdown("### Forecast Data Table")
        st.markdown(f"**Showing data for {selected_data_type} aggregated by {aggregation_level}**")
        
        # Format and display the table
        try:
            if 'future_period_labels' in locals():
                def highlight_forecasts(x):
                    if x.name in future_period_labels:
                        return ['background-color: #ffe6e6'] * len(x)
                    return [''] * len(x)
                
                styled_df = pivot_df.style\
                    .format("{:,.0f}")\
                    .apply(highlight_forecasts)
            else:
                styled_df = pivot_df.style.format("{:,.0f}")
            
            st.dataframe(styled_df, use_container_width=True)
            
        except Exception as e:
            st.dataframe(pivot_df.style.format("{:,.0f}"), use_container_width=True)

    except Exception as e:
        st.error(f"Error processing data: {str(e)}")
else:
    st.info("Please upload a file to begin analysis")
