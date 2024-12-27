import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def generate_future_periods(last_period, num_periods):
    """Generate future period labels"""
    future_periods = []
    current_year, current_period = map(int, last_period.split('-P'))
    
    for i in range(num_periods):
        next_period = current_period + i + 1
        year = current_year + (next_period - 1) // 13
        period = ((next_period - 1) % 13) + 1
        future_periods.append(f"{year}-P{str(period).zfill(2)}")
    
    return future_periods

def forecast_values(data, model_type, periods, params=None):
    """Generate forecasts using selected model"""
    values = data.values
    
    if model_type == "Simple Moving Average":
        window = params.get('window_size', 3)
        ma = pd.Series(values).rolling(window=window).mean()
        forecast = [ma.iloc[-1]] * periods
        fitted = ma.values
        
    elif model_type == "Exponential Smoothing":
        model = ExponentialSmoothing(
            values,
            seasonal_periods=13,
            seasonal='add'
        )
        fit = model.fit(smoothing_level=params.get('alpha', 0.2))
        forecast = fit.forecast(periods)
        fitted = fit.fittedvalues
        
    elif model_type == "Holt-Winters":
        model = ExponentialSmoothing(
            values,
            seasonal_periods=13,
            trend='add',
            seasonal='add'
        )
        fit = model.fit()
        forecast = fit.forecast(periods)
        fitted = fit.fittedvalues
    
    return fitted, forecast

def calculate_metrics(actual, predicted):
    """Calculate forecast accuracy metrics"""
    metrics = {
        'MAE': mean_absolute_error(actual, predicted),
        'RMSE': np.sqrt(mean_squared_error(actual, predicted)),
        'MAPE': np.mean(np.abs((actual - predicted) / actual)) * 100
    }
    return metrics

# Main App
st.set_page_config(page_title="MW Asia Auto Forecasting App", layout="wide")
st.title("MW Asia Auto Forecasting App")

uploaded_file = st.file_uploader("Upload your sales data (Excel/CSV)", type=['csv', 'xlsx', 'xls'])

if uploaded_file is not None:
    try:
        # Load data
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
            
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

        # Filter and create pivot
        filtered_df = df[
            (df['Year'].isin(selected_years)) & 
            (df['Period'].isin(selected_periods)) &
            (df['Data_Type'] == selected_data_type)
        ]

        pivot_df = pd.pivot_table(
            filtered_df,
            values='Value',
            index=aggregation_level,
            columns='Year_Period',
            aggfunc='sum',
            fill_value=0
        )
        
        pivot_df = pivot_df.reindex(sorted(pivot_df.columns), axis=1)

        # Forecasting Section
        st.markdown("### Forecasting Settings")
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
                window_size = st.slider("Window Size", 2, 12, 3)
                model_params = {'window_size': window_size}
            elif forecast_model == "Exponential Smoothing":
                alpha = st.slider("Smoothing Factor (α)", 0.0, 1.0, 0.2)
                model_params = {'alpha': alpha}
            else:
                model_params = {}

        if st.button("Generate Forecast"):
            # Generate forecasts for each row
            last_period = pivot_df.columns[-1]
            future_period_labels = generate_future_periods(last_period, future_periods)
            
            # Create forecast columns
            for idx in pivot_df.index:
                series = pivot_df.loc[idx]
                fitted, forecast = forecast_values(
                    series,
                    forecast_model,
                    future_periods,
                    model_params
                )
                
                # Add forecasts to pivot table
                for i, period in enumerate(future_period_labels):
                    pivot_df.loc[idx, period] = forecast[i]
                
                # Calculate metrics for this row
                metrics = calculate_metrics(
                    series.values[len(series)-len(fitted):],
                    fitted
                )
                
                # Store metrics (optional)
                if 'metrics' not in st.session_state:
                    st.session_state.metrics = {}
                st.session_state.metrics[idx] = metrics

        # Add total and sort
        pivot_df['Total'] = pivot_df.sum(axis=1)
        pivot_df = pivot_df.sort_values('Total', ascending=False)

        # Display the main table
        st.markdown("### Forecast Data Table")
        st.markdown(f"**Showing data for {selected_data_type} aggregated by {aggregation_level}**")
        
        # Style the table to highlight forecasts
        def highlight_forecasts(x):
            if x.name in future_period_labels:
                return ['background-color: #ffe6e6'] * len(x)
            return [''] * len(x)
        
        styled_df = pivot_df.style\
            .format("{:,.0f}")\
            .apply(highlight_forecasts)
        
        st.dataframe(styled_df, use_container_width=True)

        # Display metrics if available
        if 'metrics' in st.session_state:
            st.markdown("### Forecast Performance Metrics")
            metrics_df = pd.DataFrame.from_dict(
                st.session_state.metrics,
                orient='index'
            )
            st.dataframe(
                metrics_df.style.format("{:.2f}"),
                use_container_width=True
            )

    except Exception as e:
        st.error(f"Error processing data: {str(e)}")
else:
    st.info("Please upload a file to begin analysis")
