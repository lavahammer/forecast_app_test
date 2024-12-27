import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
import plotly.graph_objects as go

def generate_forecast(data, model_type, periods, params=None):
    """Generate forecasts for each row in the pivot table"""
    forecasts = {}
    
    for index in data.index:
        series = data.loc[index, :]
        y = series.values
        
        if model_type == "Simple Moving Average":
            window = params.get('window_size', 3)
            ma = pd.Series(y).rolling(window=window).mean()
            forecast = [ma.iloc[-1]] * periods
            
        elif model_type == "Exponential Smoothing":
            model = ExponentialSmoothing(y, seasonal_periods=13, seasonal='add')
            fitted_model = model.fit()
            forecast = fitted_model.forecast(periods)
            
        elif model_type == "Holt-Winters":
            model = ExponentialSmoothing(y, seasonal_periods=13, trend='add', seasonal='add')
            fitted_model = model.fit()
            forecast = fitted_model.forecast(periods)
            
        elif model_type == "SARIMA":
            model = ARIMA(y, order=(1,1,1), seasonal_order=(1,1,1,13))
            fitted_model = model.fit()
            forecast = fitted_model.forecast(periods)
            
        forecasts[index] = forecast
        
    return forecasts

def main():
    st.set_page_config(page_title="MW Asia Auto Forecasting App", layout="wide")
    
    # [Previous CSS and title code remains the same]

    uploaded_file = st.file_uploader(
        "Upload your sales data (Excel/CSV)",
        type=['csv', 'xlsx', 'xls']
    )

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

            # [Previous filter code remains the same]

            # Filter data
            filtered_df = df[
                (df['Year'].isin(selected_years)) & 
                (df['Period'].isin(selected_periods)) &
                (df['Data_Type'] == selected_data_type)
            ]

            if not filtered_df.empty:
                # Create pivot table
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

                # Forecasting settings
                st.markdown("### Forecasting Settings")
                forecast_col1, forecast_col2 = st.columns(2)
                
                with forecast_col1:
                    forecast_model = st.selectbox(
                        "Select Forecasting Model",
                        ["Simple Moving Average", "Exponential Smoothing", 
                         "Holt-Winters", "SARIMA"]
                    )
                
                with forecast_col2:
                    future_periods = st.number_input(
                        "Number of Periods to Forecast",
                        min_value=1,
                        max_value=26,
                        value=13
                    )

                if st.button("Generate Forecast"):
                    # Generate future period labels
                    last_year, last_period = map(int, pivot_df.columns[-1].split('-P'))
                    future_periods_list = []
                    
                    for i in range(future_periods):
                        next_period = last_period + 1 + i
                        next_year = last_year + (next_period - 1) // 13
                        adjusted_period = ((next_period - 1) % 13) + 1
                        future_periods_list.append(
                            f"{next_year}-P{str(adjusted_period).zfill(2)}"
                        )

                    # Generate forecasts
                    forecasts = generate_forecast(
                        pivot_df.iloc[:, :-1],  # Exclude Total column
                        forecast_model,
                        future_periods,
                        {}  # Add model parameters if needed
                    )

                    # Add forecast columns to pivot table
                    for i, period in enumerate(future_periods_list):
                        pivot_df[period] = pd.Series(
                            {index: forecasts[index][i] for index in pivot_df.index}
                        )

                # Add total column
                pivot_df['Total'] = pivot_df.sum(axis=1)
                pivot_df = pivot_df.sort_values('Total', ascending=False)

                # Display the pivot table
                st.markdown("### Forecast Data Table")
                st.markdown(f"**Showing data for {selected_data_type} aggregated by {aggregation_level}**")
                
                # Format numbers and highlight forecasted values
                def highlight_forecast(val):
                    return 'background-color: #ffe6e6' if pd.isna(val) else ''
                
                formatted_df = pivot_df.style\
                    .format("{:,.0f}")\
                    .applymap(highlight_forecast)
                
                st.dataframe(formatted_df, use_container_width=True)

        except Exception as e:
            st.error(f"Error processing data: {str(e)}")
    else:
        st.info("Please upload a file to begin analysis")

if __name__ == "__main__":
    main()
