import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing

st.set_page_config(page_title="MW Asia Auto Forecasting App", layout="wide")
st.title("MW Asia Auto Forecasting App")

def forecast_values(series, model_type, periods):
    """Generate forecasts using selected model"""
    values = series.values
    if model_type == "Holt-Winters":
        model = ExponentialSmoothing(values, seasonal_periods=13, trend='add', seasonal='add')
        fitted_model = model.fit()
        forecast = fitted_model.forecast(periods)
        return forecast

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
        f_col1, f_col2 = st.columns(2)
        
        with f_col1:
            forecast_model = st.selectbox(
                "Select Forecasting Model",
                ["Holt-Winters"]
            )
        
        with f_col2:
            future_periods = st.number_input(
                "Number of Periods to Forecast",
                min_value=1,
                max_value=26,
                value=13
            )

        if st.button("Generate Forecast"):
            # Initialize future_period_labels list
            future_period_labels = []
            
            # Generate future period labels
            last_year, last_period = map(int, pivot_df.columns[-1].split('-P'))
            for i in range(future_periods):
                next_period = last_period + i + 1
                year = last_year + (next_period - 1) // 13
                period = ((next_period - 1) % 13) + 1
                future_period_labels.append(f"{year}-P{str(period).zfill(2)}")

            # Generate forecasts for each row
            for idx in pivot_df.index:
                series = pivot_df.loc[idx]
                forecast = forecast_values(series, forecast_model, future_periods)
                
                # Add forecasts to pivot table
                for i, period in enumerate(future_period_labels):
                    pivot_df.loc[idx, period] = forecast[i]

        # Add total column and sort
        pivot_df['Total'] = pivot_df.sum(axis=1)
        pivot_df = pivot_df.sort_values('Total', ascending=False)

        # Display the pivot table
        st.markdown("### Forecast Data Table")
        st.markdown(f"**Showing data for {selected_data_type} aggregated by {aggregation_level}**")
        
        # Format and display the table
        try:
            # Only apply highlighting if forecasting has been done
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
