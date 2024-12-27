import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import plotly.express as px
import plotly.graph_objects as go

# [Previous imports and CSS remain the same]

def create_forecast(data, model_type, forecast_periods, params=None):
    """Generate forecasts using selected model"""
    # Prepare data for forecasting
    y = data.values
    
    if model_type == "Simple Moving Average":
        window = params.get('window_size', 3)
        ma = pd.Series(y).rolling(window=window).mean()
        forecast = [ma.iloc[-1]] * forecast_periods
        fitted = ma.values
        
    elif model_type == "Exponential Smoothing":
        alpha = params.get('alpha', 0.2)
        model = ExponentialSmoothing(y, seasonal_periods=13, seasonal='add')
        fitted_model = model.fit(smoothing_level=alpha)
        forecast = fitted_model.forecast(forecast_periods)
        fitted = fitted_model.fittedvalues
        
    elif model_type == "Holt-Winters":
        model = ExponentialSmoothing(
            y,
            seasonal_periods=13,
            trend='add',
            seasonal='add'
        )
        fitted_model = model.fit()
        forecast = fitted_model.forecast(forecast_periods)
        fitted = fitted_model.fittedvalues
        
    elif model_type == "SARIMA":
        model = ARIMA(
            y, 
            order=params.get('order', (1,1,1)),
            seasonal_order=params.get('seasonal_order', (1,1,1,13))
        )
        fitted_model = model.fit()
        forecast = fitted_model.forecast(forecast_periods)
        fitted = fitted_model.fittedvalues
        
    return fitted, forecast

def calculate_metrics(actual, predicted):
    """Calculate forecast performance metrics"""
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    r2 = r2_score(actual, predicted)
    
    return {
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape,
        'R-squared': r2
    }

def main():
    # [Previous main UI code remains until the pivot table display]

    if uploaded_file is not None:
        # [Previous code for loading and filtering data]

        # Add Forecasting Section
        st.markdown("### Forecasting Settings")
        
        forecast_col1, forecast_col2, forecast_col3 = st.columns(3)
        
        with forecast_col1:
            forecast_model = st.selectbox(
                "Select Forecasting Model",
                ["Simple Moving Average", "Exponential Smoothing", "Holt-Winters", "SARIMA"],
                help="Choose the forecasting model to use"
            )
        
        with forecast_col2:
            future_periods = st.number_input(
                "Number of Periods to Forecast",
                min_value=1,
                max_value=26,
                value=13,
                help="Number of future periods to forecast"
            )
            
        with forecast_col3:
            # Model-specific parameters
            if forecast_model == "Simple Moving Average":
                window_size = st.slider("Window Size", 2, 12, 3)
                model_params = {'window_size': window_size}
            elif forecast_model == "Exponential Smoothing":
                alpha = st.slider("Smoothing Factor (α)", 0.0, 1.0, 0.2)
                model_params = {'alpha': alpha}
            elif forecast_model == "SARIMA":
                p = st.slider("AR Order (p)", 0, 3, 1)
                d = st.slider("Difference Order (d)", 0, 2, 1)
                q = st.slider("MA Order (q)", 0, 3, 1)
                model_params = {
                    'order': (p,d,q),
                    'seasonal_order': (1,1,1,13)
                }
            else:
                model_params = {}

        if st.button("Generate Forecast"):
            # Prepare time series data
            ts_data = pd.Series(
                pivot_df.iloc[:, :-1].values[0],  # Take first row for forecasting
                index=pd.to_datetime(pivot_df.columns[:-1])  # Exclude 'Total' column
            )

            # Generate forecast
            fitted_values, forecast_values = create_forecast(
                ts_data,
                forecast_model,
                future_periods,
                model_params
            )

            # Create future dates
            last_date = ts_data.index[-1]
            future_dates = pd.date_range(
                start=pd.Timestamp(last_date) + pd.DateOffset(months=1),
                periods=future_periods,
                freq='M'
            )

            # Calculate metrics
            metrics = calculate_metrics(ts_data.values[len(ts_data)-len(fitted_values):], fitted_values)

            # Display results
            st.markdown("### Forecast Results")
            
            # Plot actual vs forecast
            fig = go.Figure()
            
            # Historical data
            fig.add_trace(go.Scatter(
                x=ts_data.index,
                y=ts_data.values,
                name='Actual',
                line=dict(color='blue')
            ))
            
            # Fitted values
            fig.add_trace(go.Scatter(
                x=ts_data.index[-len(fitted_values):],
                y=fitted_values,
                name='Fitted',
                line=dict(color='green', dash='dash')
            ))
            
            # Forecast values
            fig.add_trace(go.Scatter(
                x=future_dates,
                y=forecast_values,
                name='Forecast',
                line=dict(color='red', dash='dash')
            ))
            
            fig.update_layout(
                title=f'{forecast_model} Forecast',
                xaxis_title='Date',
                yaxis_title='Value',
                showlegend=True
            )
            
            st.plotly_chart(fig, use_container_width=True)

            # Display metrics
            st.markdown("### Model Performance Metrics")
            metrics_df = pd.DataFrame({
                'Metric': metrics.keys(),
                'Value': metrics.values()
            })
            
            st.dataframe(metrics_df.style.format({
                'Value': '{:.2f}'
            }))

            # Display forecast values
            st.markdown("### Forecast Values")
            forecast_df = pd.DataFrame({
                'Date': future_dates,
                'Forecasted Value': forecast_values
            })
            
            st.dataframe(forecast_df.style.format({
                'Forecasted Value': '{:.0f}'
            }))

    # [Rest of the previous code remains the same]

if __name__ == "__main__":
    main()
