import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Sales Analysis Dashboard", layout="wide")

def load_data(file):
    if file.name.endswith('.csv'):
        df = pd.read_csv(file)
    else:
        df = pd.read_excel(file)
    return df

def main():
    st.title("Sales Analysis Dashboard")

    # File upload
    uploaded_file = st.file_uploader("Upload your sales data (Excel/CSV)", type=['csv', 'xlsx', 'xls'])

    if uploaded_file is not None:
        # Load data
        df = load_data(uploaded_file)
        
        # Create columns for filters
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Year filter
            years = sorted(df['Year'].unique())
            selected_years = st.multiselect('Select Years', years, default=years)
            
        with col2:
            # Period filter
            periods = sorted(df['Period'].unique())
            selected_periods = st.multiselect('Select Periods', periods, default=periods)
            
        with col3:
            # Aggregation level selector
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
                'Select Aggregation Level',
                aggregation_options
            )

        # Filter data
        filtered_df = df[
            (df['Year'].isin(selected_years)) & 
            (df['Period'].isin(selected_periods))
        ]

        # Aggregate data
        agg_df = filtered_df.groupby([aggregation_level, 'Year', 'Period'])['Value'].sum().reset_index()

        # Create time series for each aggregation value
        st.subheader(f"Value Trends by {aggregation_level}")
        
        # Create unique identifier for time
        agg_df['Time'] = agg_df['Year'].astype(str) + '-' + agg_df['Period'].astype(str)
        
        # Plot
        fig = px.line(
            agg_df,
            x='Time',
            y='Value',
            color=aggregation_level,
            title=f'Value Trends by {aggregation_level}',
            height=600
        )
        st.plotly_chart(fig, use_container_width=True)

        # Summary statistics
        st.subheader("Summary Statistics")
        summary_df = filtered_df.groupby(aggregation_level)['Value'].agg([
            'count', 'mean', 'sum'
        ]).reset_index()
        summary_df = summary_df.sort_values('sum', ascending=False)
        summary_df.columns = [aggregation_level, 'Count', 'Average Value', 'Total Value']
        st.dataframe(summary_df.style.format({
            'Average Value': '{:,.2f}',
            'Total Value': '{:,.2f}'
        }))

        # Show raw data option
        if st.checkbox("Show Raw Data"):
            st.subheader("Raw Data")
            st.dataframe(filtered_df)

if __name__ == "__main__":
    main()
