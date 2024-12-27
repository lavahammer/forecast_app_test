import streamlit as st
import pandas as pd
import plotly.express as px

# Page configuration
st.set_page_config(
    page_title="MW Asia Auto Forecasting App",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stSelectbox, .stMultiSelect {
        margin-bottom: 1rem;
    }
    .title-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 5px;
        margin-bottom: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

def load_data(file):
    if file.name.endswith('.csv'):
        df = pd.read_csv(file)
    else:
        df = pd.read_excel(file)
    
    # Ensure Year and Period are integers
    df['Year'] = df['Year'].astype(int)
    df['Period'] = df['Period'].astype(int)
    
    # Create Year-Period combination
    df['Year_Period'] = df['Year'].astype(str) + '-P' + df['Period'].astype(str).str.zfill(2)
    
    return df

def main():
    # Title with custom styling
    st.markdown("""
        <div class='title-container'>
            <h1 style='text-align: center; color: #1f77b4;'>MW Asia Auto Forecasting App</h1>
        </div>
    """, unsafe_allow_html=True)

    # File upload with better styling
    uploaded_file = st.file_uploader(
        "Upload your sales data (Excel/CSV)",
        type=['csv', 'xlsx', 'xls'],
        help="Upload your sales data file in CSV or Excel format"
    )

    if uploaded_file is not None:
        # Load data
        df = load_data(uploaded_file)
        
        # Create two columns for filters
        col1, col2, col3 = st.columns([1, 1, 2])
        
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
            (df['Period'].isin(selected_periods))
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

        # Sort columns by Year-Period
        pivot_df = pivot_df.reindex(sorted(pivot_df.columns), axis=1)

        # Add total column
        pivot_df['Total'] = pivot_df.sum(axis=1)
        
        # Sort by total descending
        pivot_df = pivot_df.sort_values('Total', ascending=False)

        # Display the pivot table
        st.markdown("### Forecast Data Table")
        st.markdown(f"**Showing data aggregated by {aggregation_level}**")
        
        # Format the numbers in the dataframe
        formatted_df = pivot_df.style.format("{:,.0f}")
        
        # Display the formatted table
        st.dataframe(formatted_df, use_container_width=True)

        # Export options
        st.markdown("### Export Options")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Export to Excel"):
                # Convert to Excel
                output = pivot_df.to_excel()
                st.download_button(
                    label="Download Excel file",
                    data=output,
                    file_name="forecast_data.xlsx",
                    mime="application/vnd.ms-excel"
                )
        
        with col2:
            if st.button("Export to CSV"):
                # Convert to CSV
                csv = pivot_df.to_csv()
                st.download_button(
                    label="Download CSV file",
                    data=csv,
                    file_name="forecast_data.csv",
                    mime="text/csv"
                )

        # Additional Statistics
        st.markdown("### Summary Statistics")
        stats_df = pd.DataFrame({
            'Total Value': pivot_df['Total'],
            'Average per Period': pivot_df.iloc[:, :-1].mean(axis=1),
            'Number of Periods': pivot_df.iloc[:, :-1].count(axis=1)
        })
        
        st.dataframe(stats_df.style.format({
            'Total Value': '{:,.0f}',
            'Average per Period': '{:,.0f}'
        }))

if __name__ == "__main__":
    main()
