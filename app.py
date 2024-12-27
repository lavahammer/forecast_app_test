import streamlit as st
import pandas as pd
import plotly.express as px

# Set page config
st.set_page_config(page_title="My Dashboard", layout="wide")

# Add title
st.title("My Simple Dashboard")

# Add file uploader
uploaded_file = st.file_uploader("Upload your CSV file", type=['csv'])

if uploaded_file is not None:
    # Read CSV
    df = pd.read_csv(uploaded_file)
    
    # Show data
    st.subheader("Your Data")
    st.dataframe(df)
    
    # Select columns for plotting
    st.subheader("Create Plot")
    x_col = st.selectbox("Select X axis", df.columns)
    y_col = st.selectbox("Select Y axis", df.columns)
    
    # Create plot
    fig = px.line(df, x=x_col, y=y_col, title=f"{y_col} vs {x_col}")
    st.plotly_chart(fig, use_container_width=True)
    
    # Show statistics
    st.subheader("Basic Statistics")
    st.write(df.describe())
else:
    st.info("Please upload a CSV file to begin")

# Add sidebar
with st.sidebar:
    st.header("About")
    st.write("This is a simple dashboard that lets you:")
    st.write("1. Upload CSV data")
    st.write("2. View your data")
    st.write("3. Create plots")
    st.write("4. See basic statistics")
