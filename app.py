import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from statsmodels.formula.api import ols

# Page configuration
st.set_page_config(page_title="Mite Population & Weather Analysis", layout="wide")

st.title("🕷️ Mite Population Dynamics & Weather Analysis")
st.markdown("""
This application analyzes the relationship between mite population dynamics and meteorological parameters 
across the years 2022 to 2025.
""")

# 1. Load Data
@st.cache_data
def load_data():
    # Loading the provided dataset
    df = pd.read_csv('mite_weather_2022_2025.csv')
    # Cleaning column names (removing leading/trailing spaces)
    df.columns = df.columns.str.strip()
    return df

try:
    df = load_data()
    
    # Sidebar Filters
    st.sidebar.header("Filter Data")
    years = st.sidebar.multiselect("Select Years", options=df['Year'].unique(), default=df['Year'].unique())
    filtered_df = df[df['Year'].isin(years)]

    # Tabs for different analyses
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Descriptive Stats", "📈 Trends", "🔗 Correlations", "🤖 Regression"])

    with tab1:
        st.header("Data Summary")
        st.dataframe(filtered_df.describe())
        st.subheader("Raw Data View")
        st.write(filtered_df)

    with tab2:
        st.header("Mite Population Trends")
        fig_trend = px.line(filtered_df, x='SMW', y='Mite', color='Year', markers=True,
                            title='Mite Population across Standard Meteorological Weeks (SMW)')
        st.plotly_chart(fig_trend, use_container_width=True)
        
        st.subheader("Weather vs Mite Comparison")
        weather_var = st.selectbox("Select Weather Variable to Compare", 
                                   ['Tmax', 'Tmin', 'VP', 'RH', 'WS', 'BSSH', 'Rainfall'])
        
        fig_comp = px.scatter(filtered_df, x=weather_var, y='Mite', color='Year', 
                               trendline="ols", title=f"Relationship: {weather_var} vs Mite Population")
        st.plotly_chart(fig_comp, use_container_width=True)

    with tab3:
        st.header("Correlation Matrix")
        corr = filtered_df.corr()
        fig_corr, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
        st.pyplot(fig_corr)
        
        st.info("**Interpretation:** Values closer to 1 or -1 indicate strong positive or negative relationships respectively.")

    with tab4:
        st.header("Statistical Modeling (Regression)")
        st.write("Determine which weather factors significantly impact the Mite population.")
        
        features = st.multiselect("Select Independent Variables (Weather Factors)", 
                                  ['Tmax', 'Tmin', 'VP', 'RH', 'WS', 'BSSH', 'Rainfall'],
                                  default=['Tmax', 'RH', 'Rainfall'])
        
        if features:
            formula = f"Mite ~ {' + '.join(features)}"
            model = ols(formula, data=filtered_df).fit()
            
            st.subheader("Regression Results Summary")
            st.text(model.summary().as_text())
            
            st.subheader("Key Findings")
            p_values = model.pvalues
            significant_vars = p_values[p_values < 0.05].index.tolist()
            if 'Intercept' in significant_vars: significant_vars.remove('Intercept')
            
            if significant_vars:
                st.success(f"Statistically significant factors (p < 0.05): {', '.join(significant_vars)}")
            else:
                st.warning("No variables reached statistical significance at the 0.05 level with current selection.")

except Exception as e:
    st.error(f"Error loading data: {e}")
    st.info("Please ensure the file 'mite_weather_2022_2025.csv' is in the same directory.")