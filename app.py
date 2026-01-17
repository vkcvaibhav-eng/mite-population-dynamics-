import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from statsmodels.formula.api import ols

# Page configuration
st.set_page_config(page_title="Mite Analysis Tool", layout="wide")

st.title("🕷️ Mite Population & Weather Analysis")

# --- FILE UPLOADER SECTION ---
st.sidebar.header("1. Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload your mite_weather CSV file", type=["csv"])

def load_data(file):
    df = pd.read_csv(file)
    df.columns = df.columns.str.strip() # Cleans hidden spaces in column names
    return df

if uploaded_file is not None:
    df = load_data(uploaded_file)
    
    # --- FILTERS ---
    st.sidebar.header("2. Filter Data")
    years = st.sidebar.multiselect("Select Years", options=df['Year'].unique(), default=df['Year'].unique())
    filtered_df = df[df['Year'].isin(years)]

    # Tabs for Organization
    tab1, tab2, tab3 = st.tabs(["📈 Trends", "🔗 Correlations", "📊 Statistical Model"])

    with tab1:
        st.subheader("Population Dynamics over SMW")
        fig_line = px.line(filtered_df, x='SMW', y='Mite', color='Year', markers=True)
        st.plotly_chart(fig_line, use_container_width=True)

    with tab2:
        st.subheader("Correlation Heatmap (Weather vs Mites)")
        # Selecting only numeric columns for correlation
        numeric_df = filtered_df.select_dtypes(include=['float64', 'int64'])
        corr = numeric_df.corr()
        
        fig_corr, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(corr, annot=True, cmap='RdYlGn', center=0, ax=ax)
        st.pyplot(fig_corr)

    with tab3:
        st.subheader("Regression Analysis")
        # Let user choose factors for the model
        factors = st.multiselect("Select Weather Factors", 
                                 ['Tmax', 'Tmin', 'VP', 'RH', 'WS', 'BSSH', 'Rainfall'],
                                 default=['Tmax', 'RH'])
        
        if factors:
            # Clean column names for the formula (removes spaces)
            formula = f"Q('Mite') ~ {' + '.join([f'Q({chr(39)}{f}{chr(39)})' for f in factors])}"
            try:
                model = ols(formula, data=filtered_df).fit()
                st.write(model.summary())
            except Exception as e:
                st.error(f"Model Error: {e}")

else:
    st.info("👈 Please upload the CSV file using the sidebar to begin.")
    # Show an example of how the data should look
    st.image("https://raw.githubusercontent.com/dataprofessor/streamlit_freecodecamp/master/app_8_classification_iris/iris-flower.png", caption="Ensure your CSV has columns: Year, SMW, Mite, Tmax, etc.", width=400)
