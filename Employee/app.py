import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
# ... other imports
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# 1. PAGE CONFIGURATION (Must be the very first Streamlit command)
st.set_page_config(page_title="Salary AI Dashboard", layout="wide")

# 2. DATA LOADING SECTION
# We define a function to load the data safely
@st.cache_data
def get_data():
    try:
        return pd.read_csv('employee_data.csv')
    except FileNotFoundError:
        return None

# We call the function and define 'df' right at the top
df = get_data()

# 3. GLOBAL CHECK
if df is None:
    st.error("❌ ERROR: 'employee_data.csv' not found. Please run your Step 1 script to create the file first!")
else:
    # 4. PREPROCESSING & MODELING
    # This runs behind the scenes so the prediction button works
    df_ml = df.copy()
    le = LabelEncoder()
    for col in ['Gender', 'Education', 'Department']:
        df_ml[col] = le.fit_transform(df_ml[col])

    X = df_ml.drop(['Employee_ID', 'Salary', 'Salary_Category'], axis=1)
    y = df_ml['Salary']
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    # 5. SIDEBAR DESIGN
    st.sidebar.header("🔧 Prediction Inputs")
    u_age = st.sidebar.slider("Age", 22, 60, 30)
    u_exp = st.sidebar.slider("Experience", 0, 35, 10)
    u_hours = st.sidebar.slider("Weekly Hours", 30, 60, 40)
    u_score = st.sidebar.slider("Performance Score", 1.0, 5.0, 3.5)
    u_proj = st.sidebar.slider("Projects Completed", 1, 20, 5)

    # 6. MAIN DASHBOARD
    st.title("🚀 Employee Salary Prediction System")
    
    tab1, tab2, tab3 = st.tabs(["💡 Prediction", "📊 Visual Analytics", "📁 Raw Data"])

    with tab1:
        st.subheader("Predict Salary for New Employee")
        if st.button("Click to Predict"):
            # Dummy encoding (1, 1, 1) represents default categories for this demo
            input_df = pd.DataFrame([[u_age, 1, 1, 1, u_exp, u_hours, u_proj, u_score]], columns=X.columns)
            result = model.predict(input_df)
            st.success(f"### The Estimated Salary is: ${result[0]:,.2f}")

    with tab2:
        st.subheader("Insights & Correlations")
        col1, col2 = st.columns(2)
        with col1:
            fig1, ax1 = plt.subplots()
            sns.histplot(df['Salary'], kde=True, color='blue', ax=ax1)
            ax1.set_title("Salary Distribution")
            st.pyplot(fig1)
        with col2:
            fig2, ax2 = plt.subplots()
            sns.scatterplot(data=df, x='Experience', y='Salary', hue='Department', ax=ax2)
            ax2.set_title("Experience vs Salary")
            st.pyplot(fig2)

    with tab3:
        st.subheader("Full Employee Dataset")
        st.dataframe(df)