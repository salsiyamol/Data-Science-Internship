import streamlit as st
import joblib
import pandas as pd

# Load the saved model
model = joblib.load('model.pkl')

st.set_page_config(page_title="Titanic Survival Predictor", page_icon="🚢")

st.title("🚢 Titanic Survival Predictor")
st.write("Enter the passenger's details below to predict if they would have survived.")

# Input fields
pclass = st.selectbox("Passenger Class (1=First, 2=Second, 3=Third)", [1, 2, 3])
sex = st.radio("Sex", ["male", "female"])
age = st.slider("Age", 0, 100, 25)
sibsp = st.number_input("Number of Siblings/Spouses aboard", 0, 10, 0)
parch = st.number_input("Number of Parents/Children aboard", 0, 10, 0)

# Convert sex to numerical format (matching training)
sex_val = 0 if sex == "male" else 1

# Predict button
if st.button("Predict"):
    # Create a dataframe for prediction
    input_data = pd.DataFrame([[pclass, sex_val, age, sibsp, parch]], 
                              columns=['Pclass', 'Sex', 'Age', 'SibSp', 'Parch'])
    
    prediction = model.predict(input_data)
    
    if prediction[0] == 1:
        st.success("The passenger would have survived! 🎉")
    else:
        st.error("The passenger would not have survived. 😔")
