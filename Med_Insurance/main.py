import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import streamlit as st
from sklearn.preprocessing import OrdinalEncoder


st.set_page_config(page_title='Medical Insurance Cost Prediction')
st.markdown(f'<h1 style="text-align: center;">Medical Insurance Cost Prediction</h1>', unsafe_allow_html=True)

insurance_dataset = pd.read_csv("C:\\Users\\91915\\OneDrive\\Desktop\\Med_Insurance\\insurance.csv")
enc=OrdinalEncoder()

# encoding sex column
insurance_dataset.replace({'sex':{'male':0,'female':1}}, inplace=True)

# encoding 'smoker' column
insurance_dataset.replace({'smoker':{'yes':0,'no':1}}, inplace=True)

# encoding 'region' column
insurance_dataset.replace({'region':{'southeast':0,'southwest':1,'northeast':2,'northwest':3}}, inplace=True)


# Define Independent and Dependent Variables
X = insurance_dataset.drop(columns='charges', axis=1)
Y = insurance_dataset['charges']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

regressor = LinearRegression()
regressor.fit(X_train, Y_train)
prediction = regressor.predict(X_test)

col1, col2 = st.columns(2, gap='large')

with col1:

    age = st.text_input(label='Age')

    gender = st.selectbox(label='Gender', options=['Male', 'Female'])
    g_dict = {'Male': 1, 'Female': 0}

    bmi = st.text_input(label='BMI')

with col2:
    children = st.text_input(label='Children')

    smoker = st.selectbox(label='Smoker', options=['Yes', 'No'])
    smoking_dict = {'Yes': 1, 'No': 0}

    region = st.selectbox(label='Region', options=['southeast', 'southwest', 'northeast', 'northwest'])
    r_dict = {'southeast':0,'southwest':1,'northeast':2,'northwest':3}

st.write('')
st.write('')
col1, col2 = st.columns([0.438, 0.562])

with col2:
    submit = st.button(label='Submit')

st.write('')

if submit:
    try:
        # Input Validation and Conversion to Numeric
        age = float(age)
        children = float(children)
        bmi = float(bmi)

        # Convert user data to numeric values
        user_data = np.array([[age,g_dict[gender] ,bmi , children, smoking_dict[smoker], r_dict[region]]])

        input_data_reshaped = user_data.reshape(1,-1)

        test_result = regressor.predict(input_data_reshaped)

        st.write(f"Debug - Input Data: {user_data}")
        st.write(f"Debug - Reshaped Input Data: {input_data_reshaped}")
        

        # Print debug information
        st.write(f"Debug - Predicted Value: {test_result[0]}")

        if test_result[0] >= 0:
            col1, col2, col3 = st.columns([0.33,0.75, 0.35])
            with col2:
                st.success(f'The Insurance Cost in USD :  {test_result[0]}')
        else:
            col1, col2, col3 = st.columns([0.215, 0.57, 0.215])
            with col2:
                st.error('Wrong Data')

    except Exception as e:
        st.error(f'An error occurred: {e}')

