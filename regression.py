import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import tensorflow
import pickle

## Load the trained model
model = tensorflow.keras.models.load_model('regression_model.h5')

## load the encoders and scaler
with open('label_encoder_reg_gender.pkl','rb') as file:
    label_encoder_reg_gender = pickle.load(file)

with open('onehot_encoder_reg_geo.pkl','rb') as file:
    onehot_encoder_reg_geo = pickle.load(file)

with open('scaler_reg.pkl','rb') as file:
    scaler_reg = pickle.load(file)

## streamlit app
st.title('Salary Prediction')
# User input
geography = st.selectbox('Geography', onehot_encoder_reg_geo.categories_[0])
gender = st.selectbox('Gender', label_encoder_reg_gender.classes_)
age = st.slider('Age', 18, 92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
exited = st.selectbox('Exited',[0,1])
tenure = st.slider('Tenure',0,10)
num_of_products = st.slider('Number Of Products',1, 4)
has_cr_card = st.selectbox('Has Credit Card',[0,1])
is_active_member = st.selectbox('Is Active Member',[0,1])

# Example input data
input_data = pd.DataFrame({
    'CreditScore':[credit_score],
    'Gender':[label_encoder_reg_gender.transform([gender])[0]],
    'Age':[age],
    'Tenure':[age],
    'Balance':[balance],
    'NumOfProducts':[num_of_products],
    'HasCrCard':[has_cr_card],
    'IsActiveMember':[is_active_member],
    'Exited':[exited]
})

# One_hot encode geography
geo_encoded = onehot_encoder_reg_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_reg_geo.get_feature_names_out(['Geography']))

# combine one hot encoded columns with input data
input_data = pd.concat([input_data.reset_index(drop=True),geo_encoded_df],axis=1)

#scale the input data
input_data_scaled = scaler_reg.transform(input_data)

# Predict Churn
prediction = model.predict(input_data_scaled)
prediction_salary = prediction[0][0]

st.write(f"Predicted Estimated Salary : ${prediction_salary:.2f}")
