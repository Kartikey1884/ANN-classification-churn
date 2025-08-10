import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler,LabelEncoder
import tensorflow as tf
import pickle

# Load the pre-trained model
model = tf.keras.models.load_model('model.keras')

with open('ohe_geo.pkl', 'rb') as f:
    ohe_geo = pickle.load(f)

with open('label_encoder_gender.pkl', 'rb') as f:
    label_encoder_gender = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

##streamlit app
st.title("Bank Customer Churn Prediction")

# Input fields for user data

geography = st.selectbox('Geography',ohe_geo.categories_[0])
gender = st.selectbox('Gender',label_encoder_gender.classes_)
age = st.slider('Age', min_value=18, max_value=100, value=30)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score', min_value=300, max_value=850, value=600)
estimated_salary = st.number_input('Estimated Salary', min_value=0, max_value=1000000, value=50000)
tenure = st.slider('Tenure', min_value=0, max_value=10, value=5)
number_of_products = st.slider('Number of Products', min_value=1, max_value=4, value=2)
has_cr_card = st.selectbox('Has Credit Card', [0,1])
is_active_member = st.selectbox('Is Active Member', [0,1])

# Create a DataFrame from the input data
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender':[label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [number_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary],
})

# One-hot encode the Geography feature
geo_encoded = ohe_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=ohe_geo.get_feature_names_out(['Geography']))

# Concatenate the encoded Geography feature with the input data
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

# Scale the input data
input_scaled = scaler.transform(input_data)

# Predict the output
predictions = model.predict(input_scaled)
prediction_probability = predictions[0][0]  # Assuming binary classification, get the first prediction
if prediction_probability > 0.5:
    prediction = "The customer is likely to leave the bank."    
else:
    prediction = "The customer is likely to stay with the bank."

# Display the prediction
st.subheader("Prediction Result")
st.write(f"Prediction Probability: {prediction_probability:.2f}")
st.write(prediction)