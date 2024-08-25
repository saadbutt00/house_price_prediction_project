import streamlit as st
import pandas as pd
import numpy as np
import pickle as pkl
import os

# Set dark theme
st.set_page_config(page_title="House Price Prediction", page_icon="🏠", layout="wide", initial_sidebar_state="expanded")

# Title of the app
st.title('House Price Prediction App 🏠')

# Load the pre-trained model, label encoders, and scaler using pickle
with open(os.path.join('F:/MyProject/house price prediction/', 'lr_model.pkl'), 'rb') as model_file:
    model = pkl.load(model_file)

with open(os.path.join('F:/MyProject/house price prediction/', 'le_city.pkl'), 'rb') as le_city_file:
    le_city = pkl.load(le_city_file)

with open(os.path.join('F:/MyProject/house price prediction/', 'le_country.pkl'), 'rb') as le_country_file:
    le_country = pkl.load(le_country_file)

with open(os.path.join('F:/MyProject/house price prediction/', 'scaler.pkl'), 'rb') as scaler_file:
    scaler = pkl.load(scaler_file)

# Sidebar for user input
st.sidebar.header('User Input Features')

# Define default values for features
default_features = {
    'sqft_living': 1500,
    'sqft_lot': 5000,
    'waterfront': 0,
    'view': 0,
    'city': 0,
}

# Create a list of city options based on the LabelEncoder
city_names = le_city.inverse_transform(range(len(le_city.classes_)))
city_list = list(enumerate(city_names, start=1))
city_options = [f"{i}. {city}" for i, city in city_list]

# Display city options and get user input
city_choice = st.sidebar.selectbox('Select a City', options=city_options)
selected_city_index = int(city_choice.split('.')[0]) - 1
city = le_city.transform([city_names[selected_city_index]])[0]

# Input features
sqft_living = st.sidebar.number_input('Number of Area (in sq. foot)', min_value=500, max_value=10000, value=default_features['sqft_living'])
sqft_lot = st.sidebar.number_input('Lot Size (in sq. foot)', min_value=1000, max_value=50000, value=default_features['sqft_lot'])
waterfront = st.sidebar.number_input('Waterfront (0=No, 1=Yes)', min_value=0, max_value=1, value=default_features['waterfront'])
view = st.sidebar.number_input('View (0=No, 1=Yes)', min_value=0, max_value=4, value=default_features['view'])

# Set country as 'USA'
st.sidebar.write('Country : USA')
country = le_country.transform(['USA'])[0]

# Prepare the input data for prediction
user_input = pd.DataFrame([[sqft_living, sqft_lot, waterfront, view, city, country]],
                          columns=['sqft_living', 'sqft_lot', 'waterfront', 'view', 'city', 'country'])

# Scale the input data
user_input_scaled = scaler.transform(user_input)

# Predict the price
predicted_log_price = model.predict(user_input_scaled)
predicted_price = int(np.exp(predicted_log_price)[0])

# Display the predicted price
st.write(f"### Predicted Price")
st.write(f"💲 {predicted_price:,.2f}")
