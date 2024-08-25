import streamlit as st
import pandas as pd
import numpy as np
import pickle as pkl

# Set dark theme
st.set_page_config(page_title="House Price Prediction", page_icon="🏠", layout="wide", initial_sidebar_state="expanded")

# Title of the app
st.title('House Price Prediction App 🏠')

lrm = 'lr_model.pkl'
lc = 'le_city.pkl'
lec = 'le_country.pkl'
s = 'scaler.pkl'

# Loading the model, encoders, and scaler
with open(lrm, 'rb') as model_file:
    loaded_model = pkl.load(model_file)

with open(lc, 'rb') as le_city_file:
    loaded_le_city = pkl.load(le_city_file)

with open(lec, 'rb') as le_country_file:
    loaded_le_country = pkl.load(le_country_file)

with open(s, 'rb') as scaler_file:
    loaded_scaler = pkl.load(scaler_file)

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
city_names = loaded_le_city.inverse_transform(range(len(loaded_le_city.classes_)))
city_list = list(enumerate(city_names, start=1))
city_options = [f"{i}. {city}" for i, city in city_list]

# Display city options and get user input
city_choice = st.sidebar.selectbox('Select a City', options=city_options)
selected_city_index = int(city_choice.split('.')[0]) - 1
city = loaded_le_city.transform([city_names[selected_city_index]])[0]

# Input features
sqft_living = st.sidebar.number_input('Number of Area (in sq. foot)', min_value=500, max_value=10000, value=default_features['sqft_living'])
sqft_lot = st.sidebar.number_input('Lot Size (in sq. foot)', min_value=1000, max_value=50000, value=default_features['sqft_lot'])
waterfront = st.sidebar.number_input('Waterfront (0=No, 1=Yes)', min_value=0, max_value=1, value=default_features['waterfront'])
view = st.sidebar.number_input('View (0=No, 1=Yes)', min_value=0, max_value=4, value=default_features['view'])

# Set country as 'USA'
st.sidebar.write('Country : USA')
country = loaded_le_country.transform(['USA'])[0]

# Prepare the input data for prediction
user_input = pd.DataFrame([[sqft_living, sqft_lot, waterfront, view, city, country]],
                          columns=['sqft_living', 'sqft_lot', 'waterfront', 'view', 'city', 'country'])

# Scale the input data
user_input_scaled = loaded_scaler.transform(user_input)

# Predict the price
predicted_log_price = loaded_model.predict(user_input_scaled)
predicted_price = int(np.exp(predicted_log_price)[0])

# Display the predicted price
st.write(f"### Predicted Price")
st.write(f"💲 {predicted_price:,.2f}")
