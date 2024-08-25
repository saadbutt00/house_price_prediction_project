import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Define file paths
model_path = 'F:/MyProject/house price prediction/lr_model.pkl'
encoder_path = 'F:/MyProject/house price prediction/label_encoders.pkl'
scaler_path = 'F:/MyProject/house price prediction/scaler.pkl'

# Load the trained model, label encoders, and scaler
with open(model_path, 'rb') as model_file:
    lr_model = pickle.load(model_file)

with open(encoder_path, 'rb') as encoder_file:
    label_encoders = pickle.load(encoder_file)

with open(scaler_path, 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# Set dark theme
st.set_page_config(page_title="House Price Prediction", page_icon="🏠", layout="wide", initial_sidebar_state="expanded")

# Title of the app
st.title('House Price Prediction App 🏠')

# Create an empty DataFrame to use for column names and default values
data = pd.DataFrame(columns=['sqft_living', 'sqft_lot', 'waterfront', 'view', 'city', 'country'])

# Define default values for features
default_features = {
    'sqft_living': int(data['sqft_living'].median()) if not data.empty else 2000,
    'sqft_lot': int(data['sqft_lot'].median()) if not data.empty else 5000,
    'waterfront': int(data['waterfront'].mode().iloc[0]) if not data.empty else 0,
    'view': int(data['view'].mode().iloc[0]) if not data.empty else 0,
    'city': data['city'].mode().iloc[0] if not data.empty else 'Seattle',
    'country': data['country'].mode().iloc[0] if not data.empty else 'USA'
}

# Create a list of cities with numbers
city_names = label_encoders['city'].classes_  # Use classes_ attribute to get unique city names
city_list = list(enumerate(city_names, start=1))
city_options = [f"{i}. {city}" for i, city in city_list]

# Sidebar for user input
st.sidebar.header('User Input Features')

# Display city options and get user input
city_choice = st.sidebar.selectbox('Select a City', options=city_options)
selected_city_index = int(city_choice.split('.')[0]) - 1
city = label_encoders['city'].transform([city_names[selected_city_index]])[0]

# Input features
sqft_living = st.sidebar.number_input('Number of Area (in sq. foot)', min_value=int(data['sqft_living'].min()) if not data.empty else 1000, max_value=int(data['sqft_living'].max()) if not data.empty else 10000, value=default_features['sqft_living'])
sqft_lot = st.sidebar.number_input('Lot Size (in sq. foot)', min_value=int(data['sqft_lot'].min()) if not data.empty else 1000, max_value=int(data['sqft_lot'].max()) if not data.empty else 10000, value=default_features['sqft_lot'])
waterfront = st.sidebar.number_input('Waterfront (0=No, 1=Yes)', min_value=0, max_value=1, value=default_features['waterfront'])
view = st.sidebar.number_input('View (0=No, 1=Yes)', min_value=0, max_value=4, value=default_features['view'])

# Set country as 'USA'
st.sidebar.write('Country: USA')
country = label_encoders['country'].transform(['USA'])[0]  # Encode 'USA' to match the trained model's expected input

# Prepare the input data for prediction
user_input = pd.DataFrame([[sqft_living, sqft_lot, waterfront, view, city, country]],
                          columns=['sqft_living', 'sqft_lot', 'waterfront', 'view', 'city', 'country'])

# Scale the input data
user_input_scaled = scaler.transform(user_input)

# Predict the price
predicted_log_price = lr_model.predict(user_input_scaled)
predicted_price = int(np.exp(predicted_log_price)[0])

# Display the predicted price
st.write(f"### Predicted Price")
st.write(f"💲 {predicted_price:,.2f}")

# streamlit run "F:\MyProject\house price prediction\app.py"
