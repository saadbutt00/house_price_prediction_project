import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

# Set dark theme
st.set_page_config(page_title="House Price Prediction", page_icon="🏠", layout="wide", initial_sidebar_state="expanded")

# Title of the app
st.title('House Price Prediction App 🏠')

# Load the dataset
df = pd.read_csv('F:/MyProject/data.csv')

# Update the dataset with the new attribute 'sqft_lot'
df = df.drop(['sqft_above', 'sqft_basement', 'statezip', 'street', 'date', 'yr_built', 'yr_renovated', 'bedrooms', 'bathrooms', 'condition', 'floors'], axis=1)

# Handle price values
df['price'] = np.where(df['price'] <= 0, 1, df['price'])
df['log_price'] = np.log(df['price'])
df.drop('price', axis=1, inplace=True)

# Encode categorical variables
le_city = LabelEncoder()
le_country = LabelEncoder()
df['city'] = le_city.fit_transform(df['city'])
df['country'] = le_country.fit_transform(df['country'])

# Prepare the data for model training
x = df.drop(['log_price'], axis=1)
y = df['log_price']

# Split the data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Standardize the data
scaler = StandardScaler()
x_tr_scaled = scaler.fit_transform(x_train)
x_ts_scaled = scaler.transform(x_test)

# Train the model
model = LinearRegression()
model.fit(x_tr_scaled, y_train)

# Save the model and encoders
with open('model.pkl', 'wb') as file:
    pickle.dump(model, file)
with open('scaler.pkl', 'wb') as file:
    pickle.dump(scaler, file)
with open('le_city.pkl', 'wb') as file:
    pickle.dump(le_city, file)
with open('le_country.pkl', 'wb') as file:
    pickle.dump(le_country, file)

# Load the saved model and encoders
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)
with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)
with open('le_city.pkl', 'rb') as file:
    le_city = pickle.load(file)
with open('le_country.pkl', 'rb') as file:
    le_country = pickle.load(file)

# Sidebar for user input
st.sidebar.header('User Input Features')

# Define default values for features
default_features = {
    'sqft_living': int(df['sqft_living'].median()),
    'sqft_lot': int(df['sqft_lot'].median()),
    'waterfront': int(df['waterfront'].mode().iloc[0]),
    'view': int(df['view'].mode().iloc[0]),
    'city': df['city'].mode().iloc[0],
}

# Create a list of cities with numbers
city_names = le_city.inverse_transform(df['city'].unique())
city_list = list(enumerate(city_names, start=1))
city_options = [f"{i}. {city}" for i, city in city_list]

# Display city options and get user input
city_choice = st.sidebar.selectbox('Select a City', options=city_options)
selected_city_index = int(city_choice.split('.')[0]) - 1
city = le_city.transform([city_names[selected_city_index]])[0]

# Input features
sqft_living = st.sidebar.number_input('Number of Area (in sq. foot)', min_value=int(df['sqft_living'].min()), max_value=int(df['sqft_living'].max()), value=default_features['sqft_living'])
sqft_lot = st.sidebar.number_input('Lot Size (in sq. foot)', min_value=int(df['sqft_lot'].min()), max_value=int(df['sqft_lot'].max()), value=default_features['sqft_lot'])
waterfront = st.sidebar.number_input('Waterfront (0=No, 1=Yes)', min_value=0, max_value=1, value=default_features['waterfront'])
view = st.sidebar.number_input('View (0=No, 1=Yes)', min_value=0, max_value=4, value=default_features['view'])

# Set country as 'USA'
st.sidebar.write('Country : USA')
country = le_country.transform(['USA'])[0]  # Encode 'USA' to match the trained model's expected input

# Prepare the input data for prediction
user_input = pd.DataFrame([[sqft_living, sqft_lot, waterfront, view, city, country]],
                          columns=x.columns)

# Scale the input data
user_input_scaled = scaler.transform(user_input)

# Predict the price
predicted_log_price = model.predict(user_input_scaled)
predicted_price = int(np.exp(predicted_log_price)[0])

# Display the predicted price
st.write(f"### Predicted Price")
st.write(f"💲 {predicted_price:,.2f}")
