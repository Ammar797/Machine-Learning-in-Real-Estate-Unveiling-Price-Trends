import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load the dataset from JSON file
df = pd.read_json('data.json')

# Define the predict_price function
def predict_price(location, society, total_sqft, bath, balcony, BHK, model, X):
    loc_index = np.where(X.columns == location)[0][0]
    soc_index = np.where(X.columns == society)[0][0]

    x = np.zeros(len(X.columns))
    x[0] = total_sqft
    x[1] = bath
    x[2] = balcony
    x[3] = BHK
    if loc_index >= 0:
        x[loc_index] = 1
    if soc_index >= 0:
        x[soc_index] = 1

    return model.predict([x])[0]

# Function to format the predicted price
def format_price(price):
    if price >= 1e7:
        return f'Rs {price / 1e7:.2f} Crores'
    else:
        return f'Rs {price / 1e5:.2f} Lakhs'

# Streamlit UI
st.title('House Price Prediction')

# Input fields
location = st.text_input('Location (Add prefix "l_" before location name):')
society = st.text_input('Society (Add prefix "s_" before society name):')
total_sqft = st.number_input('Total Square Feet Area:', min_value=0)
bath = st.number_input('Number of Bathrooms:', min_value=0)
balcony = st.number_input('Number of Balconies:', min_value=0)
BHK = st.number_input('Number of Bedrooms (BHK):', min_value=0)

# Assuming 'linear_reg' is the trained Linear Regression model
# Adjust this to load your trained model
with open('linear_regression_model.pkl', 'rb') as f:
    linear_reg = pickle.load(f)

# Define X as the feature matrix from your DataFrame
X = df.drop(columns=['price', 'price_per_sqft'])

# Predict button
if st.button('Predict Price'):
    # Call predict_price function
    predicted_price = predict_price(location, society, total_sqft, bath, balcony, BHK, linear_reg, X)
    formatted_price = format_price(predicted_price)
    st.success(f'Predicted Price: {formatted_price}')
