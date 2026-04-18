import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load the model and scaler
model = pickle.load(open('housing_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

st.title("🏡 California House Price Predictor")
st.write("Enter the details below to estimate the median house value.")

# Create input fields based on the California dataset features
# Use columns to make it look cleaner
col1, col2 = st.columns(2)

with col1:
    med_inc = st.number_input("Median Income (in $10k)", value=3.5)
    house_age = st.number_input("Median House Age", value=20)
    ave_rooms = st.number_input("Average Rooms", value=5)
    ave_bedrms = st.number_input("Average Bedrooms", value=1)

with col2:
    population = st.number_input("Area Population", value=1000)
    ave_occup = st.number_input("Average Occupancy", value=3)
    latitude = st.number_input("Latitude", value=34.0)
    longitude = st.number_input("Longitude", value=-118.0)

if st.button("Predict Price"):
    # 1. Arrange inputs into a 2D array
    features = np.array([[med_inc, house_age, ave_rooms, ave_bedrms, 
                          population, ave_occup, latitude, longitude]])
    
    # 2. SCALE the features using the saved scaler
    features_scaled = scaler.transform(features)
    
    # 3. Predict
    prediction = model.predict(features_scaled)
    
    # The California dataset target is in $100,000s
    final_price = prediction[0] * 100000
    
    st.success(f"Estimated House Value: ${final_price:,.2f}")