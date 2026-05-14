import streamlit as st
import pickle
import numpy as np
import pandas as pd

@st.cache_resource
def load_assets():
    try:
        with open('housing_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        return model, scaler
    except FileNotFoundError as e:
        st.error(f"Error: Could not find model or scaler file. {e}")
        return None, None

model, scaler = load_assets()

st.title("California House Price Predictor")
st.write("Enter the details below to estimate the median house value.")

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
    if model is not None and scaler is not None:
        # 1. Arrange inputs into a list
        input_data = [[med_inc, house_age, ave_rooms, ave_bedrms, 
                       population, ave_occup, latitude, longitude]]
        
        # 2. Get the exact names the scaler was trained on
        try:
            # Most modern sklearn scalers store this in 'feature_names_in_'
            feature_names = scaler.feature_names_in_
        except AttributeError:
            # If for some reason the names aren't there, we'll fallback to a numpy array
            # and just ignore the warning.
            feature_names = None

        if feature_names is not None:
            features_df = pd.DataFrame(input_data, columns=feature_names)
            features_scaled = scaler.transform(features_df)
        else:
            # Fallback to numpy if names are missing
            features_scaled = scaler.transform(np.array(input_data))
        
        # 3. Predict
        prediction = model.predict(features_scaled)
        
        final_price = prediction[0] * 100000
        st.success(f"Estimated House Value: ${final_price:,.2f}")