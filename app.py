import streamlit as st
import numpy as np
import joblib

# Load trained model and scaler
model = joblib.load('linear_model.pkl')
scaler = joblib.load('scaler.pkl')

# Streamlit app interface
st.set_page_config(page_title="Paris Housing Price Predictor")
st.title("🏠 Paris Housing Price Predictor")
st.markdown("Enter property features below to estimate the price of a Parisian home.")

# Input fields for features
squareMeters = st.number_input("Total Area (square meters)", min_value=10.0, value=100.0, step=1.0)
numberOfRooms = st.number_input("Number of Rooms", min_value=1, value=3, step=1)
floors = st.number_input("Number of Floors", min_value=1, value=1, step=1)
basement = st.number_input("Basement Area (m²)", min_value=0.0, value=20.0, step=1.0)
attic = st.number_input("Attic Area (m²)", min_value=0.0, value=15.0, step=1.0)
garage = st.number_input("Garage Size (cars)", min_value=0, value=1, step=1)
numPrevOwners = st.number_input("Number of Previous Owners", min_value=0, value=1, step=1)
houseAge = st.number_input("House Age (years)", min_value=0, value=5, step=1)
amenityScore = st.slider("Amenity Score (yard, pool, storage, etc.)", 0, 6, 3)

# Predict button
if st.button("Predict Price"):
    # Arrange features in the same order as model training
    input_array = np.array([[squareMeters, numberOfRooms, floors,
                             basement, attic, garage,
                             numPrevOwners, houseAge, amenityScore]])

    # Scale inputs
    input_scaled = scaler.transform(input_array)

    # Make prediction
    predicted_price = model.predict(input_scaled)[0]

    # Display prediction
    st.success(f"💶 Estimated Property Price: €{predicted_price:,.2f}")
