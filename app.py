import streamlit as st
import numpy as np
import joblib
import pandas as pd

# Load model and pipeline
model = joblib.load("model.pkl")
pipeline = joblib.load("pipeline.pkl")

# UI Inputs
st.title("Used Car Price Predictor")

brand = st.selectbox("Brand", ['Maruti', 'Hyundai', 'Honda', 'Toyota', 'Ford'])  # update with your actual categories
vehicle_age = st.selectbox("Vehicle Age", list(range(0, 26)))
  # or numeric if applicable
km_driven = st.number_input("Kilometers Driven", min_value=0)
seller_type = st.selectbox("Seller Type", ['Individual', 'Dealer', 'Trustmark Dealer'])
fuel_type = st.selectbox("Fuel Type", ['Petrol', 'Diesel', 'CNG', 'LPG', 'Electric'])
transmission_type = st.selectbox("Transmission", ['Manual', 'Automatic'])
mileage = st.number_input("Mileage (e.g. 20.4)")
engine = st.number_input("Engine Capacity (e.g. 1197)")
max_power = st.number_input("Max Power (e.g. 82)")
seats = st.selectbox("Seats", [2, 4, 5, 6, 7, 8, 9])

# Submit
if st.button("Predict Price"):
    input_dict = {
        'brand': brand,
        'vehicle_age': vehicle_age,
        'km_driven': km_driven,
        'seller_type': seller_type,
        'fuel_type': fuel_type,
        'transmission_type': transmission_type,
        'mileage': mileage,
        'engine': engine,
        'max_power': max_power,
        'seats': seats
    }

    input_df = pd.DataFrame([input_dict])
    transformed_input = pipeline.transform(input_df)
    prediction = model.predict(transformed_input)
    final_price = np.expm1(prediction[0])  # reverse log1p

    st.success(f"Estimated Selling Price: ₹{final_price:,.0f}")
