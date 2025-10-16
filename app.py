import streamlit as st
import pandas as pd

st.title("🚗 Road Accident Risk Prediction")

st.markdown("""
This app predicts the likelihood of road accidents using XGBoost.
""")

# Input fields
road_type = st.selectbox("Road Type", ["highway", "urban", "rural"])
speed_limit = st.selectbox("Speed Limit (km/h)", [25 , 35, 45, 60, 70])
weather = st.selectbox("Weather", ["clear", "rainy", "foggy"])
lighting =st.selectbox("Lighting",["daylight","dim","night"])

# Display user input
st.write("### Input Summary")
st.write({
    "Road Type": road_type,
    "Speed Limit": speed_limit,
    "Weather": weather,
    "Lighting": lighting
})

# Dummy prediction (replace later with model)
if st.button("Predict Risk"):
    st.success("Predicted Accident Risk: 0.072 (Example Output)")
