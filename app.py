import streamlit as st
import pandas as pd
import joblib

model = joblib.load("model.pkl")
    
st.title("🚗 Road Accident Risk Prediction")

st.markdown("""
This app predicts the likelihood of road accidents using XGBoost.
""")

# Input fields
road_type = st.selectbox("Road Type", ["highway", "urban", "rural"])
speed_limit = st.selectbox("Speed Limit (km/h)", [25 , 35, 45, 60, 70])
weather = st.selectbox("Weather", ["clear", "rainy", "foggy"])
lighting =st.selectbox("Lighting",["daylight","dim","night"])
num_lanes=st.selectbox("num_lanes",[1,2,3,4])
road_signs_present=st.selectbox("road_signs_present",[0,1])
public_road=st.selectbox("public_road",[0,1])
holiday=st.selectbox("holiday",[0,1])
school_season=st.selectbox("school_season",[0,1])
num_reported_accidents=st.selectbox("num_reported_accidents",[0,1,2,3,4,5,6,7])
time_of_day=st.selectbox("time_of_day",["afternoon","evening","morning"])
curvature = st.slider("Select road curvature",
                      min_value=0.0,max_value=1.0,
                      value=0.5, step=0.01,
                      help="Higher curvature means sharper turns."
                     )


# Display user input
st.write("### Input Summary")
st.write({
    "Road Type": road_type,
    "Speed Limit": speed_limit,
    "Weather": weather,
    "Lighting": lighting,
    "Number of Lanes": num_lanes,
    "Road Signs Present": road_signs_present,
    "Public Road": public_road,
    "Holiday": holiday,
    "School Season": school_season,
    "Reported Accidents": num_reported_accidents,
    "Time of Day": time_of_day,
    "Curvature": curvature
})

input_df = pd.DataFrame([{
    "road_type": road_type,
    "num_lanes": num_lanes,
    "speed_limit": speed_limit,
    "lighting": lighting,
    "weather": weather,
    "road_signs_present": road_signs_present,
    "public_road": public_road,
    "holiday": holiday,
    "school_season": school_season,
    "num_reported_accidents": num_reported_accidents,
    "time_of_day": time_of_day,
    "curvature": curvature
}])

if st.button("Predict Accident Risk"):
    try:
        prediction = model.predict(input_df)[0]
        st.success(f"🚧 Predicted Accident Risk: **{prediction:.4f}**")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
