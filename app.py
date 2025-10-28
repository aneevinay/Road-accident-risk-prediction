import streamlit as st
import pandas as pd
import joblib

encoder = joblib.load("encoder.pkl")
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
    "curvature": curvature,
    "speed_limit": speed_limit,
    "lighting": lighting,
    "weather": weather,
    "road_signs_present": road_signs_present,
    "public_road": public_road,
    "time_of_day": time_of_day,
    "holiday": holiday,
    "school_season": school_season,
    "num_reported_accidents": num_reported_accidents
}])

expected_cols = encoder.feature_names_in_ if hasattr(encoder, 'feature_names_in_') else input_data.columns

encoded_data = encoder.transform(input_df)

if not isinstance(encoded_data, pd.DataFrame):
    encoded_data = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(expected_cols))

y_pred = model.predict(encoded_data)

st.success(f"Predicted accident risk: {y_pred[0]:.4f}")
