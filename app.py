import streamlit as st
import pandas as pd
import numpy as np
import pycaret.regression
import os

# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="Bike Rental Demand Predictor",
    layout="centered", # Use 'wide' if you want more space
    initial_sidebar_state="auto"
)

# --- Define Model File Path ---
MODEL_NAME = 'final_bike_demand_model' # This should match the name you used in exp.save_model()
MODEL_FILE_PATH = f'{MODEL_NAME}.pkl'

# --- Load the PyCaret Model with Caching ---
@st.cache_resource # Cache the model loading for performance
def load_pycaret_model(model_path_without_extension):
    """Loads the trained PyCaret regression model."""
    full_model_file_path = f"{model_path_without_extension}.pkl"
    if not os.path.exists(full_model_file_path):
        st.error(f"Model file '{full_model_file_path}' not found. Please ensure the model is saved correctly and is accessible in the same directory as app.py.")
        return None
    try:
        loaded_model = pycaret.regression.load_model(model_path_without_extension)
        st.success("Prediction model loaded successfully!")
        return loaded_model
    except Exception as e:
        st.error(f"An error occurred while loading the model from '{full_model_file_path}': {e}")
        st.exception(e) # Display the full error for debugging
        return None

# Attempt to load the model
final_model = load_pycaret_model(MODEL_NAME)

# --- App Title and Overview ---
st.title("🚴‍♀️ Capital Bikeshare Demand Predictor")
st.markdown("""
    Use this app to predict the total number of bike rentals
    from the Capital Bikeshare system in Washington D.C.
    based on various environmental and temporal factors.
""")
st.markdown("---") # Separator


if final_model is not None: # Proceed only if the model is loaded successfully
    # --- User Input Section ---
    st.header("Provide Input Data")

    # Using columns for a cleaner layout
    input_col1, input_col2 = st.columns(2)

    with input_col1:
        st.subheader("Date and Time")
        date_input = st.date_input("Select a Date")
        time_input = st.time_input("Select a Time")

        # Combine Date and Time
        datetime_combined = pd.to_datetime(str(date_input) + ' ' + str(time_input))
        st.write(f"You selected: **{datetime_combined.strftime('%Y-%m-%d %H:%M:%S')}**")

        st.subheader("Categorical Factors")
        # Mapping for display vs value
        season_map = {1: '1: Winter', 2: '2: Spring', 3: '3: Summer', 4: '4: Fall'}
        selected_season = st.selectbox("Season", options=list(season_map.keys()), format_func=lambda x: season_map[x])

        holiday_map = {0: '0: No (Regular)', 1: '1: Yes (Public Holiday)'}
        selected_holiday = st.selectbox("Is it a Public Holiday?", options=list(holiday_map.keys()), format_func=lambda x: holiday_map[x])

        workingday_map = {0: '0: Weekend or Holiday', 1: '1: Working Day'}
        selected_workingday = st.selectbox("Day Type", options=list(workingday_map.keys()), format_func=lambda x: workingday_map[x])

        weather_map = {1: '1: Clear/Few Clouds', 2: '2: Mist/Cloudy', 3: '3: Light Rain/Snow', 4: '4: Heavy Rain/Fog'}
        selected_weather = st.selectbox("Weather Situation", options=list(weather_map.keys()), format_func=lambda x: weather_map[x])


    with input_col2:
        st.subheader("Numerical Measurements")
        # Sliders for numerical features with tooltips
        selected_temp = st.slider("Temperature (°C)", min_value=0.0, max_value=45.0, value=20.0, step=0.1, help="Air temperature in Celsius.")
        selected_humidity = st.slider("Humidity (%)", min_value=0, max_value=100, value=50, step=1, help="Relative humidity percentage.")
        selected_windspeed = st.slider("Windspeed (km/h)", min_value=0.0, max_value=60.0, value=15.0, step=0.1, help="Wind speed.")


    # --- Create DataFrame for Prediction ---
    # Column names must match the original training data columns
    input_df = pd.DataFrame({
        'datetime': [datetime_combined],
        'season': [selected_season],
        'holiday': [selected_holiday],
        'workingday': [selected_workingday],
        'weather': [selected_weather],
        'temp': [selected_temp],
        # 'atemp' was dropped in your preprocessing, do not include here
        'humidity': [selected_humidity],
        'windspeed': [selected_windspeed]
        # 'casual', 'registered', 'count' are NOT input features
    })

    # Ensure datetime column is in the correct format
    input_df['datetime'] = pd.to_datetime(input_df['datetime'])

    st.markdown("---") # Separator

    # --- Prediction Button and Output ---
    if st.button("Get Prediction", help="Click to see the predicted bike demand."):
        st.subheader("Prediction Result")
        try:
            # Use PyCaret's predict_model to apply the saved pipeline and model
            # This function automatically handles all preprocessing steps defined during PyCaret setup
            prediction_result = pycaret.regression.predict_model(final_model, data=input_df)

            # Extract the prediction
            # PyCaret's predict_model typically returns a DataFrame with a 'prediction_label' column
            if 'prediction_label' in prediction_result.columns:
                predicted_count = prediction_result['prediction_label'].iloc[0]

                # Display the result, ensuring it's non-negative and an integer
                final_predicted_count = max(0, int(round(predicted_count)))

                st.success(f"Predicted Total Bike Rentals: **{final_predicted_count}**")
                st.write("*(This prediction estimates the combined rentals for both casual and registered users)*")
            else:
                st.error("Prediction output format unexpected from PyCaret. Cannot find the 'prediction_label' column.")
                st.write("Output DataFrame columns:", prediction_result.columns.tolist())


        except Exception as e:
            st.error("An error occurred during the prediction process.")
            st.exception(e) # Show full error details in logs and Streamlit app

# --- Footer/About (Optional) ---
st.markdown("---")
st.markdown("App based on the Capital Bikeshare Demand Prediction project.")
