import streamlit as st
import pandas as pd
import numpy as np
import pycaret.regression
import os
from scipy.stats.mstats import winsorize # Keep this for the custom function


# --- Define Custom Function Used in Pipeline ---
# This function MUST be defined in app.py so the loaded model can find it
def winsorize_series_robust(df_or_series, limits=(0.01, 0.01)):
    if isinstance(df_or_series, pd.DataFrame):
        series_to_winsorize = df_or_series.iloc[:, 0].copy()
    else:
        series_to_winsorize = df_or_series.copy()

    # Handle cases where the series might be empty or have non-numeric data before winsorize
    if series_to_winsorize.empty or not pd.api.types.is_numeric_dtype(series_to_winsorize):
            # Return the original series or handle appropriately if non-numeric
            return series_to_winsorize

    winsorized_array = winsorize(series_to_winsorize, limits=limits)
    return pd.Series(winsorized_array.flatten(), name=series_to_winsorize.name) # Return as Series


# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="Bike Rental Demand Predictor",
    layout="centered", # 'centered' or 'wide'
    initial_sidebar_state="auto"
)

# --- Define Model File Path ---
MODEL_NAME = 'final_bike_demand_model' # This should match the name you used in exp.save_model()
MODEL_FILE_PATH = f'{MODEL_NAME}.pkl'

# --- Load the PyCaret Model (Simple Load) ---
# Since the loaded object is the Pipeline itself, we can simplify the loading function
@st.cache_resource # Cache the model loading for performance
def load_pycaret_pipeline(model_path_without_extension):
    """Loads the trained PyCaret Pipeline."""
    full_model_file_path = f"{model_path_without_extension}.pkl"
    if not os.path.exists(full_model_file_path):
        st.error(f"Model file '{full_model_file_path}' not found. Please ensure the model is saved correctly and is accessible.")
        return None
    try:
        # Load the model - based on your output, this directly loads the Pipeline
        loaded_pipeline = pycaret.regression.load_model(model_path_without_extension)
        st.success("Prediction pipeline loaded successfully!")
        return loaded_pipeline
    except Exception as e:
        st.error(f"An error occurred while loading the model from '{full_model_file_path}': {e}")
        st.exception(e) # Display the full error for debugging
        return None


# Attempt to load the PyCaret Pipeline
final_pipeline = load_pycaret_pipeline(MODEL_NAME)

# --- App Title and Overview ---
st.title("🚴‍♀️ Capital Bikeshare Demand Predictor")
st.markdown("""
    Use this app to predict the total number of bike rentals
    from the Capital Bikeshare system in Washington D.C.
    based on various environmental and temporal factors.
""")
st.markdown("---") # Separator


if final_pipeline is not None: # Proceed only if the pipeline is loaded successfully
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
    input_df_raw = pd.DataFrame({
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
    input_df_raw['datetime'] = pd.to_datetime(input_df_raw['datetime'])

    # --- FEATURE ENGINEERING: ADD MISSING TIME-BASED FEATURES ---
    # These columns are expected by your PyCaret pipeline
    # Existing features from previous error
    input_df_raw['day'] = input_df_raw['datetime'].dt.day
    input_df_raw['dayofyear'] = input_df_raw['datetime'].dt.dayofyear

    # Cyclical features
    input_df_raw['hour_sin'] = np.sin(2 * np.pi * input_df_raw['datetime'].dt.hour / 24.0)
    input_df_raw['hour_cos'] = np.cos(2 * np.pi * input_df_raw['datetime'].dt.hour / 24.0)

    input_df_raw['month_sin'] = np.sin(2 * np.pi * input_df_raw['datetime'].dt.month / 12.0)
    input_df_raw['month_cos'] = np.cos(2 * np.pi * input_df_raw['datetime'].dt.month / 12.0)

    input_df_raw['weekday_sin'] = np.sin(2 * np.pi * input_df_raw['datetime'].dt.weekday / 7.0)
    input_df_raw['weekday_cos'] = np.cos(2 * np.pi * input_df_raw['datetime'].dt.weekday / 7.0)

    # NEW: Features from the latest KeyError
    input_df_raw['hour_val'] = input_df_raw['datetime'].dt.hour
    input_df_raw['month_val'] = input_df_raw['datetime'].dt.month
    input_df_raw['weekday_val'] = input_df_raw['datetime'].dt.weekday # Monday=0, Sunday=6

    # For 'year_cat', this typically comes from the year of the datetime.
    # If your original dataset only spanned a few years, PyCaret might have
    # treated year as a categorical feature.
    input_df_raw['year_cat'] = input_df_raw['datetime'].dt.year.astype(str) # Convert to string for categorical treatment
    # If your model was trained on only 2011 and 2012 data, and you input 2025,
    # 'year_cat' might be an unknown category. PyCaret usually handles this gracefully
    # during prediction by assigning a default value (e.g., NaN, then imputed)
    # or by raising a warning/error if the category is completely new and not seen during training.
    # For now, converting to string is the direct fix for the KeyError.
    # --- END FEATURE ENGINEERING ---


    st.markdown("---") # Separator

    # --- Prediction Button and Output ---
    if st.button("Get Prediction", help="Click to see the predicted bike demand."):
        st.subheader("Prediction Result")
        try:
            # --- Direct Prediction Using the Loaded Pipeline ---
            # Since final_pipeline IS the PyCaret Pipeline object,
            # its .predict() method handles preprocessing internally.
            raw_prediction = final_pipeline.predict(input_df_raw)

            predicted_count = raw_prediction[0] # Get the single prediction value

            # Display the result, ensuring it's non-negative and an integer
            final_predicted_count = max(0, int(round(predicted_count)))

            st.success(f"Predicted Total Bike Rentals: **{final_predicted_count}**")
            st.write("*(This prediction estimates the combined rentals for both casual and registered users)*")


        except Exception as e:
            st.error("An error occurred during the prediction process.")
            st.exception(e) # Show full error details in logs and Streamlit app

else:
     # Message shown if model loading failed
     st.warning("The prediction pipeline could not be loaded. Please check the model file and deployment logs for details.")


# --- Footer/About (Optional) ---
st.markdown("---")
st.markdown("App based on the Capital Bikeshare Demand Prediction project.")
