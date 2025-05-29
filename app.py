import streamlit as st
import pandas as pd
import numpy as np
import pycaret.regression
import os
from scipy.stats.mstats import winsorize # Import winsorize here
from sklearn.pipeline import Pipeline # Import Pipeline
from sklearn.compose import ColumnTransformer # Import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer, StandardScaler, OneHotEncoder # Import necessary transformers

# --- Define Custom Function Used in Pipeline ---
# This function MUST be defined in app.py so the loaded model can find it
# Ensure the output is a Series to maintain column name for later steps
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

# --- Load the PyCaret Model and Pipeline Separately (Attempt) ---
# In some cases, directly loading the underlying estimator and the preprocessing
# pipeline components might be more stable than relying on PyCaret's full predict_model
# However, extracting the exact pipeline components from the saved PyCaret model can be complex.

# Let's stick to loading the full PyCaret object but call its internal components
@st.cache_resource # Cache the model loading for performance
def load_pycaret_model_and_pipeline(model_path_without_extension):
    """Loads the trained PyCaret regression model and tries to access the internal pipeline."""
    full_model_file_path = f"{model_path_without_extension}.pkl"
    if not os.path.exists(full_model_file_path):
        st.error(f"Model file '{full_model_file_path}' not found. Please ensure the model is saved correctly and is accessible.")
        return None, None
    try:
        loaded_pycaret_model = pycaret.regression.load_model(model_path_without_extension)
        st.success("Prediction model loaded successfully!")

        # Attempt to access the internal preprocessing pipeline
        # This is an internal PyCaret structure, might vary slightly by version
        if hasattr(loaded_pycaret_model, 'pipeline'):
             # If using transform_target, the main pipeline is wrapped
             if hasattr(loaded_pycaret_model.pipeline, 'steps') and loaded_pycaret_model.pipeline.steps[0][0] == 'target_transformer':
                  preprocessing_pipeline = loaded_pycaret_model.pipeline.steps[1][1] # The ColumnTransformer is the second step
                  final_estimator = loaded_pycaret_model.pipeline.steps[2][1] # The final model is the third step
                  st.info("Loaded model with target transformation.")
             else:
                  preprocessing_pipeline = loaded_pycaret_model.pipeline.steps[0][1] # The ColumnTransformer is the first step
                  final_estimator = loaded_pycaret_model.pipeline.steps[1][1] # The final model is the second step
                  st.info("Loaded model without target transformation.")

             # We need to return the preprocessing pipeline and the final estimator
             # return loaded_pycaret_model, preprocessing_pipeline, final_estimator # Returning all three might be useful

             # For manual prediction, we primarily need the preprocessing pipeline and the final estimator
             return loaded_pycaret_model, preprocessing_pipeline, final_estimator

        else:
            st.error("Could not access internal PyCaret pipeline from the loaded model.")
            st.write("Loaded object type:", type(loaded_pycaret_model))
            return loaded_pycaret_model, None, None # Return the loaded object but indicate pipeline access failed

    except Exception as e:
        st.error(f"An error occurred while loading model or accessing pipeline from '{full_model_file_path}': {e}")
        st.exception(e)
        return None, None, None # Return None for all if loading fails completely


# Attempt to load the model and pipeline components
loaded_pycaret_object, preprocessing_pipeline, final_estimator = load_pycaret_model_and_pipeline(MODEL_NAME)

# --- App Title and Overview ---
st.title("🚴‍♀️ Capital Bikeshare Demand Predictor")
st.markdown("""
    Use this app to predict the total number of bike rentals
    from the Capital Bikeshare system in Washington D.C.
    based on various environmental and temporal factors.
""")
st.markdown("---") # Separator


# Only show input and prediction if both pipeline and estimator were successfully accessed
if preprocessing_pipeline is not None and final_estimator is not None:
    st.info("Preprocessing pipeline and final estimator accessed.")
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


    st.markdown("---") # Separator

    # --- Prediction Button and Output ---
    if st.button("Get Prediction", help="Click to see the predicted bike demand."):
        st.subheader("Prediction Result")
        try:
            # --- Manual Preprocessing and Prediction ---
            # Apply the preprocessing pipeline
            # This is the part that replaces pycaret.regression.predict_model(final_model, data=input_df_raw)
            input_processed = preprocessing_pipeline.transform(input_df_raw)

            # Convert the processed output back to a DataFrame (important for some estimators)
            # This step is tricky because ColumnTransformer output is numpy array.
            # We need to reconstruct the feature names.
            # For simplicity, we might skip explicit DataFrame conversion here
            # if the final estimator can handle numpy arrays, but it's safer.

            # A robust way to get feature names after ColumnTransformer is complex
            # and depends on the transformers used (OHE, scaling, custom funcs).
            # Let's proceed assuming the final_estimator can take a numpy array,
            # which is often the case for tree-based models like XGBoost trained with sklearn/PyCaret.

            # Make prediction using the final estimator
            raw_prediction = final_estimator.predict(input_processed)

            # --- Inverse Transform Target if used in PyCaret Setup ---
            # Check if the PyCaret object had a target transformer
            if hasattr(loaded_pycaret_object.pipeline, 'steps') and loaded_pycaret_object.pipeline.steps[0][0] == 'target_transformer':
                 target_transformer = loaded_pycaret_object.pipeline.steps[0][1]
                 # Need to inverse transform the prediction
                 predicted_count = target_transformer.inverse_transform(raw_prediction.reshape(-1, 1))[0][0]
            else:
                 # No target transformation was applied
                 predicted_count = raw_prediction[0]


            # Display the result, ensuring it's non-negative and an integer
            final_predicted_count = max(0, int(round(predicted_count)))

            st.success(f"Predicted Total Bike Rentals: **{final_predicted_count}**")
            st.write("*(This prediction estimates the combined rentals for both casual and registered users)*")


        except Exception as e:
            st.error("An error occurred during the prediction process (manual steps).")
            st.exception(e) # Show full error details in logs and Streamlit app

else:
     # Message shown if model or pipeline components failed to load
     st.warning("The prediction model components could not be loaded. Please check the model file and deployment logs.")


# --- Footer/About (Optional) ---
st.markdown("---")
st.markdown("App based on the Capital Bikeshare Demand Prediction project.")

# Optional: Display input DataFrame for debugging
# with st.expander("View Raw Input Data"):
#    st.dataframe(input_df_raw)
