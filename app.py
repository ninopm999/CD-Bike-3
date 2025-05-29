import streamlit as st
import pandas as pd
import numpy as np
import pycaret.regression
import os
from scipy.stats.mstats import winsorize # Keep this for the custom function


# --- Define Custom Functions Used in Pipeline ---
# These functions MUST be defined in app.py so the loaded model can find them

def winsorize_series_robust(df_or_series, limits=(0.01, 0.01)):
    if isinstance(df_or_series, pd.DataFrame):
        series_to_winsorize = df_or_series.iloc[:, 0].copy()
    else:
        series_to_winsorize = df_or_series.copy()

    if series_to_winsorize.empty or not pd.api.types.is_numeric_dtype(series_to_winsorize):
         return np.empty((len(series_to_winsorize), 0)) if series_to_winsorize.empty else np.empty((0, 0))

    winsorized_array = winsorize(series_to_winsorize, limits=limits)
    return winsorized_array.reshape(-1, 1) # Ensure 2D output


# Function to preprocess initial features (extract datetime components and DROP original)
def preprocess_initial_features_v3(input_df):
    df = input_df.copy()
    # Keep original datetime for now, as create_cyclical_features_v3 needs it,
    # and it might be needed by PyCaret's internal datetime handling steps *before*
    # the ColumnTransformer for feature engineering that PyCaret might have
    # secretly added before your 'custom_pipeline'. Let's engineer first, then drop datetime
    # before passing to the ColumnTransformer logic within the pipeline.

    if 'datetime' in df.columns:
        df['hour_val'] = df['datetime'].dt.hour
        df['month_val'] = df['datetime'].dt.month
        df['weekday_val'] = df['datetime'].dt.weekday # Mon=0, Sun=6
        df['day'] = df['datetime'].dt.day
        df['year_cat'] = df['datetime'].dt.year.astype(str)
        df['dayofyear'] = df['datetime'].dt.dayofyear
        # DO NOT DROP DATETIME HERE YET. We need it for cyclical features.
        # The dropping will happen *after* engineering, just before prediction.
    if 'atemp' in df.columns:
        df = df.drop('atemp', axis=1) # Drop atemp
    return df

# Function to create cyclical datetime features
def create_cyclical_features_v3(input_df):
    df = input_df.copy()
    # Ensure hour_val, month_val, weekday_val are present from preprocess_initial_features_v3
    if 'hour_val' in df.columns: # Check for extracted column names
        df['hour_sin'] = np.sin(2 * np.pi * df['hour_val']/24.0)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour_val']/24.0)
    if 'month_val' in df.columns: # Check for extracted column names
        df['month_sin'] = np.sin(2 * np.pi * df['month_val']/12.0)
        df['month_cos'] = np.cos(2 * np.pi * df['month_val']/12.0)
    if 'weekday_val' in df.columns: # Check for extracted column names
        df['weekday_sin'] = np.sin(2 * np.pi * df['weekday_val']/7.0)
        df['weekday_cos'] = np.cos(2 * np.pi * df['weekday_val']/7.0)
    # The _val columns (hour_val, month_val, weekday_val) are needed by PyCaret's OHE step, DO NOT DROP HERE
    # The original 'datetime' column should be dropped *after* this step, before passing to the PyCaret pipeline
    return df


# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="Bike Rental Demand Predictor",
    layout="centered", # 'centered' or 'wide'
    initial_sidebar_state="auto"
)

# --- Define Model File Path ---
MODEL_NAME = 'XGBoost_BikeSharing_Final_Model_PyCaret' # Use your EXACT filename without .pkl
MODEL_FILE_PATH = f'{MODEL_NAME}.pkl'

# --- Load the PyCaret Pipeline and get Target Transformer ---
@st.cache_resource # Cache the model loading for performance
def load_pycaret_pipeline_and_transformers(model_path_without_extension):
    """Loads the trained PyCaret Pipeline and extracts necessary transformers."""
    full_model_file_path = f"{model_path_without_extension}.pkl"
    if not os.path.exists(full_model_file_path):
        st.error(f"Model file '{full_model_file_path}' not found. Please ensure the model is saved correctly and is accessible.")
        return None, None

    loaded_pipeline = None
    target_transformer = None

    try:
        # Load the model - this directly loads the Pipeline
        loaded_pipeline = pycaret.regression.load_model(model_path_without_extension)
        st.success("Prediction pipeline loaded successfully!")

        # Try to find the target transformer
        if isinstance(loaded_pipeline, Pipeline):
             for step_name, step_transformer in loaded_pipeline.steps:
                  if step_name == 'target_transformation':
                       target_transformer = step_transformer
                       st.info("Target transformer found.")
                       break

        return loaded_pipeline, target_transformer

    except Exception as e:
        st.error(f"An error occurred while loading the model from '{full_model_file_path}': {e}")
        st.exception(e) # Display the full error for debugging
        return None, None


# Attempt to load the PyCaret Pipeline and target transformer
final_pipeline, target_transformer = load_pycaret_pipeline_and_transformers(MODEL_NAME)


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
    # Start with the original columns PyCaret setup received
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

    # Ensure datetime column is in the correct format before engineering
    input_df_raw['datetime'] = pd.to_datetime(input_df_raw['datetime'])

    # --- Perform Manual Feature Engineering on Input Data ---
    # Apply the same datetime feature engineering functions as in Colab
    # Do NOT drop datetime within preprocess_initial_features_v3 anymore, drop it *after* all engineering
    input_df_engineered = preprocess_initial_features_v3(input_df_raw.copy())
    input_df_engineered = create_cyclical_features_v3(input_df_engineered.copy())

    # --- KEY CHANGE: Drop the original 'datetime' column AFTER engineering ---
    # This is the DataFrame that should be passed to the PyCaret pipeline's predict method.
    if 'datetime' in input_df_engineered.columns:
         input_df_for_prediction = input_df_engineered.drop('datetime', axis=1)
    else:
         input_df_for_prediction = input_df_engineered # Should not happen if engineering functions work


    # st.write("Columns before passing to pipeline (should NOT include datetime):", input_df_for_prediction.columns.tolist())
    # st.write("Sample data before passing to pipeline:", input_df_for_prediction.head())


    st.markdown("---") # Separator

    # --- Prediction Button and Output ---
    if st.button("Get Prediction", help="Click to see the predicted bike demand."):
        st.subheader("Prediction Result")
        try:
            # --- Direct Prediction Using the Loaded Pipeline ---
            # Pass the manually ENGINEERED and DATETIME-DROPPED DataFrame to the pipeline's predict method
            raw_prediction_transformed = final_pipeline.predict(input_df_for_prediction) # Use the ENGINEERED dataframe

            # The output raw_prediction_transformed is in the TRANSFORMED target scale
            # We need to apply the inverse transformation if the transformer was found
            if target_transformer is not None:
                 # Inverse transform expects a 2D array
                 predicted_count = target_transformer.inverse_transform(raw_prediction_transformed.reshape(-1, 1))[0][0]
            else:
                 # If no target transformer was found (unexpected based on pipeline structure)
                 st.warning("Could not find target transformer. Prediction is in the original scale.")
                 predicted_count = raw_prediction_transformed[0]


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

# Optional: Display input DataFrame for debugging
# with st.expander("View Raw Input Data"):
#    st.dataframe(input_df_raw)
# with st.expander("View Engineered Input Data"):
#    st.dataframe(input_df_engineered)
# with st.expander("View Data Passed to Pipeline"):
#     if 'input_df_for_prediction' in locals():
#          st.dataframe(input_df_for_prediction)
