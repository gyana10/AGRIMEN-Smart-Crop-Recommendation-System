import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Odisha Crop Yield Prediction",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- LOAD SAVED MODELS AND PREPROCESSORS ---
@st.cache_resource
def load_artifacts():
    """
    Loads the trained model, one-hot encoder, and scaler.
    """
    model_path = 'models/crop_yield_model.pkl'
    ohe_path = 'models/yield_ohe_encoder.pkl'
    scaler_path = 'models/yield_scaler.pkl'

    # Check if all required model files exist
    if not all(os.path.exists(p) for p in [model_path, ohe_path, scaler_path]):
        st.error("Error: One or more model files are missing from the 'models' directory.")
        st.info("Please ensure you have run the `train_yield_model.py` script to generate the required model files.")
        return None, None, None
        
    try:
        model = joblib.load(model_path)
        ohe_encoder = joblib.load(ohe_path)
        scaler = joblib.load(scaler_path)
        return model, ohe_encoder, scaler
    except Exception as e:
        st.error(f"An unexpected error occurred while loading model files: {e}")
        return None, None, None

model, ohe_encoder, scaler = load_artifacts()

# --- DATA FOR UI SELECTIONS ---
# These lists should be derived from the unique values in your original training dataset
CROP_TYPES = [
    'Rice', 'Maize', 'Wheat', 'Ragi', 'Moong', 'Groundnut', 'Sugarcane',
    'Cotton', 'Jute', 'Potato', 'Onion', 'Turmeric', 'Ginger', 'Chilli'
]
SEASONS = ['Kharif', 'Rabi', 'Summer']
DISTRICTS = [
    'Angul', 'Balangir', 'Balasore', 'Bargarh', 'Bhadrak', 'Boudh',
    'Cuttack', 'Deogarh', 'Dhenkanal', 'Gajapati', 'Ganjam', 'Jagatsinghpur',
    'Jajpur', 'Jharsuguda', 'Kalahandi', 'Kandhamal', 'Kendrapara',
    'Keonjhar', 'Khordha', 'Koraput', 'Malkangiri', 'Mayurbhanj',
    'Nabarangpur', 'Nayagarh', 'Nuapada', 'Puri', 'Rayagada', 'Sambalpur',
    'Subarnapur', 'Sundargarh'
]

# --- UI (USER INTERFACE) ---
st.title("🌾 Odisha Crop Yield Prediction")
st.markdown("Enter the details of the crop and environmental factors to predict the crop yield in tonnes per hectare.")

st.sidebar.header("Enter Input Parameters")

# --- USER INPUTS VIA SIDEBAR (CORRECTED) ---
# TARGET LEAKAGE FIX: REMOVED inputs for 'Cultivated area (hectares)' and 'Total production (tonnes)'
crop_type = st.sidebar.selectbox("Select Crop Type", options=sorted(CROP_TYPES))
season = st.sidebar.selectbox("Select Season", options=sorted(SEASONS))
district = st.sidebar.selectbox("Select District", options=sorted(DISTRICTS))

annual_rainfall = st.sidebar.number_input(
    "Annual Rainfall (mm)",
    min_value=0.0,
    max_value=5000.0,
    value=1500.0,
    step=50.0,
    help="Enter the total annual rainfall in millimeters."
)

pesticide_fertilizer_used = st.sidebar.number_input(
    "Pesticide & Fertilizer Used (kg/hectare)",
    min_value=0.0,
    max_value=1000.0,
    value=100.0,
    step=10.0,
    help="Enter the amount of pesticide and fertilizer used per hectare in kilograms."
)

# --- PREDICTION LOGIC ---
if st.sidebar.button("Predict Yield", type="primary"):
    if model is not None and ohe_encoder is not None and scaler is not None:
        # 1. Create a DataFrame from user inputs
        input_data = pd.DataFrame({
            'Crop type': [crop_type],
            'Season': [season],
            'District': [district],
            'Annual rainfall (mm)': [annual_rainfall],
            'Pesticide and fertilizer used (kg/hectare)': [pesticide_fertilizer_used]
        })

        # 2. Feature Engineering (must be identical to training script)
        input_data['Rainfall_Pesticide'] = input_data['Annual rainfall (mm)'] * input_data['Pesticide and fertilizer used (kg/hectare)']

        # 3. Separate categorical and numerical features
        cat_features = input_data[['Crop type', 'Season', 'District']]
        # CORRECTED: Only include features the scaler was trained on
        num_features = input_data[['Annual rainfall (mm)', 'Pesticide and fertilizer used (kg/hectare)', 'Rainfall_Pesticide']]

        # 4. Apply pre-trained One-Hot Encoding and Scaling
        try:
            cat_encoded = ohe_encoder.transform(cat_features)
            num_scaled = scaler.transform(num_features)

            # 5. Combine features in the correct order
            final_features = np.hstack([cat_encoded, num_scaled])

            # 6. Make prediction
            prediction = model.predict(final_features)
            predicted_yield = prediction[0]

            # --- DISPLAY RESULT ---
            st.subheader("Prediction Result")
            col1, col2 = st.columns([2, 3])
            with col1:
                st.metric(
                    label="Predicted Crop Yield",
                    value=f"{predicted_yield:.2f} tonnes/hectare"
                )
                st.info(
                    "This is a predictive model. Actual yield can be influenced by other factors like soil health, pests, and extreme weather events."
                )
            with col2:
                st.image(
                    "https://placehold.co/600x400/2E8B57/FFFFFF?text=Lush+Fields&font=lato",
                    caption="Crop yield prediction helps in agricultural planning and ensuring food security."
                )
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")

    else:
        st.warning("Cannot predict as model files are not loaded correctly. Please check that the 'models' folder exists and is populated.")

st.markdown("---")
st.write("Developed based on the Odisha Crop Dataset.")

