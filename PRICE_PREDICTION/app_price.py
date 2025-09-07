import streamlit as st
import pandas as pd
import joblib

# Load only required artifacts
model = joblib.load("models/price_prediction_model.pkl")
scaler = joblib.load("models/price_scaler.pkl")

# Translation dictionary (full translations as before)
translations = {
    "en": {
        "title": "🌾 AGRIMEN – Smart Crop Price Prediction",
        "welcome": "Welcome! Predict crop market price based on soil & climate data.",
        "manual_input": "📝 Manual Input Prediction",
        "predict_btn": "💰 Predict Price",
        "bulk_prediction": "📂 Bulk Prediction (CSV/Excel Upload)",
        "required_cols": "👉 Required Columns: Crop, Crop_Year, Season, State, Area, Production, Annual_Rainfall, Fertilizer, Pesticide",
        "results": "📊 Predicted Price",
        "unit": "INR per quintal",
        "download": "📥 Download Results as CSV"
    },
    "hi": {
        "title": "🌾 AGRIMEN – स्मार्ट फसल मूल्य पूर्वानुमान",
        "welcome": "स्वागत है! मिट्टी और जलवायु डेटा के आधार पर फसल मूल्य का अनुमान लगाएँ।",
        "manual_input": "📝 मैनुअल इनपुट पूर्वानुमान",
        "predict_btn": "💰 मूल्य अनुमानित करें",
        "bulk_prediction": "📂 बल्क पूर्वानुमान (CSV/Excel अपलोड)",
        "required_cols": "👉 आवश्यक कॉलम: Crop, Crop_Year, Season, State, Area, Production, Annual_Rainfall, Fertilizer, Pesticide",
        "results": "📊 अनुमानित मूल्य",
        "unit": "रुपये प्रति क्विंटल",
        "download": "📥 परिणाम CSV के रूप में डाउनलोड करें"
    },
    "or": { /* Odia Translations */ },
    "te": { /* Telugu Translations */ },
    "ta": { /* Tamil Translations */ },
    "ml": { /* Malayalam Translations */ },
    "bn": { /* Bengali Translations */ }
}

st.set_page_config(page_title="Crop Price Prediction", page_icon="🌾", layout="wide")

lang_options = {
    "en": "English",
    "hi": "Hindi",
    "or": "Odia",
    "te": "Telugu",
    "ta": "Tamil",
    "ml": "Malayalam",
    "bn": "Bengali"
}

lang = st.sidebar.selectbox("🌐 Select Language", list(lang_options.keys()), format_func=lambda x: lang_options[x])

if 'current_lang' not in st.session_state or st.session_state.current_lang != lang:
    st.session_state.current_lang = lang
    st.experimental_rerun()

T = translations[lang]

st.title(T["title"])
st.write(T["welcome"])

st.subheader(T["manual_input"])

col1, col2, col3 = st.columns(3)
with col1:
    crop = st.text_input("Crop (as in dataset)")
    crop_year = st.number_input("Crop Year", 1997, 2025, 2020)
    season = st.text_input("Season (as in dataset)")
with col2:
    state = st.text_input("State (as in dataset)")
    area = st.number_input("Area (in hectares)", min_value=0.1, max_value=1e7, value=1000.0, step=0.1)
    production = st.number_input("Production (in kg)", min_value=0.0, max_value=1e10, value=5000.0, step=1.0)
with col3:
    annual_rainfall = st.number_input("Annual Rainfall (mm)", min_value=0.0, max_value=10000.0, value=1200.0, step=0.1)
    fertilizer = st.number_input("Fertilizer Used (kg)", min_value=0.0, max_value=1e8, value=50000.0, step=1.0)
    pesticide = st.number_input("Pesticide Used (kg)", min_value=0.0, max_value=1e7, value=1000.0, step=1.0)

if st.button(T["predict_btn"]):
    data = [[crop, crop_year, season, state, area, production, annual_rainfall, fertilizer, pesticide]]
    data_df = pd.DataFrame(data, columns=['Crop', 'Crop_Year', 'Season', 'State', 'Area', 'Production', 'Annual_Rainfall', 'Fertilizer', 'Pesticide'])

    # Apply same preprocessing logic from training (use scaler only)
    data_scaled = scaler.transform(data_df)
    price_pred = model.predict(data_scaled)[0]

    st.success(f"✅ {T['results']}: {price_pred:.2f} {T['unit']}")

st.subheader(T["bulk_prediction"])
st.write(T["required_cols"])

file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx"])

if file:
    if file.name.endswith('.csv'):
        df = pd.read_csv(file)
    else:
        df = pd.read_excel(file)

    required_cols = ['Crop', 'Crop_Year', 'Season', 'State', 'Area', 'Production', 'Annual_Rainfall', 'Fertilizer', 'Pesticide']
    if not all(col in df.columns for col in required_cols):
        st.error("❌ Missing required columns in the uploaded file.")
    else:
        input_data = scaler.transform(df[required_cols])
        predictions = model.predict(input_data)

        df['Predicted_Price'] = [f"{pred:.2f} {T['unit']}" for pred in predictions]

        st.write(T["results"])
        st.dataframe(df)

        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(label=T["download"], data=csv, file_name='predicted_prices.csv', mime='text/csv')
