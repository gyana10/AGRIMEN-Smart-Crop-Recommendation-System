# app_price.py
import streamlit as st
import pandas as pd
import joblib
import os

# -------------------------------
# Load models and metadata
# -------------------------------
model = joblib.load("models/price_prediction_model.pkl")
scaler = joblib.load("models/price_scaler.pkl")
expected_features = joblib.load("models/feature_columns.pkl")

# -------------------------------
# Multilingual dictionary
# -------------------------------
translations = {
    'English': {
        'title': "Crop Price Prediction",
        'Year': "Year",
        'Month': "Month",
        'DayOfWeek': "Day of Week",
        'District': "District",
        'Market': "Market",
        'Commodity': "Commodity",
        'Variety': "Variety",
        'Grade': "Grade",
        'Predict': "Predict Price",
        'Upload_File': "Upload CSV/Excel for Batch Prediction",
        'Predicted Price': "Predicted Price"
    },
    'Hindi': {
        'title': "फसल मूल्य भविष्यवाणी",
        'Year': "वर्ष",
        'Month': "महीना",
        'DayOfWeek': "सप्ताह का दिन",
        'District': "जिला",
        'Market': "बाजार",
        'Commodity': "कृषि उत्पाद",
        'Variety': "प्रजाति",
        'Grade': "ग्रेड",
        'Predict': "मूल्य अनुमानित करें",
        'Upload_File': "बैच भविष्यवाणी के लिए CSV/Excel अपलोड करें",
        'Predicted Price': "अनुमानित मूल्य"
    },
    'Telugu': {
        'title': "పంట ధర అంచనా",
        'Year': "వర్షం",
        'Month': "నెల",
        'DayOfWeek': "వారం రోజు",
        'District': "జిల్లా",
        'Market': "మార్కెట్",
        'Commodity': "పంట",
        'Variety': "వేరియంట్స్",
        'Grade': "గ్రేడ్",
        'Predict': "ధర అంచనా వేయండి",
        'Upload_File': "బ్యాచ్ అంచనాకు CSV/Excel అప్‌లోడ్ చేయండి",
        'Predicted Price': "అంచనా ధర"
    },
    'Tamil': {
        'title': "பயிர் விலை கணிப்பு",
        'Year': "ஆண்டு",
        'Month': "மாதம்",
        'DayOfWeek': "வாரம் நாள்",
        'District': "மாவட்டம்",
        'Market': "சந்தை",
        'Commodity': "பயிர்",
        'Variety': "விதம்",
        'Grade': "தரம்",
        'Predict': "விலை கணிக்கவும்",
        'Upload_File': "தொகுப்பு கணிப்பிற்கு CSV/Excel பதிவேற்றவும்",
        'Predicted Price': "கணிக்கப்பட்ட விலை"
    },
    'Odia': {
        'title': "ଫସଲ ମୂଲ୍ୟ ପୂର୍ବାନୁମାନ",
        'Year': "ବର୍ଷ",
        'Month': "ମାସ",
        'DayOfWeek': "ସପ୍ତାହର ଦିନ",
        'District': "ଜିଲ୍ଲା",
        'Market': "ବଜାର",
        'Commodity': "ଫସଲ",
        'Variety': "ବିଭିନ୍ନ ପ୍ରକାର",
        'Grade': "ଗ୍ରେଡ୍",
        'Predict': "ମୂଲ୍ୟ ପୂର୍ବାନୁମାନ କରନ୍ତୁ",
        'Upload_File': "ବ୍ୟାଚ୍ ପୂର୍ବାନୁମାନ ପାଇଁ CSV/Excel ଅପଲୋଡ୍ କରନ୍ତୁ",
        'Predicted Price': "ପୂର୍ବାନୁମାନିତ ମୂଲ୍ୟ"
    },
    'Bengali': {
        'title': "ফসলের মূল্য পূর্বাভাস",
        'Year': "বছর",
        'Month': "মাস",
        'DayOfWeek': "সপ্তাহের দিন",
        'District': "জেলা",
        'Market': "বাজার",
        'Commodity': "ফসল",
        'Variety': "প্রকার",
        'Grade': "গ্রেড",
        'Predict': "মূল্য পূর্বাভাস করুন",
        'Upload_File': "ব্যাচ পূর্বাভাসের জন্য CSV/Excel আপলোড করুন",
        'Predicted Price': "পূর্বাভাসিত মূল্য"
    }
}

# -------------------------------
# Sidebar for language selection
# -------------------------------
language = st.sidebar.selectbox("Select Language / ভাষা / భాష / மொழி / ଭାଷା / ভাষা", 
                                ['English','Hindi','Telugu','Tamil','Odia','Bengali'])
t = translations[language]

st.title(t['title'])

# -------------------------------
# Input fields
# -------------------------------
year = st.number_input(t['Year'], min_value=2000, max_value=2030, value=2025)
month = st.number_input(t['Month'], min_value=1, max_value=12, value=9)
day_of_week = st.number_input(t['DayOfWeek'], min_value=1, max_value=7, value=1)

# Get all available options from feature columns
districts = sorted([f.replace('District_', '') for f in expected_features if f.startswith('District_')])
markets = sorted([f.replace('Market_', '') for f in expected_features if f.startswith('Market_')])
commodities = sorted([f.replace('Commodity_', '') for f in expected_features if f.startswith('Commodity_')])
varieties = sorted([f.replace('Variety_', '') for f in expected_features if f.startswith('Variety_')])
grades = sorted([f.replace('Grade_', '') for f in expected_features if f.startswith('Grade_')])

district = st.selectbox(t['District'], districts)
market = st.selectbox(t['Market'], markets)
commodity = st.selectbox(t['Commodity'], commodities)
variety = st.selectbox(t['Variety'], varieties)
grade = st.selectbox(t['Grade'], grades)

submit_button = st.button(t['Predict'])

# -------------------------------
# Single prediction
# -------------------------------
if submit_button:
    input_data = {
        'Year': year,
        'Month': month,
        'DayOfWeek': day_of_week,
        'District_' + district: 1,
        'Market_' + market: 1,
        'Commodity_' + commodity: 1,
        'Variety_' + variety: 1,
        'Grade_' + grade: 1
    }
    input_df = pd.DataFrame([input_data])
    input_df = input_df.reindex(columns=expected_features, fill_value=0)
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)[0]
    st.success(f"{t['Predicted Price']}: ₹ {prediction:,.2f}")

# -------------------------------
# Batch prediction (CSV/Excel)
# -------------------------------
st.subheader(t['Upload_File'])
uploaded_file = st.file_uploader("", type=['csv','xlsx'])
if uploaded_file:
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    # Ensure all feature columns exist
    for col in expected_features:
        if col not in df.columns:
            df[col] = 0

    df_scaled = scaler.transform(df[expected_features])
    predictions = model.predict(df_scaled)
    df['Predicted_Price (₹)'] = predictions
    st.dataframe(df)
    df.to_csv('predicted_prices.csv', index=False)
    st.success('Predictions saved to predicted_prices.csv')
