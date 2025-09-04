import streamlit as st
import pandas as pd
import joblib


model = joblib.load("models/crop_model.pkl")
scaler = joblib.load("models/scaler.pkl")
le = joblib.load("models/label_encoder.pkl")



crop_info = {
    "rice": {"sowing": "June-July", "harvest": "Oct-Nov", "duration": "120-150 days"},
    "wheat": {"sowing": "Nov-Dec", "harvest": "Mar-Apr", "duration": "140-160 days"},
    "maize": {"sowing": "June-July", "harvest": "Sep-Oct", "duration": "90-120 days"},
    "barley": {"sowing": "Nov-Dec", "harvest": "Mar-Apr", "duration": "120-140 days"},
    "sorghum": {"sowing": "Jun-Jul", "harvest": "Sep-Oct", "duration": "100-120 days"},
    "millet": {"sowing": "Jun-Jul", "harvest": "Sep-Oct", "duration": "90-110 days"},
    "soybean": {"sowing": "Jun-Jul", "harvest": "Sep-Oct", "duration": "90-120 days"},
    "groundnut": {"sowing": "Jun-Jul", "harvest": "Sep-Oct", "duration": "100-120 days"},
    "mustard": {"sowing": "Oct-Nov", "harvest": "Feb-Mar", "duration": "120-130 days"},
    "cotton": {"sowing": "Apr-May", "harvest": "Oct-Nov", "duration": "150-180 days"},
    "sugarcane": {"sowing": "Feb-Mar", "harvest": "Nov-Dec", "duration": "270-365 days"},
    "tobacco": {"sowing": "Feb-Mar", "harvest": "Aug-Sep", "duration": "150-180 days"},
    "tea": {"sowing": "Jun-Jul", "harvest": "Dec-Feb", "duration": "180-210 days"},
    "coffee": {"sowing": "Jun-Jul", "harvest": "Dec-Feb", "duration": "180-210 days"},
    "banana": {"sowing": "Year-round", "harvest": "9-12 months", "duration": "270-365 days"},
    "mango": {"sowing": "Jul-Aug", "harvest": "Apr-Jun", "duration": "300+ days"},
    "orange": {"sowing": "Jun-Jul", "harvest": "Dec-Jan", "duration": "180-210 days"},
    "apple": {"sowing": "Jan-Feb", "harvest": "Sep-Oct", "duration": "150-180 days"},
    "grapes": {"sowing": "Dec-Jan", "harvest": "Apr-May", "duration": "120-150 days"},
    "coconut": {"sowing": "Year-round", "harvest": "6-10 years", "duration": "2190-3650 days"},
    "potato": {"sowing": "Oct-Nov", "harvest": "Jan-Feb", "duration": "90-120 days"},
    "tomato": {"sowing": "Sep-Oct", "harvest": "Dec-Jan", "duration": "90-100 days"}
}



crop_translations = {
    "wheat": {"en": "Wheat", "hi": "गेहूँ", "or": "ଗହମ", "te": "గోధుమ", "ta": "கோதுமை", "ml": "ഗേहुँ", "bn": "গম"},
    "rice": {"en": "Rice", "hi": "चावल", "or": "ଚାଉଳ", "te": "బియ్యం", "ta": "அரிசி", "ml": "അരിപ്പ്", "bn": "চাল"},
    "maize": {"en": "Maize", "hi": "मक्का", "or": "ମକା", "te": "మక్కా", "ta": "மக்காச்சோளம்", "ml": "ഭൂട്ടാൻ", "bn": "মকাই"},
    "barley": {"en": "Barley", "hi": "जौ", "or": "ଜୌ", "te": "జొన్న", "ta": "பார்லி", "ml": "ബാർലി", "bn": "যব"},
    "sorghum": {"en": "Sorghum", "hi": "ज्वार", "or": "ଜ୍ୱାର", "te": "జొన్న", "ta": "சோர்கம்", "ml": "സോർഗം", "bn": "জোয়ার"},
    "millet": {"en": "Millet", "hi": "बाजरा", "or": "ବାଜରା", "te": "సజ్జా", "ta": "சாமை", "ml": "സജ്ജ", "bn": "বজরা"},
    "soybean": {"en": "Soybean", "hi": "सोयाबीन", "or": "ସୋୟାବିନ", "te": "సోయాబీన్", "ta": "சாய்பீன்", "ml": "സോയാബീൻ", "bn": "সয়াবিন"},
    "groundnut": {"en": "Groundnut", "hi": "मूँगफली", "or": "ବେର ଚଣା", "te": "వేరుశనగ", "ta": "வேர்க்கடலை", "ml": "അറുക്ക്", "bn": "শুঁটকোলা"},
    "mustard": {"en": "Mustard", "hi": "सरसों", "or": "ସରିସ", "te": "ఆవాలు", "ta": "கடலைப்பூசணி", "ml": "മസ്താർഡ്", "bn": "সরিষা"},
    "cotton": {"en": "Cotton", "hi": "कपास", "or": "କପାସ", "te": "పత్తి", "ta": "பருத்தி", "ml": "പട്ടു", "bn": "কাপাস"},
    "sugarcane": {"en": "Sugarcane", "hi": "गन्ना", "or": "ଇଁଡ଼ିଆ ଗନ୍ନା", "te": "చెరకు", "ta": "கரும்பு", "ml": "ശർക്കരക്കടി", "bn": "আখ"},
    "tobacco": {"en": "Tobacco", "hi": "तमाकू", "or": "ତମାକୁ", "te": "తమాకు", "ta": "புகையிலை", "ml": "തൂവാലി", "bn": "তামাক"},
    "tea": {"en": "Tea", "hi": "चाय", "or": "ଚା", "te": "టీ", "ta": "டீ", "ml": "ചായ", "bn": "চা"},
    "coffee": {"en": "Coffee", "hi": "कॉफी", "or": "କଫି", "te": "కాఫీ", "ta": "காஃபி", "ml": "കോഫി", "bn": "কফি"},
    "banana": {"en": "Banana", "hi": "केला", "or": "କଦଳୀ", "te": "అరటి", "ta": "வாழை", "ml": "വാഴപ്പഴം", "bn": "কলা"},
    "mango": {"en": "Mango", "hi": "आम", "or": "ଆମ୍ବ", "te": "మామిడి", "ta": "மாம்பழம்", "ml": "മാമ്പഴം", "bn": "আম"},
    "orange": {"en": "Orange", "hi": "संतरा", "or": "କମଳ", "te": "కమలఫలం", "ta": "ஆரஞ்சு", "ml": "ഓറഞ്ച്", "bn": "কমলা"},
    "apple": {"en": "Apple", "hi": "सेब", "or": "ସେବ", "te": "ఆపిల్", "ta": "ஆப்பிள்", "ml": "ആപ്പിൾ", "bn": "আপেল"},
    "grapes": {"en": "Grapes", "hi": "अंगूर", "or": "ଆଙ୍ଗୁର", "te": "ద్రాక్ష", "ta": "திராட்சை", "ml": "ദ്രാക്ഷ", "bn": "আঙুর"},
    "coconut": {"en": "Coconut", "hi": "नारियल", "or": "ନାଡିଆଳ", "te": "కొబ్బరి", "ta": "தேங்காய்", "ml": "തേങ്ങ", "bn": "নারিকেল"},
    "potato": {"en": "Potato", "hi": "आलू", "or": "ଆଳୁ", "te": "ఆలుగడ్డ", "ta": "உருளைக்கிழங்கு", "ml": "ഉരുളകിഴങ്ങ്", "bn": "আলু"},
    "tomato": {"en": "Tomato", "hi": "टमाटर", "or": "ଟମାଟର", "te": "టమోటా", "ta": "தக்காளி", "ml": "തക്കാളി", "bn": "টমেটো"}
}



translations = {
    "en": {"title":"🌾 AGRIMEN - Smart Crop Recommendation System","welcome":"Welcome to AGRIMEN! This tool predicts the best crop based on soil and climate conditions.","manual_input":"📝 Manual Input Prediction","predict_btn":"🌱 Predict Crop","bulk_prediction":"📂 Bulk Prediction (CSV/Excel Upload)","required_cols":"👉 Upload a file with columns: N, P, K, temperature, humidity, ph, rainfall","results":"📊 Predicted Crop Timetable","download":"📥 Download as CSV","sowing":"Sowing","harvest":"Harvest","duration":"Duration"},
    "hi": {"title":"🌾 AGRIMEN - स्मार्ट फसल सिफारिश प्रणाली","welcome":"AGRIMEN में आपका स्वागत है! यह उपकरण मिट्टी और जलवायु के आधार पर सर्वोत्तम फसल की भविष्यवाणी करता है।","manual_input":"📝 मैन्युअल इनपुट भविष्यवाणी","predict_btn":"🌱 फसल भविष्यवाणी करें","bulk_prediction":"📂 बल्क भविष्यवाणी (CSV/Excel अपलोड)","required_cols":"👉 इन कॉलम वाली फ़ाइल अपलोड करें: N, P, K, temperature, humidity, ph, rainfall","results":"📊 भविष्यवाणी की गई फसल समय सारणी","download":"📥 CSV के रूप में डाउनलोड करें","sowing":"बुआई","harvest":"कटाई","duration":"अवधि"},
    "or": {"title":"🌾 AGRIMEN - ସ୍ମାର୍ଟ ଫସଲ ସୁପାରିଶ ପ୍ରଣାଳୀ","welcome":"AGRIMEN କୁ ସ୍ୱାଗତ! ଏହି ଉପକରଣ ମାଟି ଏବଂ ଜଳବାୟୁ ଉପରେ ଆଧାର କରି ଭଲ ଫସଲ ପୂର୍ବାନୁମାନ କରେ।","manual_input":"📝 ମାନୁଆଲ୍ ଇନପୁଟ ପୂର୍ବାନୁମାନ","predict_btn":"🌱 ଫସଲ ପୂର୍ବାନୁମାନ କରନ୍ତୁ","bulk_prediction":"📂 ବଲ୍କ ପୂର୍ବାନୁମାନ (CSV/Excel ଅପଲୋଡ୍)","required_cols":"👉 ଏହି କଲମଗୁଡିକ ସହିତ ଫାଇଲ୍ ଅପଲୋଡ୍ କରନ୍ତୁ: N, P, K, temperature, humidity, ph, rainfall","results":"📊 ପୂର୍ବାନୁମାନ ହୋଇଥିବା ଫସଲ ଟାଇମଟେବଲ୍","download":"📥 CSV ଭାବରେ ଡାଉନଲୋଡ୍ କରନ୍ତୁ","sowing":"ବିଆର ମସିବା","harvest":"କାଟନି","duration":"ମିୟାଦ"},
    "te": {"title":"🌾 AGRIMEN - స్మార్ట్ పంట సిఫారసు వ్యవస్థ","welcome":"AGRIMEN కి స్వాగతం! ఈ సాధనం నేల మరియు వాతావరణ పరిస్థితుల ఆధారంగా ఉత్తమ పంటను అంచనా వేస్తుంది.","manual_input":"📝 మాన్యువల్ ఇన్‌పుట్ అంచనా","predict_btn":"🌱 పంటను అంచనా వేయండి","bulk_prediction":"📂 బల్క్ అంచనా (CSV/Excel అప్‌లోడ్)","required_cols":"👉 ఈ కాలమ్‌లతో ఫైల్‌ని అప్‌లోడ్ చేయండి: N, P, K, temperature, humidity, ph, rainfall","results":"📊 అంచనా వేసిన పంట షెడ్యూల్","download":"📥 CSV గా డౌన్‌లోడ్ చేయండి","sowing":"విత్తనం","harvest":"కత్తెర","duration":"వ్యవధి"},
    "ta": {"title":"🌾 AGRIMEN - ஸ்மார்ட் பயிர் பரிந்துரை அமைப்பு","welcome":"AGRIMEN-க்கு வரவேற்கிறோம்! இந்த கருவி மண் மற்றும் காலநிலையின் அடிப்படையில் சிறந்த பயிரை கணிக்கிறது.","manual_input":"📝 கையேடு உள்ளீட்டு கணிப்பு","predict_btn":"🌱 பயிரை கணிக்கவும்","bulk_prediction":"📂 மொத்த கணிப்பு (CSV/Excel பதிவேற்றம்)","required_cols":"👉 இந்த நெடுவரிசைகளுடன் கோப்பைப் பதிவேற்றவும்: N, P, K, temperature, humidity, ph, rainfall","results":"📊 கணிக்கப்பட்ட பயிர் அட்டவணை","download":"📥 CSV ஆக பதிவிறக்கவும்","sowing":"விதைப்பது","harvest":"பழுப்பு","duration":"காலம்"},
    "ml": {"title":"🌾 AGRIMEN - സ്മാർട്ട് വിള നിർദ്ദേശ സംവിധാനം","welcome":"AGRIMEN-ലേക്ക് സ്വാഗതം! ഈ ഉപകരണം മണ്ണിന്റെയും കാലാവസ്ഥയുടെയും അടിസ്ഥാനത്തിൽ മികച്ച വിള പ്രവചിക്കുന്നു.","manual_input":"📝 മാനുവൽ ഇൻപുട്ട് പ്രവചനം","predict_btn":"🌱 വിള പ്രവചിക്കുക","bulk_prediction":"📂 ബൾക്ക് പ്രവചനം (CSV/Excel അപ്ലോഡ്)","required_cols":"👉 ഈ കോളങ്ങളോടുകൂടിയ ഫയൽ അപ്ലോഡ് ചെയ്യുക: N, P, K, temperature, humidity, ph, rainfall","results":"📊 പ്രവചിച്ച വിള ഷെഡ്യൂൾ","download":"📥 CSV ആയി ഡൗൺലോഡ് ചെയ്യുക","sowing":"തൈവ്","harvest":"പൊക്കം","duration":"കാലാവധി"},
    "bn": {"title":"🌾 AGRIMEN - স্মার্ট ফসল সুপারিশ ব্যবস্থা","welcome":"AGRIMEN-এ স্বাগতম! এই টুল মাটি ও জলবায়ুর উপর ভিত্তি করে সেরা ফসল পূর্বাভাস দেয়।","manual_input":"📝 ম্যানুয়াল ইনপুট পূর্বাভাস","predict_btn":"🌱 ফসল পূর্বাভাস করুন","bulk_prediction":"📂 বাল্ক পূর্বাভাস (CSV/Excel আপলোড)","required_cols":"👉 নিম্নলিখিত কলাম সহ ফাইল আপলোড করুন: N, P, K, temperature, humidity, ph, rainfall","results":"📊 পূর্বাভাসিত ফসল সময়সূচি","download":"📥 CSV হিসাবে ডাউনলোড করুন","sowing":"বপন","harvest":"কাটা","duration":"সময়কাল"}
}




st.set_page_config(page_title="AGRIMEN", page_icon="🌾", layout="wide")

# Dropdown shows language names in English only
lang_options = {"en":"English","hi":"Hindi","or":"Odia","te":"Telugu","ta":"Tamil","ml":"Malayalam","bn":"Bengali"}
lang = st.sidebar.selectbox("🌐 Choose Language", options=list(lang_options.keys()), format_func=lambda x: lang_options[x])
T = translations[lang]

st.title(T["title"])
st.write(T["welcome"])



col1, col2, col3 = st.columns(3)
with col1:
    N = st.number_input("Nitrogen (N)", 0, 200, 50)
    P = st.number_input("Phosphorus (P)", 0, 200, 50)
    K = st.number_input("Potassium (K)", 0, 200, 50)
with col2:
    temperature = st.number_input("Temperature (°C)", -10.0, 60.0, 25.0)
    humidity = st.number_input("Humidity (%)", 0.0, 100.0, 50.0)
with col3:
    ph = st.number_input("pH", 0.0, 14.0, 6.5)
    rainfall = st.number_input("Rainfall (mm)", 0.0, 500.0, 100.0)

if st.button(T["predict_btn"]):
    data = scaler.transform([[N, P, K, temperature, humidity, ph, rainfall]])
    pred = model.predict(data)
    crop_en = le.inverse_transform(pred)[0]
    crop_local = crop_translations.get(crop_en.lower(), {}).get(lang, crop_en)
    st.success(f"✅ {T['results']}: {crop_local}")

    if crop_en.lower() in crop_info:
        info = crop_info[crop_en.lower()]
        st.info(f"📅 **{T['sowing']}:** {info['sowing']} | 🌾 **{T['harvest']}:** {info['harvest']} | ⏳ **{T['duration']}:** {info['duration']}")



st.subheader(T["bulk_prediction"])
st.write(T["required_cols"])
file = st.file_uploader("Upload CSV or Excel", type=["csv","xlsx"])

if file:
    if file.name.endswith(".csv"):
        df = pd.read_csv(file)
    else:
        df = pd.read_excel(file)

    input_data = scaler.transform(df[["N","P","K","temperature","humidity","ph","rainfall"]])
    preds = model.predict(input_data)
    df["Predicted Crop"] = le.inverse_transform(preds)
    df["Predicted Crop"] = df["Predicted Crop"].apply(lambda x: crop_translations.get(x.lower(), {}).get(lang,x))

    st.write(T["results"])
    st.dataframe(df)

    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(label=T["download"], data=csv, file_name="predicted_crops.csv", mime="text/csv")
