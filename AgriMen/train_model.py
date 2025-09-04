import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os


os.makedirs("models", exist_ok=True)


df = pd.read_excel("Crop_recommendation.xlsx")


X = df.drop("label", axis=1)
y = df["label"]


le = LabelEncoder()
y_encoded = le.fit_transform(y)


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)


rf = RandomForestClassifier(n_estimators=300, random_state=42)
rf.fit(X_train, y_train)


y_pred = rf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print("Random Forest Accuracy:", acc)
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=le.classes_))


joblib.dump(rf, "models/crop_model.pkl")
joblib.dump(le, "models/label_encoder.pkl")
joblib.dump(scaler, "models/scaler.pkl")

print("\n Model, Label Encoder & Scaler saved inside 'models/' folder successfully!")
