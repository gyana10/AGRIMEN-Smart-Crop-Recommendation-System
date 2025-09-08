import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os

# Step 1: Load dataset
df = pd.read_csv('market.csv')

print(f"Columns in dataset: {df.columns.tolist()}")

# Step 2: Drop constant columns
df = df.loc[:, (df != df.iloc[0]).any()]

# Step 3: Define Features and Target
X = df.drop(['Modal_Price'], axis=1)
y = df['Modal_Price']

# Step 4: Data Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 5: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Step 6: Hyperparameter Tuning (RandomizedSearchCV)
param_distributions = {
    'n_estimators': [100, 200, 300, 400],
    'max_depth': [4, 6, 8, 10],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'subsample': [0.7, 0.8, 0.9, 1.0]
}

random_search = RandomizedSearchCV(
    estimator=XGBRegressor(objective='reg:squarederror', random_state=42),
    param_distributions=param_distributions,
    n_iter=15,
    scoring='r2',
    cv=5,
    n_jobs=-1,
    verbose=1,
    random_state=42
)

random_search.fit(X_train, y_train)

# Step 7: Best Model
best_model = random_search.best_estimator_

# Step 8: Model Evaluation
y_pred = best_model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\nModel Performance:")
print(f"MAE: {mae:.2f}")
print(f"MSE: {mse:.2f}")
print(f"R² Score: {r2:.4f}")

# Step 9: Save Artifacts
os.makedirs("models", exist_ok=True)
joblib.dump(best_model, "models/price_prediction_model.pkl")
joblib.dump(scaler, "models/price_scaler.pkl")
joblib.dump(X.columns.tolist(), "models/feature_columns.pkl")

print("\nModel, Scaler, and Feature Columns saved successfully in 'models/' folder!")
