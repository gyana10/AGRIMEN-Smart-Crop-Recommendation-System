import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os

# --- SETUP ---
# Create a directory to save the models if it doesn't exist
print("Creating 'models' directory...")
os.makedirs('models', exist_ok=True)

# --- DATA LOADING ---
print("Loading dataset: 'odisha_crop_data.csv'...")
try:
    df = pd.read_csv('odisha_crop_data.csv')
except FileNotFoundError:
    print("Error: 'odisha_crop_data.csv' not found. Please place it in the same directory.")
    exit()

# --- FEATURE SELECTION (CORRECTED) ---
# TARGET LEAKAGE FIX: Removed 'Cultivated area (hectares)' and 'Total production (tonnes)' from features
print("Selecting features and target. 'Total production' and 'Cultivated area' are excluded from features to prevent target leakage.")
X = df[['Crop type', 'Season', 'District', 'Annual rainfall (mm)', 'Pesticide and fertilizer used (kg/hectare)']]
y = df['Yield (tonnes/hectare)']

# --- FEATURE ENGINEERING ---
# Create an interaction term between rainfall and pesticide/fertilizer use
print("Performing feature engineering...")
X = X.copy()
X['Rainfall_Pesticide'] = X['Annual rainfall (mm)'] * X['Pesticide and fertilizer used (kg/hectare)']

# --- PREPROCESSING ---
# One-Hot Encode categorical columns
print("Applying One-Hot Encoding for categorical features...")
ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
X_cat_encoded = ohe.fit_transform(X[['Crop type', 'Season', 'District']])
joblib.dump(ohe, 'models/yield_ohe_encoder.pkl')
print("Saved One-Hot Encoder to 'models/yield_ohe_encoder.pkl'")

# Robust Scaling for numeric features
print("Applying Robust Scaling for numerical features...")
scaler = RobustScaler()
numeric_cols = ['Annual rainfall (mm)', 'Pesticide and fertilizer used (kg/hectare)', 'Rainfall_Pesticide']
X_num_scaled = scaler.fit_transform(X[numeric_cols])
joblib.dump(scaler, 'models/yield_scaler.pkl')
print("Saved Scaler to 'models/yield_scaler.pkl'")

# Combine encoded categorical and scaled numerical features
print("Combining preprocessed features...")
X_final = np.hstack([X_cat_encoded, X_num_scaled])

# --- MODEL TRAINING ---
# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_final, y, test_size=0.2, random_state=42)
print(f"Data split into training ({X_train.shape[0]} samples) and testing ({X_test.shape[0]} samples) sets.")

# Hyperparameter tuning grid for RandomizedSearchCV
param_grid = {
    'n_estimators': [100, 200, 300, 500],
    'max_depth': [6, 8, 10, 12, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 0.8] # 'auto' is deprecated
}

rf = RandomForestRegressor(random_state=42, n_jobs=-1)

# Randomized Search Cross-Validation
print("\nStarting Hyperparameter Tuning with RandomizedSearchCV...")
grid_search = RandomizedSearchCV(
    estimator=rf,
    param_distributions=param_grid,
    n_iter=50, # Increased iterations for better search
    scoring='r2',
    cv=5,
    verbose=1, # Reduced verbosity for cleaner output
    random_state=42
)

grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
print("Hyperparameter tuning complete.")

# --- MODEL EVALUATION ---
print("\n--- Model Evaluation ---")
y_pred = best_model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Cross-validation R² mean
cv_scores = cross_val_score(best_model, X_final, y, cv=5, scoring='r2')
cv_mean = cv_scores.mean()

# --- SAVE THE FINAL MODEL ---
joblib.dump(best_model, 'models/crop_yield_model.pkl')
print("Saved final trained model to 'models/crop_yield_model.pkl'")

# --- PRINT RESULTS ---
print("\n--- Final Results ---")
print(f"Random Forest Model Evaluation:")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"R² Score (on test set): {r2:.4f}")
print(f"Cross-Validated R² Mean: {cv_mean:.4f}")
print(f"\nBest Hyperparameters Found:")
print(grid_search.best_params_)
print("\n--- Training Script Finished ---")
