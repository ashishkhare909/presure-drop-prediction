import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import joblib


# Load CFD simulation dataset from CSV
# Replace with your actual file name/path if needed
data_file = "cfd_simulation_data.csv"
df = pd.read_csv(data_file)

# Define input features (X) and output target (y)
feature_columns = ["pipe_diameter", "pipe_length", "flow_rate", "inlet_pressure"]
target_column = "pressure_drop"

X = df[feature_columns]
y = df[target_column]

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Random Forest regressor
rf_model = RandomForestRegressor(n_estimators=200, random_state=42)
rf_model.fit(X_train, y_train)

# Predict and evaluate Random Forest model
rf_predictions = rf_model.predict(X_test)
rf_r2 = r2_score(y_test, rf_predictions)
print(f"Random Forest R^2 score: {rf_r2:.4f}")

# Optional baseline comparison: Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

lr_predictions = lr_model.predict(X_test)
lr_r2 = r2_score(y_test, lr_predictions)
print(f"Linear Regression R^2 score: {lr_r2:.4f}")

# Save trained Random Forest model for later use
model_file = "pressure_drop_rf_model.joblib"
joblib.dump(rf_model, model_file)
print(f"Model saved to: {model_file}")

# Save useful metadata for reproducibility in simple projects
metadata = {
    "feature_columns": feature_columns,
    "target_column": target_column,
    "train_size": int(X_train.shape[0]),
    "test_size": int(X_test.shape[0]),
    "random_state": 42,
    "numpy_version": np.__version__,
    "pandas_version": pd.__version__,
}
joblib.dump(metadata, "pressure_drop_model_metadata.joblib")
