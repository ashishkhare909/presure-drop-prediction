import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
import joblib

# =========================
# 1. Load Dataset
# =========================
data_file = "ML Datasheet for simulation.csv"
df = pd.read_csv(data_file)

# =========================
# 2. Feature Engineering (VERY IMPORTANT for CFD)
# =========================

# Velocity = flow_rate / area
df["velocity"] = df["flow_rate"] / (np.pi * (df["pipe_diameter"] / 2) ** 2)

# Length to diameter ratio
df["L_by_D"] = df["pipe_length"] / df["pipe_diameter"]

# =========================
# 3. Define Features & Target
# =========================
feature_columns = [
    "pipe_diameter",
    "pipe_length",
    "flow_rate",
    "inlet_pressure",
    "velocity",
    "L_by_D"
]

target_column = "pressure_drop"

X = df[feature_columns]
y = df[target_column]

# =========================
# 4. Train-Test Split
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# 5. Random Forest (No Scaling Needed)
# =========================
rf_model = RandomForestRegressor(
    n_estimators=50,       # reduced for small dataset
    max_depth=3,           # prevents overfitting
    random_state=42
)

rf_model.fit(X_train, y_train)

rf_predictions = rf_model.predict(X_test)
rf_r2 = r2_score(y_test, rf_predictions)

print(f"Random Forest R^2 score: {rf_r2:.4f}")

# =========================
# 6. Linear Regression with Scaling (Pipeline)
# =========================
lr_pipeline = Pipeline([
    ('scaler', RobustScaler()),
    ('model', LinearRegression())
])

lr_pipeline.fit(X_train, y_train)

lr_predictions = lr_pipeline.predict(X_test)
lr_r2 = r2_score(y_test, lr_predictions)

print(f"Linear Regression R^2 score (scaled): {lr_r2:.4f}")

# =========================
# 7. Cross Validation (Better for small data)
# =========================
rf_cv_scores = cross_val_score(rf_model, X, y, cv=5)
lr_cv_scores = cross_val_score(lr_pipeline, X, y, cv=5)

print(f"RF CV Mean R^2: {rf_cv_scores.mean():.4f}")
print(f"LR CV Mean R^2: {lr_cv_scores.mean():.4f}")

# =========================
# 8. Save Models
# =========================
joblib.dump(rf_model, "pressure_drop_rf_model.joblib")
joblib.dump(lr_pipeline, "pressure_drop_lr_pipeline.joblib")

print("Models saved successfully.")

# =========================
# 9. Save Metadata
# =========================
metadata = {
    "feature_columns": feature_columns,
    "target_column": target_column,
    "train_size": int(X_train.shape[0]),
    "test_size": int(X_test.shape[0]),
    "random_state": 42,
    "notes": "Includes feature engineering (velocity, L/D) and scaling via pipeline",
    "numpy_version": np.__version__,
    "pandas_version": pd.__version__,
}

joblib.dump(metadata, "pressure_drop_model_metadata.joblib")

print("Metadata saved successfully.")
