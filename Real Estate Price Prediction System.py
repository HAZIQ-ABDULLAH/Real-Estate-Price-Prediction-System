# Generated from: Real Estate Price Prediction System.ipynb
# Converted at: 2026-01-05T18:05:45.326Z
# Next step (optional): refactor into modules & generate tests with RunCell
# Quick start: pip install runcell

# PROBLEM STATEMENT:
# Predict residential house prices based on property features
# using supervised machine learning regression techniques.

#Problem Type: Supervised Regression
#Target Variable: SalePrice
#Input Variables: Numerical + Categorical housing attributes


# =========================================================
# PROJECT TITLE:
# House Price Prediction using Machine Learning
# =========================================================


# =========================================================
# STEP 1: IMPORT LIBRARIES
# =========================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor

from sklearn.metrics import mean_squared_error

import os
os.makedirs("figures", exist_ok=True)

# =========================================================
# STEP 2: LOAD DATA
# =========================================================

DATA_PATH = r"C:\Users\LAPTOP INSIDE\Downloads\train.csv"
df = pd.read_csv(DATA_PATH)

print("Dataset Shape:", df.shape)
print(df.head())
print(df.info())



# Dataset contains 1460 observations and 81 features
# Mix of numerical (area, year built) and categorical (neighborhood, zoning)
# Target variable is right-skewed


# =========================================================
# STEP 3: EXPLORATORY DATA ANALYSIS (EDA)
# =========================================================

# Target distribution
plt.figure(figsize=(8,5))
sns.histplot(df['SalePrice'], bins=40, kde=True)
plt.title("Distribution of SalePrice")
plt.savefig("figures/saleprice_distribution.png", dpi=300, bbox_inches="tight")
plt.show()


# Correlation heatmap
important_features = [
    'SalePrice', 'OverallQual', 'GrLivArea',
    'TotalBsmtSF', 'GarageCars', 'YearBuilt'
]

plt.figure(figsize=(6,4))
sns.heatmap(df[important_features].corr(), annot=True, cmap='coolwarm')
plt.title("Correlation with SalePrice")
plt.savefig("figures/correlation_heatmap.png", dpi=300, bbox_inches="tight")
plt.show()

# Missing values
missing_ratio = df.isnull().mean()
missing_ratio = missing_ratio[missing_ratio > 0]

plt.figure(figsize=(10,5))
missing_ratio.sort_values(ascending=False).plot(kind='bar')
plt.title("Missing Value Ratio per Feature")
plt.savefig("figures/missing_values.png", dpi=300, bbox_inches="tight")
plt.show()
# Log transformation applied to reduce right skewness in target variable
df['SalePrice'] = np.log1p(df['SalePrice'])


# =========================================================
# STEP 4: FEATURE & TARGET SEPARATION
# =========================================================
X = df.drop(columns=['SalePrice'])
y = df['SalePrice']

numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X.select_dtypes(include=['object']).columns

# =========================================================
# STEP 5: PREPROCESSING PIPELINE
# =========================================================
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median'))
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# =========================================================
# STEP 6: TRAIN-TEST SPLIT
# =========================================================
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 80-20 split used to balance training data availability
# while preserving sufficient unseen validation samples



# =========================================================
# STEP 7: MODEL COMPARISON
# =========================================================

# Linear Regression used as baseline model
# to compare performance gains of ensemble models


models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(
        n_estimators=300,
        max_depth=20,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1
    ),
    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=3,
        random_state=42
    )
}

results = {}

for name, regressor in models.items():
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', regressor)
    ])
    
    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, preds))
    results[name] = rmse

print("\nModel Comparison (RMSE):")
for model_name, rmse in results.items():
    print(f"{model_name}: {rmse:.2f}")

# RMSE computed on log-transformed target
# Final model RMSE is later evaluated in original price units

#Random Forest showed lower RMSE and more stable performance.



# =========================================================
# STEP 8: FINAL MODEL (BEST ONE)
# =========================================================
final_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', RandomForestRegressor(
        n_estimators=300,
        max_depth=20,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1
    ))
])

# Increased number of trees improves generalization
# Depth limited to prevent overfitting
# Minimum leaf size smooths predictions



final_model.fit(X_train, y_train)

# Predictions in log scale
y_pred_log = final_model.predict(X_val)

# Convert back to original price scale
y_pred = np.expm1(y_pred_log)
y_val_actual = np.expm1(y_val)

rmse = np.sqrt(mean_squared_error(y_val_actual, y_pred))
print(f"\nFinal Model RMSE: {rmse:.2f}")

# =========================================================
# STEP 9: FEATURE IMPORTANCE
# =========================================================
rf_model = final_model.named_steps['model']

ohe = final_model.named_steps['preprocessor'] \
    .named_transformers_['cat'] \
    .named_steps['onehot']

categorical_feature_names = ohe.get_feature_names_out(categorical_features)

all_features = np.concatenate([
    numeric_features,
    categorical_feature_names
])

importance_df = pd.DataFrame({
    'Feature': all_features,
    'Importance': rf_model.feature_importances_
}).sort_values(by='Importance', ascending=False)

print("\nTop 15 Important Features:")
print(importance_df.head(15))

# OverallQual and GrLivArea are dominant predictors
# confirming real-world housing valuation principles


plt.figure(figsize=(8,6))
plt.barh(
    importance_df.head(15)['Feature'],
    importance_df.head(15)['Importance']
)
plt.gca().invert_yaxis()
plt.title("Top 15 Feature Importances")
plt.savefig("figures/feature_importance.png", dpi=300, bbox_inches="tight")
plt.show()

plt.scatter(y_val_actual, y_pred, alpha=0.5)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")

plt.title("Actual vs Predicted Prices")
plt.savefig("figures/actual_vs_predicted.png", dpi=300, bbox_inches="tight")
plt.show()


import joblib
joblib.dump(final_model, "house_price_model.pkl")


# =========================================================
# STEP 10: TEST DATA PREDICTION
# =========================================================
# Test dataset does not contain SalePrice
# Model predicts log-transformed prices
# Predictions are converted back to original scale



test_df = pd.read_csv(r"C:\Users\LAPTOP INSIDE\Downloads\test.csv")

print(test_df.shape)
test_df.head()


test_predictions_log = final_model.predict(test_df)
test_predictions = np.expm1(test_predictions_log)


submission = pd.DataFrame({
    "Id": test_df["Id"],
    "SalePrice": test_predictions
})

submission.to_csv("test_predictions.csv", index=False)



# CONCLUSION:
# Random Forest outperformed baseline models
# Preprocessing pipelines improved model robustness and generalization
# Model is suitable for real-world house price estimation