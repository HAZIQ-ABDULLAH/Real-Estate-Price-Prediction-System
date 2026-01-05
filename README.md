# 🏠 House Price Prediction using Machine Learning

## 📌 Project Overview
This project predicts residential house prices using supervised machine learning techniques.
It demonstrates a complete end-to-end ML workflow including data preprocessing, feature
engineering, model training, evaluation, and interpretation.

## 🎯 Problem Statement
Predict house prices based on property characteristics such as quality, area, location,
and construction details.

## 📂 Dataset
- **Name:** House Prices – Advanced Regression Techniques
- **Source:** Kaggle
- **Samples:** 1460
- **Features:** 80 input features (numerical + categorical)
- **Target Variable:** SalePrice

## ⚙️ Preprocessing
- Missing value imputation (median & most frequent)
- One-hot encoding for categorical features
- Log transformation applied to target variable to reduce skewness
- Unified preprocessing using Scikit-learn pipelines

## 🤖 Models Implemented
- Linear Regression (Baseline)
- Random Forest Regressor (Final Model)
- Gradient Boosting Regressor

## 📊 Evaluation Metrics
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- Diagnostic visualization (Actual vs Predicted)

## 🔍 Key Insights
- OverallQual and GrLivArea are the strongest predictors
- Tree-based models outperform linear models
- Log transformation improves model stability and performance

## 📈 Visual Outputs
All plots are saved in the `figures/` directory:
- Target distribution
- Correlation heatmap
- Missing values
- Feature importance
- Actual vs predicted prices

## 💾 Model Saving
The trained model is saved using `joblib` for reuse and deployment.

## 🚀 Conclusion
The final Random Forest model demonstrates strong predictive performance and can be extended
for real-world real estate price estimation systems.
