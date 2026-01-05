📌 Project Overview

This project implements a complete end-to-end machine learning regression pipeline to predict residential house prices based on property attributes.
The solution is built using the Kaggle House Prices: Advanced Regression Techniques dataset and follows industry best practices in data preprocessing, model evaluation, and prediction.

The objective is to accurately estimate house prices by learning relationships between numerical and categorical housing features such as property quality, living area, construction year, and neighborhood characteristics.

📂 Dataset

Source: Kaggle – House Prices: Advanced Regression Techniques

The project utilizes two datasets:


Download link:

https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data
train.csv



Contains historical housing data with the target variable SalePrice

Used for exploratory data analysis, model training, and validation

test.csv

Contains unseen housing records without SalePrice

Used for generating final price predictions

Dataset Characteristics

1,460 training observations

81 total features

Combination of numerical and categorical variables

Target variable exhibits a right-skewed distribution

⚙️ Technologies & Libraries

Python

Pandas – data manipulation and analysis

NumPy – numerical computation

Matplotlib & Seaborn – data visualization

Scikit-learn – preprocessing, modeling, and evaluation

🔄 Machine Learning Workflow
1. Data Loading & Inspection

Dataset shape analysis

Data type validation

Missing value assessment

2. Exploratory Data Analysis (EDA)

Target variable distribution analysis

Correlation analysis with key predictors

Visualization of missing value patterns

3. Data Preprocessing

Median imputation for numerical features

Most-frequent imputation for categorical features

One-hot encoding of categorical variables

Unified preprocessing using ColumnTransformer

4. Feature Engineering

Log transformation of the target variable to reduce skewness and stabilize variance

5. Model Training

Baseline model: Linear Regression

Ensemble models:

Random Forest Regressor

Gradient Boosting Regressor

6. Model Evaluation

Performance comparison using RMSE

Final evaluation conducted on the original price scale

7. Final Model Selection

Random Forest Regressor selected based on superior validation performance

8. Prediction on Unseen Data

Predictions generated for the test dataset

Results saved for submission or deployment

📊 Evaluation Metrics

Root Mean Squared Error (RMSE)
Measures the average magnitude of prediction error

R² Score
Evaluates the proportion of variance explained by the model (optional extension)

📁 Output Files
figures/

Sale price distribution

Correlation heatmap

Missing value analysis

Feature importance visualization

Actual vs. predicted price plot

house_price_model.pkl

Serialized trained machine learning model

test_predictions.csv

Predicted house prices for unseen test data

📌 Key Insights

Property quality (OverallQual) and living area (GrLivArea) are the most influential predictors

Log transformation significantly improves model stability and performance

Ensemble models outperform the linear regression baseline

Pipeline-based preprocessing effectively prevents data leakage and improves generalization

🚀 Future Enhancements

Hyperparameter tuning using GridSearchCV or RandomizedSearchCV

Cross-validation for more robust performance estimation

Integration of advanced models (XGBoost, LightGBM, CatBoost)

Model deployment using Flask or FastAPI

Development of a web-based user interface

📜 License

This project is intended for educational and portfolio purposes.
