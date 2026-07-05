# 📈Inflation Trend Forecasting using Time Series and Machine Learning
![Python](https://img.shields.io/badge/Python-3.13-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red?logo=streamlit)
![XGBoost](https://img.shields.io/badge/XGBoost-ML-success)
![Prophet](https://img.shields.io/badge/Prophet-Time%20Series-orange)
![SARIMA](https://img.shields.io/badge/SARIMA-Forecasting-purple)
![License](https://img.shields.io/badge/License-MIT-green)


## Project Overview
Inflation is one of the most significant economic indicators influencing the prices of essential commodities. This project analyzes historical price trends of vegetables, groceries, and fuel to identify inflation patterns and forecast future prices using statistical and machine learning techniques.
The project combines exploratory data analysis (EDA), feature engineering, seasonal decomposition, correlation analysis, inflation index calculation, and multiple forecasting models. The generated insights are presented through an interactive Streamlit dashboard, enabling users to explore historical trends, compare forecasting models, and better understand inflation behavior across different commodity categories.


## 🏗️ Project Architecture

```text
Historical Commodity Datasets
            │
            ▼
     Data Preprocessing
            │
            ▼
Exploratory Data Analysis (EDA)
            │
            ▼
Feature Engineering
            │
            ▼
Forecasting Models
(SARIMA • Prophet • XGBoost • Random Forest • Hybrid Models)
            │
            ▼
Model Evaluation
(MAE • RMSE • MAPE • SMAPE)
            │
            ▼
Interactive Streamlit Dashboard
```


## 🎯Objectives
- Analyze historical price trends of essential commodities.
- Study seasonal and inflation patterns across different commodity categories.
- Compare multiple forecasting models for price prediction.
- Evaluate model performance using standard forecasting metrics.
- Present insights through an interactive Streamlit dashboard.


## ✨Features
- Historical price trend analysis
- Seasonal decomposition (STL)
- Correlation and lag correlation analysis
- Inflation index visualization
- Feature engineering using lag and rolling statistics
- Multiple forecasting models
- Interactive Streamlit dashboard
- Model performance comparison using MAE, RMSE, MAPE, and SMAPE


## 🛠️Tech Stack

### 💻Programming Language
- Python

### Libraries & Frameworks
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Statsmodels
- Prophet
- XGBoost
- Streamlit

### Forecasting Models
- SARIMA
- Prophet
- Random Forest
- XGBoost
- Hybrid SARIMA + LSTM
- Rolling Walk-Forward SARIMA

### Evaluation Metrics
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- MAPE (Mean Absolute Percentage Error)
- SMAPE (Symmetric Mean Absolute Percentage Error)

## 📊Datasets
The project analyzes historical prices of essential commodities collected from multiple categories:

- 🥕 Vegetables
- 🛒 Grocery Items
- ⛽ Fuel

The datasets were cleaned, preprocessed, and transformed before analysis and forecasting.


## 🤖Forecasting Models

The following forecasting techniques were implemented and compared:

- Grid Search SARIMA
- Random Search SARIMA
- Prophet
- XGBoost Regressor
- Random Forest Regressor
- Hybrid SARIMA + LSTM
- Rolling Walk-Forward SARIMA

Each model was evaluated using standard forecasting metrics to identify the best-performing approach for different commodity categories.


![Dashboard](assets/dashboard_home.png)

![Dashboard](assets/dashboard_market.png)

![Dashboard](assets/dashboard_shopkeeper.png)

![Dashboard](assets/season_festive_analysis.png)

![Dashboard](assets/rolling_correlation.png)

![Dashboard](assets/model_accuracy_summary.png)

![Dashboard](assets/mape_heatmap.png)


![Dashboard](assets/forecast_example.png)


↓

Home Dashboard Image

↓

Market Overview Image

↓

Shopkeeper Dashboard

↓

Season Analysis

↓

Rolling Correlation

↓

Model Accuracy

↓

Heatmap

↓

Forecast Example
