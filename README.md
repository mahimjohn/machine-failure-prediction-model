# Predictive Maintenance for Industrial Equipment

## Project Overview

This project implements an AI-based predictive maintenance system for industrial equipment.

The system analyzes equipment sensor readings and uses a trained Machine Learning model to estimate the probability of equipment failure. Based on the predicted risk, the system provides maintenance recommendations and generates alerts for high-risk conditions.

## Objectives

- Predict potential equipment failures before they occur.
- Monitor equipment health using sensor readings.
- Estimate equipment failure probability.
- Classify equipment into different risk levels.
- Generate maintenance recommendations.
- Provide alerts for high-risk equipment.
- Visualize equipment health through an interactive dashboard.

## Dataset

The project uses the AI4I 2020 Predictive Maintenance Dataset.

The dataset contains industrial equipment sensor measurements including:

- Machine Type
- Air Temperature
- Process Temperature
- Rotational Speed
- Torque
- Tool Wear

The target variable is:

- Machine Failure

Failure-mode indicator columns were excluded from the primary model because they are closely related to the failure target and could introduce target leakage.

## Machine Learning Model

A Random Forest Classifier was developed for equipment failure prediction.

The preprocessing pipeline includes:

1. Selecting relevant equipment features.
2. Encoding the machine type using one-hot encoding.
3. Splitting the data into training and testing sets using stratification.
4. Scaling numerical sensor features.
5. Training the Random Forest model.
6. Saving the trained model and scaler for use by the monitoring application.

## Monitoring System

The monitoring system accepts current equipment sensor readings and:

1. Preprocesses the incoming readings.
2. Generates a failure probability.
3. Predicts whether equipment failure is likely.
4. Assigns a risk level.
5. Generates a maintenance recommendation.
6. Generates an alert when failure risk is high.

## Risk Levels

The monitoring system uses failure probability to classify equipment risk.

### LOW

Normal monitoring is recommended.

### MEDIUM

The equipment should receive increased monitoring and maintenance attention.

### HIGH

Immediate maintenance inspection is recommended. Equipment load reduction may also be considered when operational procedures permit.

## Dashboard

The project includes an interactive Streamlit dashboard.

The dashboard allows users to enter:

- Machine Type
- Air Temperature
- Process Temperature
- Rotational Speed
- Torque
- Tool Wear

The dashboard displays:

- Failure Probability
- Predicted Failure
- Risk Level
- Maintenance Recommendation
- Current Sensor Readings
- Model Feature Importance

## Project Structure

```text
predictive-maintenance/
│
├── dashboard/
│   └── app.py
│
├── data/
│   ├── raw/
│   │   └── ai4i2020.csv
│   │
│   └── processed/
│       ├── X_train.csv
│       ├── X_test.csv
│       ├── y_train.csv
│       └── y_test.csv
│
├── models/
│   ├── random_forest_model.joblib
│   └── scaler.joblib
│
├── notebooks/
│   ├── 01_data_understanding.ipynb
│   ├── 02_data_preprocessing.ipynb
│   ├── 03_model_development.ipynb
│   └── 04_monitoring_and_alerts.ipynb
│
├── reports/
│
├── src/
│
├── tests/
│
├── .gitignore
├── README.md
└── requirements.txt