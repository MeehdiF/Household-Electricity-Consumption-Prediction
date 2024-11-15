# Household-Electricity-Consumption-Prediction

Project Overview

This project focuses on predicting household electricity consumption using Linear Regression. The goal is to predict the “Global Active Power” based on historical data of other power and voltage features. This project demonstrates skills in time series data processing and regression modeling.

Dataset

The dataset used is the Household Electric Power Consumption dataset from the UCI Machine Learning Repository. It contains features like voltage, global intensity, and different types of sub-metered power consumption readings.

Key Features

The main features used in this project include:
 • Global Reactive Power
 • Voltage
 • Global Intensity
 • Sub-metering values (Sub_metering_1, Sub_metering_2, Sub_metering_3)

The target variable is Global Active Power, representing the primary household power usage.

Steps in the Project

Step 1: Import Libraries

Libraries like pandas, NumPy, matplotlib, and scikit-learn are imported to handle data manipulation, modeling, and visualization.

Step 2: Data Import and Cleaning

 • The dataset is imported, and unnecessary columns (Date and Time) are removed.
 • Missing or invalid values are replaced with 0, and all columns are converted to float type for modeling.

Step 3: Split the Data

Data is split into training and testing sets (80-20 split) to assess the model’s accuracy on unseen data.

Step 4: Model Creation

Linear Regression is used as the model for initial experimentation.

Step 5: Model Training

The model is trained on the training dataset, and the coefficients are printed.

Step 6: Prediction and Evaluation

 • Predictions are made on the test set.
 • Evaluation metrics include the Residual Sum of Squares and the Variance Score.

Results

The linear regression model provides a baseline prediction of household electricity consumption. Future improvements could involve exploring LSTM or other time series models for better accuracy in predictions over time.

Requirements

 • Python 3.x
 • Libraries: pandas, NumPy, scikit-learn, matplotlib
