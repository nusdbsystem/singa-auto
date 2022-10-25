# Singa-Auto Demo - Wind Power Predictor.

This folder contains model(s) for predicting wind speed based on historical data.

## Dataset Preparation

The training and evaluation data should be CSV files. Both CSV files are in the following format:

The first line is the header, which must contain an identifier "Wind Speed (km/h)" that indicates the column of wind speed data.

From the second line, each line is a data point. The data in the "Wind Speed (km/h)" column should be real numbers indicating the wind speed. The wind speed data should be collected in a consistent time interval, e.g., one data point per hour. The data in other columns are not used.

## Prediction/Inference

The format of query data should be the same as the training and evaluation data. Given a query file in CSV, the model can predict future wind speeds.

## Model Description

There are two models. Their usages are almost the same.
1. RFWindPowerPredictor.py, which is a random forest. It will perform auto parameter tuning on the length of sequential data, the maximum depth of decision trees, the number of estimators and feature selection criteria.
2. MLPWindPowerPredictor.py, which is a feedforward neural network. It will perform auto parameter tuning on the length of sequential data, the number of hidden layers and hidden units.

