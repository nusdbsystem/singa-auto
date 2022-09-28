# Singa-Auto Demo - Battery Capacity Estimator.

This folder contains a number of models for estimating the capacity of rechargeable batteries in mobile phones, electric vehicles, etc.

## Dataset Preparation

The training and evaluation data should be compressed into a single .zip file. The zip file contains two CSV files: train.csv and valid.csv. Both CSV files are in the following format:

The first line is the header, which must be:

Time,Voltage_measured,Current_measured,Temperature_measured,Current_load,Voltage_load,Capacity

From the second line, the data is split into many blocks. Each block starts from a '\<Start of Discharging\>' line and ends at an '\<End of Discharging\>' line. Each block contains data during a single round of discharging. Each line is a data point. For example:

\<Start of Discharging\>  
0.00,4.19,-0.000261,24.3,0.000600,0.00,1.36  
9.37,4.19,0.000523,24.3,0.000800,4.21,1.36  
19.51,3.97,-2.01,24.3,1.99,2.96,1.36  
28.82,3.95,-2.01,24.4,1.99,2.95,1.36  
...  
2873.25,3.58,-0.003372,35.0,0.0006,0.00,1.36  
2883.01,3.58,-0.002502,34.9,0.0006,0.00,1.36  
\<End of Discharging\>  

## Prediction/Inference

A query file is a CSV file that contains some historical discharging data. The first line is the header, which must be:

Time,Voltage_measured,Current_measured,Temperature_measured,Current_load,Voltage_load

From the second line, the data is split into many blocks. Each block starts from a '\<Start of Discharging\>' line and ends at an '\<End of Discharging\>' line. Each block contains data during a single round of discharging. Each line is a data point. For example:

\<Start of Discharging\>  
0.00,4.19,-0.000261,24.3,0.000600,0.00  
9.37,4.19,0.000523,24.3,0.000800,4.21  
19.51,3.97,-2.01,24.3,1.99,2.96  
28.82,3.95,-2.01,24.4,1.99,2.95  
...  
2873.25,3.58,-0.003372,35.0,0.0006,0.00  
2883.01,3.58,-0.002502,34.9,0.0006,0.00  
\<End of Discharging\>  

Given a query CSV file, the model returns an estimated capacity.

## Model Description

There are two models:
1. RFBatteryCapacityEstimator.py, which is a random forest. It will perform auto parameter tuning on the maximum depth of decision trees, the number of estimators and feature selection criteria.
2. MLPBatteryCapacityEstimator.py, which is a feedforward neural network. It will perform auto parameter tuning on the number of hidden layers and hidden units.

Their usages are almost the same. However, the random forest is usually much faster, whereas the feedforward neural network is more robust.

