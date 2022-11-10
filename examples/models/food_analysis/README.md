# Singa-Auto Demo - Food Recommendation.

This folder contains a number of models for food recommendation with knowledge graphs.

## Dataset Preparation

The training and evaluation data should be compressed into a single .tar file. The two tar files contain a "training_data" folder and an "evaluation_data", respectively.

The "training_data" folder has the following files:

(1) food_knowledge_base.tri: It is a knowledge base containing triples of the form "\<subject\> \<predicate\> \<object\>". For example:

milk contain protein  
milk contain vitamin_a  
protein is_good_for brain  
...  
prawn contain protein  

(2) tag_list.txt: It contains N prediction tags. For example:

pregnant_tag  
puerpera_tag  
lactation_tag  
baby_tag  

(3) N files named "\<tag_name\>_training.txt". Each file contains training data of the form "\<entity\> \<class\>", which indicates the class of the entity with respect to the tag. For example, let Class 0 denote food suitable for a baby, and Class 1 denote food not suitable for a baby, we have a file named "baby_tag_training.txt":

milk 0  
prawn 1  
...  
orange 0  

The data is used to train classifiers predicting the probability that a given entity belongs to each class. Trained classifiers are evaluated using the "evaluation_data" folder that has N files named "\<tag_name\>_evaluation.txt". The format of evaluation data is the same as training data.

## Prediction/Inference

A query should be a Python list of strings. Each string is of the form "[\<entity\>, \<tag\>]". For example:

[str(["milk", "baby_tag"]), str(["prawn", "baby_tag"]), str(["orange", "baby_tag"])]

The model will return the probability that a given entity belongs to each class with respect to a given tag.

## Model Description

There are two models:
1. RFFoodRecommendationModel.py, which is a random forest. It performs auto parameter tuning on the number of estimators.
2. MLPFoodRecommendationModel.py, which is a feedforward neural network. It performs auto parameter tuning on the number of hidden layers and hidden units.

