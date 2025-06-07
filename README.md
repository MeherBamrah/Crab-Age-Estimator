# Crab-Age-Estimator

Competition: Kaggle Weekly ML Challenge 2 (https://www.kaggle.com/competitions/weekly-ml-challenge-2 )
Final Score: 1.30

This repository contains my solution for the Kaggle Weekly ML Challenge 2. The objective of the competition was to Predict house prices based on various features based on the given Dataset.

Dataset
* The dataset was provided by Kaggle as part of the competition.
* The dataset for this competition (both train and test) was generated from a deep learning model trained on the Crab Age Prediction dataset. Feature distributions are close to, but not exactly the same, as the original.
* Files included -
    1 train.csv - the training dataset; Age is the target
    2 test.csv - the test dataset; your objective is to predict the probability of Age (the ground truth is int but you can predict int or float)
    3 sample_submission.csv - a sample submission file in the correct format

Approach
1. Data Handling: Automated download and extraction of Kaggle competition data using the Kaggle API, with support for both local and Colab environments.

2. Feature Engineering: Extensive domain-inspired features (ratios, proxy variables, interactions, squared terms), one-hot encoding for categorical variables, and polynomial feature expansion (degree 2) to capture non-linear effects. Robust missing value handling.

3. Modeling: Used XGBoost Regressor with GPU acceleration. Hyperparameters were optimized via Optuna with 5-fold cross-validation (RMSE metric).

4. Submission: Final predictions were clipped and rounded as required, then saved in the competition format for submission.
