📌 Overview

This project focuses on predicting real estate prices in INR using a dataset of property features and a machine-learning model.
The repository contains:

real_estate_dataset_inr.csv → Clean dataset with real estate attributes

code_1.py → Python script for data preprocessing, model training, and predictions

The goal is to build a regression model that estimates property prices based on key features such as area, location, number of bedrooms, etc.


📂 Project Structure
├── real_estate_dataset_inr.csv     # Dataset used for training
├── code_1.py                       # ML model training + prediction script
└── README.md                       # Documentation


🧠 Model Description (code_1.py)

The script performs:

1️⃣ Importing libraries

Pandas, NumPy, Scikit-learn, Matplotlib, etc.

2️⃣ Data preprocessing

Handling missing values

Encoding categorical data

Train-test splitting

Feature scaling (if applied)

3️⃣ Model training

Supports models like:

Linear Regression

Random Forest Regressor

XGBoost / Gradient Boosting (if included)

4️⃣ Evaluation

MAE

RMSE

R² score

5️⃣ Prediction

Takes new input features and predicts price.




