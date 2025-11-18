import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load the updated dataset
df = pd.read_csv("real_estate_dataset_inr.csv")

# Prepare features and target
X = df.drop("Price(INR)", axis=1)
y = df["Price(INR)"]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train models
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)

tree_reg = DecisionTreeRegressor(random_state=42)
tree_reg.fit(X_train, y_train)

forest_reg = RandomForestRegressor(n_estimators=100, random_state=42)
forest_reg.fit(X_train, y_train)

# Streamlit app
st.title("Advanced Real Estate Valuation")
st.write("## Predict Property Price Using Trained Models")

# User input
area = st.slider("Area (sqft)", 500, 3000, 1500)
bedrooms = st.slider("Bedrooms", 1, 5, 3)
bathrooms = st.slider("Bathrooms", 1, 3, 2)
age = st.slider("Age of Property (years)", 0, 30, 10)
location_score = st.slider("Location Score (1-10)", 1.0, 10.0, 5.0, step=0.1)
pet_friendly = st.selectbox("Pet Friendly", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
parking_spaces = st.slider("Parking Spaces", 0, 3, 1)
amenities_score = st.slider("Nearby Amenities Score (1-10)", 1.0, 10.0, 5.0, step=0.1)

input_data = pd.DataFrame({
    "Area(sqft)": [area],
    "Bedrooms": [bedrooms],
    "Bathrooms": [bathrooms],
    "Age(years)": [age],
    "Location_Score": [location_score],
    "Pet_Friendly": [pet_friendly],
    "Parking_Spaces": [parking_spaces],
    "Amenities_Score": [amenities_score]
})

# Predictions
lin_pred = lin_reg.predict(input_data)[0]
tree_pred = tree_reg.predict(input_data)[0]
forest_pred = forest_reg.predict(input_data)[0]

# Display results
st.write("### Predicted Prices (in ₹):")
st.write(f"- Linear Regression: ₹{lin_pred:,.2f}")
st.write(f"- Decision Tree: ₹{tree_pred:,.2f}")
st.write(f"- Random Forest (Ensemble): ₹{forest_pred:,.2f}")

# Evaluation
st.write("## Model Evaluation (on test set)")
st.write(f"- Linear Regression RMSE: ₹{np.sqrt(mean_squared_error(y_test, lin_reg.predict(X_test))):,.2f}")
st.write(f"  - R² Score: {r2_score(y_test, lin_reg.predict(X_test)):.2f}")

st.write(f"- Decision Tree RMSE: ₹{np.sqrt(mean_squared_error(y_test, tree_reg.predict(X_test))):,.2f}")
st.write(f"  - R² Score: {r2_score(y_test, tree_reg.predict(X_test)):.2f}")

st.write(f"- Random Forest RMSE: ₹{np.sqrt(mean_squared_error(y_test, forest_reg.predict(X_test))):,.2f}")
st.write(f"  - R² Score: {r2_score(y_test, forest_reg.predict(X_test)):.2f}")

