import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Title
st.title("🏠 House Price Prediction App")
st.markdown("Predict house prices using **Linear Regression** with interactive graphs.")

# Load dataset
df = pd.read_csv("Housing.csv")
st.write("### Dataset Preview", df.head())

# Convert categorical columns to numeric
df = pd.get_dummies(df, drop_first=True)

# Define features and target
X = df.drop('price', axis=1)
y = df['price']

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Plot 1: Actual vs Predicted Prices
st.subheader("📈 Actual vs Predicted Prices")
fig1, ax1 = plt.subplots(figsize=(6, 4))
ax1.scatter(y_test, y_pred, color='blue', alpha=0.6)
ax1.set_xlabel("Actual Price")
ax1.set_ylabel("Predicted Price")
ax1.set_title("Actual vs Predicted House Prices")
axis_range = np.linspace(min(y_test), max(y_test), 100)
ax1.plot(axis_range, axis_range, color="red", linestyle="--")
st.pyplot(fig1)

# Plot 2: Distribution of House Prices
st.subheader("📊 Distribution of House Prices")
fig2, ax2 = plt.subplots(figsize=(6, 4))
sns.histplot(y, kde=True, color="green", ax=ax2)
ax2.set_title("Distribution of House Prices")
st.pyplot(fig2)

# Plot 3: Residuals
st.subheader("📉 Residual Plot")
residuals = y_test - y_pred
fig3, ax3 = plt.subplots(figsize=(6, 4))
ax3.scatter(y_pred, residuals, color='purple', alpha=0.6)
ax3.axhline(y=0, color='red', linestyle='--')
ax3.set_xlabel("Predicted Price")
ax3.set_ylabel("Residuals")
ax3.set_title("Residual Plot")
st.pyplot(fig3)

# Plot 4: Feature Correlation Heatmap
st.subheader("🧩 Feature Correlation Heatmap")
fig4, ax4 = plt.subplots(figsize=(8, 6))
sns.heatmap(df.corr(), annot=False, cmap="coolwarm", ax=ax4)
ax4.set_title("Correlation Heatmap")
st.pyplot(fig4)

# Allow user to enter custom input
st.subheader("🔮 Predict Price for Custom Input")
input_data = {}
for col in X.columns:
    val = st.number_input(f"Enter value for {col}", int(X[col].min()), int(X[col].max()), int(X[col].mean()))
    input_data[col] = val

if st.button("Predict Price"):
    input_df = pd.DataFrame([input_data])
    prediction = model.predict(input_df)[0]
    st.success(f"Estimated House Price: {int(prediction)}")
