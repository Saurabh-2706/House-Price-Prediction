import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# Title
st.title("🏠 House Price Prediction App with AI Insights 🤖")
st.markdown("Predict house prices using **Linear Regression** and get **AI-powered insights**.")

# Load dataset
df = pd.read_csv("Housing.csv")
st.write("### Dataset Preview", df.head())

# Convert categorical columns to numeric
df = pd.get_dummies(df, drop_first=True)

# Define features and target
X = df.drop('price', axis=1)
y = df['price']

# Normalize numeric data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluation Metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

st.write(f"**Model Performance:**")
st.write(f"- Mean Squared Error: `{mse:.2f}`")
st.write(f"- R² Score: `{r2:.2f}`")

# --- AI Feature: Intelligent Insights ---
st.subheader("🤖 AI-Powered Insights")

coefficients = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_
}).sort_values(by="Coefficient", ascending=False)

top_positive = coefficients.head(3)
top_negative = coefficients.tail(3)

ai_summary = f"""
### 🔍 AI Insights Summary
- The top factors **increasing** house price are: **{', '.join(top_positive['Feature'].tolist())}**.
- The top factors **decreasing** house price are: **{', '.join(top_negative['Feature'].tolist())}**.
- The model explains about **{r2*100:.1f}%** of the variance in house prices.
- Features with higher coefficients have a stronger impact on price.
"""

st.markdown(ai_summary)

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
    val = st.number_input(f"Enter value for {col}", float(X[col].min()), float(X[col].max()), float(X[col].mean()))
    input_data[col] = val

if st.button("Predict Price"):
    input_df = pd.DataFrame([input_data])
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)[0]
    st.success(f"💰 Estimated House Price: **{int(prediction)}**")

    # AI contextual explanation
    st.markdown(f"🧠 *AI Insight:* Based on your inputs, features like "
                f"**{top_positive.iloc[0]['Feature']}** and **{top_positive.iloc[1]['Feature']}** "
                f"likely contributed to the higher predicted price.*")
