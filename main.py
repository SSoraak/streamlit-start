import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# Load the data
data_path = 'C:\Users\LOQ\OneDrive\Pose Health Care\MA (PPop)\สถิติ Pose Repairman.xlsx'  
sheet_name = 'ข้อมูลการใช้นำยา'
df = pd.read_excel(data_path, sheet_name=sheet_name)

# Preprocess the data (convert to numeric)
df['ระยะเวลาในใช้น้ำยา /แบต (วัน)'] = pd.to_numeric(df['ระยะเวลาในใช้น้ำยา /แบต (วัน)'], errors='coerce')

# Streamlit app
st.title("Machinery Maintenance Information and Prediction")

# Display the data
st.header("Maintenance Records")
st.table(df)

# Select features and target variable for the model
X = df[['หมายเลขเครื่อง']]
y = df['ระยะเวลาในใช้น้ำยา /แบต (วัน)']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Model evaluation
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

# Display model performance
st.header("Model Performance")
st.write(f"Mean Absolute Error (MAE): {mae}")
st.write(f"Mean Squared Error (MSE): {mse}")
st.write(f"Root Mean Squared Error (RMSE): {rmse}")

# Predict for a new input
st.header("Predict Time Until Maintenance Issue")
machine_id = st.number_input("Enter Machine ID:", min_value=int(df['หมายเลขเครื่อง'].min()), max_value=int(df['หมายเลขเครื่อง'].max()))
if st.button("Predict"):
    prediction = model.predict([[machine_id]])
    st.write(f"Predicted Time Until Maintenance Issue: {prediction[0]:.2f} days")

# Filter by machine type
st.sidebar.header("Filter by Machine Type")
selected_machine_type = st.sidebar.selectbox("Select Machine Type", df['แผนก'].unique())
filtered_data = df[df['แผนก'] == selected_machine_type]
st.header(f"Records for Machine Type: {selected_machine_type}")
st.table(filtered_data)
