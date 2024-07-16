import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import OneHotEncoder
import numpy as np

# Load the data
data_path = r'C:\Users\LOQ\OneDrive\Pose Health Care\MA (PPop)\สถิติ Pose Repairman.xlsx'  
sheet_name = 'ข้อมูลการใช้นำยา'
df = pd.read_excel(data_path, sheet_name=sheet_name)

# Preprocess the data (convert to numeric)
df['ระยะเวลาในใช้น้ำยา /แบต (วัน)'] = pd.to_numeric(df['ระยะเวลาในใช้น้ำยา /แบต (วัน)'], errors='coerce')

# Streamlit app
st.title("Machinery Maintenance Information and Prediction")

# Display the data
st.header("Maintenance Records")
st.table(df.head(10))

# Define column names
department_column = 'แผนก'
machine_id_column = 'หมายเลขเครื่อง'
issue_column = 'ปัญหา'
maintenance_duration_column = 'ระยะเวลาในใช้น้ำยา /แบต (วัน)'

# One-hot encode the categorical features
encoder = OneHotEncoder(sparse_output=False)
encoded_features = encoder.fit_transform(df[[department_column, issue_column]])

# Create a DataFrame with the encoded features
encoded_feature_names = encoder.get_feature_names_out([department_column, issue_column])
encoded_df = pd.DataFrame(encoded_features, columns=encoded_feature_names)

# Combine the encoded features with the numeric feature
X = pd.concat([df[[machine_id_column]], encoded_df], axis=1)
y = df[maintenance_duration_column]

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
machine_id = st.number_input("Enter Machine ID:", min_value=int(df[machine_id_column].min()), max_value=int(df[machine_id_column].max()))
selected_department = st.selectbox("Select Department", df[department_column].unique())
selected_issue = st.selectbox("Select Issue", df[issue_column].unique())

# Prepare the input data for prediction
input_data = pd.DataFrame([[machine_id, selected_department, selected_issue]], columns=[machine_id_column, department_column, issue_column])
input_encoded = encoder.transform(input_data[[department_column, issue_column]])
input_encoded_df = pd.DataFrame(input_encoded, columns=encoded_feature_names)
input_final = pd.concat([input_data[[machine_id_column]], input_encoded_df], axis=1)

if st.button("Predict"):
    prediction = model.predict(input_final)
    st.write(f"Predicted Time Until Maintenance Issue: {prediction[0]:.2f} days")

# Filter by machine type
st.sidebar.header("Filter by Machine Type")
selected_machine_type = st.sidebar.selectbox("Select Machine Type", df[department_column].unique())
filtered_data = df[df[department_column] == selected_machine_type]
st.header(f"Records for Machine Type: {selected_machine_type}")
st.table(filtered_data)
