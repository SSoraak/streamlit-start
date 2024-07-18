import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt

# Load the data
data_path = 'https://github.com/SSoraak/streamlit-start/blob/main/%E0%B8%AA%E0%B8%96%E0%B8%B4%E0%B8%95%E0%B8%B4%20Pose%20Repairman.xlsx'
sheet_name = 'ข้อมูลการใช้นำยา'
df = pd.read_excel(data_path, sheet_name=sheet_name, engine='openpyxl' )

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

# Train the models
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42)
}

model_results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r_squared = r2_score(y_test, y_pred)
    model_results[name] = {"R Squared": r_squared, "MAE": mae, "MSE": mse, "RMSE": rmse}

# Time Series Forecasting using ARIMA
# Prepare the time series data
ts_data = df[['หมายเลขเครื่อง', 'ระยะเวลาในใช้น้ำยา /แบต (วัน)']].copy()
ts_data = ts_data.groupby('หมายเลขเครื่อง').sum()

# Train the ARIMA model
arima_model = ARIMA(ts_data, order=(5, 1, 0))
arima_model_fit = arima_model.fit()

# Make predictions
arima_forecast = arima_model_fit.forecast(steps=10)
arima_rmse = np.sqrt(mean_squared_error(ts_data.values, arima_model_fit.fittedvalues))


# Display model performance
st.header("Model Performance")
for name, metrics in model_results.items():
    st.write(f"**{name}**")
    st.write(f"R Squared: {metrics['R Squared']}")
    st.write(f"Mean Absolute Error (MAE): {metrics['MAE']}")
    st.write(f"Mean Squared Error (MSE): {metrics['MSE']}")
    st.write(f"Root Mean Squared Error (RMSE): {metrics['RMSE']}")
    st.write("---")
""" 
st.write("**ARIMA Model**")
st.write(f"Root Mean Squared Error (RMSE): {arima_rmse}")

# Visualize the ARIMA forecast
st.header("ARIMA Forecast")
fig, ax = plt.subplots()
ts_data.plot(ax=ax, label='Observed', legend=True)
arima_model_fit.fittedvalues.plot(ax=ax, style='--', color='red', label='Fitted')
arima_forecast.plot(ax=ax, style='--', color='green', label='Forecast')
plt.legend()
st.pyplot(fig)
"""
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

st.table(input_final)

if st.button("Predict"):
    predictions = {name: model.predict(input_final)[0] for name, model in models.items()}
    for name, prediction in predictions.items():
        st.write(f"**{name} Prediction**: {prediction:.2f} days")
# Predict and add predictions to the DataFrame

def predict_maintenance_duration(row):
    input_data = pd.DataFrame([[row[machine_id_column], row[department_column], row[issue_column]]], 
                              columns=[machine_id_column, department_column, issue_column])
    input_encoded = encoder.transform(input_data[[department_column, issue_column]])
    input_encoded_df = pd.DataFrame(input_encoded, columns=encoded_feature_names)
    input_final = pd.concat([input_data[[machine_id_column]], input_encoded_df], axis=1)
    prediction = model.predict(input_final)
    return prediction[0]

df['Predicted Maintenance Duration (days)'] = df.apply(predict_maintenance_duration, axis=1)

# Display the updated DataFrame with predictions
st.header("Maintenance Records with Predictions")
st.table(df.drop_duplicates())

# Filter by machine type
st.sidebar.header("Filter")
selected_machine_type = st.sidebar.selectbox("Select Machine Type", df[department_column].unique())
selected_Issue_type = st.sidebar.selectbox("Select Issue Type", df[issue_column].unique())
filtered_data = (df[(df[department_column] == selected_machine_type) & (df[issue_column]==selected_Issue_type)])
st.header(f"Records for Machine Type: {selected_machine_type} and Isuse Type: {selected_Issue_type}")
st.table(filtered_data)

