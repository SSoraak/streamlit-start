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
import plotly.graph_objects as go
import plotly.express as px

# Load the data
data_path = r'สถิติ Pose Repairman.xlsx'
sheet_name = 'ข้อมูลการใช้นำยา'
df = pd.read_excel(data_path, sheet_name=sheet_name, engine='openpyxl')

# Preprocess the data (convert to numeric)
df['ระยะเวลาในใช้น้ำยา /แบต (วัน)'] = pd.to_numeric(df['ระยะเวลาในใช้น้ำยา /แบต (วัน)'], errors='coerce')

# Streamlit app
st.title("Machinery Maintenance Information and Prediction")

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

# Binarize the target variable for classification (using a threshold)
threshold = y.median()
y_binary = (y >= threshold).astype(int)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train_bin, X_test_bin, y_train_bin, y_test_bin = train_test_split(X, y_binary, test_size=0.2, random_state=42)

# Train the models
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42)
}

model_results = {}
predictions = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r_squared = r2_score(y_test, y_pred)

    model_results[name] = {"R Squared": r_squared, "MAE": mae, "MSE": mse, "RMSE": rmse}
    predictions[name] = y_pred

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
    st.info(f"**{name}**")
    st.write(f"R Squared: {metrics['R Squared']}")
    st.write(f"Mean Absolute Error (MAE): {metrics['MAE']}")
    st.write(f"Mean Squared Error (MSE): {metrics['MSE']}")
    st.write(f"Root Mean Squared Error (RMSE): {metrics['RMSE']}")
    st.write("---")

# Plot predictions from all four models vs actual values
fig = go.Figure()

# Add actual values
fig.add_trace(go.Scatter(x=list(range(len(y_test))), y=y_test, mode='lines', name='Actual Values'))

# Add predictions for each model
colors = ['red', 'green', 'blue', 'orange']
for idx, (name, y_pred) in enumerate(predictions.items()):
    fig.add_trace(go.Scatter(x=list(range(len(y_pred))), y=y_pred, mode='lines', name=f'{name} Predictions', line=dict(color=colors[idx])))

fig.update_layout(
    title="Model Predictions vs Actual Values",
    xaxis_title="Sample Index",
    yaxis_title="Maintenance Duration (days)",
    legend_title="Legend",
    template="plotly_white"
)

st.plotly_chart(fig)

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
    predictions = {name: model.predict(input_final)[0] for name, model in models.items()}
    for name, prediction in predictions.items():
        st.info(f"**{name} Prediction**: {prediction:.0f} days")

# Predict and add predictions to the DataFrame
def predict_maintenance_duration(row):
    input_data = pd.DataFrame([[row[machine_id_column], row[department_column], row[issue_column]]], 
                              columns=[machine_id_column, department_column, issue_column])
    input_encoded = encoder.transform(input_data[[department_column, issue_column]])
    input_encoded_df = pd.DataFrame(input_encoded, columns=encoded_feature_names)
    input_final = pd.concat([input_data[[machine_id_column]], input_encoded_df], axis=1)
    prediction = model.predict(input_final)
    return prediction[0]

df['Predicted Maintenance Duration (days)'] = df.apply(predict_maintenance_duration, axis=1).astype(int)


# Convert date column to datetime (assuming date column is 'วันที่')
df['วันที่'] = pd.to_datetime(df['วันที่'], errors='coerce')
df['วันที่'] = df['วันที่'].dt.date
# Sort by date and remove duplicates, keeping the most recent
df = df.sort_values(by='วันที่').drop_duplicates(subset=['แผนก', 'หมายเลขเครื่อง', 'ปัญหา'], keep='last')

# Calculate the dates for the next day
df['Predicted next date'] = df['วันที่'] + pd.to_timedelta(df['Predicted Maintenance Duration (days)'], unit='d')

# Sidebar for filters
st.sidebar.header("Filter Records for Machine")

# Select Issue Type with 'All' option
issue_types = ['All'] + list(df[issue_column].unique())
selected_Issue_type = st.sidebar.selectbox("Select Issue Type", issue_types)

# Select Department Type with 'All' option
department_types = ['All'] + list(df[department_column].unique())
selected_department_type = st.sidebar.selectbox("Select Department Type", department_types)

# Dynamic Machine Type based on selected Department
if selected_department_type == 'All':
    machine_options = ['All'] + list(df[machine_id_column].unique())
else:
    machine_options = ['All'] + list(df[df[department_column] == selected_department_type][machine_id_column].unique())
selected_machine_type = st.sidebar.selectbox("Select Machine Type", machine_options)

# Apply filters to the DataFrame
filtered_data = df.copy()
if selected_Issue_type != 'All':
    filtered_data = filtered_data[filtered_data[issue_column] == selected_Issue_type]
if selected_department_type != 'All':
    filtered_data = filtered_data[filtered_data[department_column] == selected_department_type]
if selected_machine_type != 'All':
    filtered_data = filtered_data[filtered_data[machine_id_column] == selected_machine_type]


# Display header and filtered data
st.header(f"Records for Machine: {selected_machine_type}  Issue: {selected_Issue_type} Department: {selected_department_type}")
st.info("Gradient Boosting Model")
st.table(filtered_data.reset_index(drop=True))

# Interactive graphs
#st.header("Interactive Graphs")

# Number of issues per month
#df['เดือน'] = df['วันที่'].dt.to_period('M').astype(str)
#issue_count_per_month = df.groupby(['เดือน', issue_column]).size().reset_index(name='count')
#fig1 = px.bar(issue_count_per_month, x='เดือน', y='count', color=issue_column, title='Number of Issues per Month')
#st.plotly_chart(fig1)

# Average maintenance duration per department
#avg_duration_per_department = df.groupby(department_column)[maintenance_duration_column].mean().reset_index()
#fig2 = px.bar(avg_duration_per_department, x=department_column, y=maintenance_duration_column, title='Average Maintenance Duration per Department')
#st.plotly_chart(fig2) 

