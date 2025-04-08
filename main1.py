import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import OneHotEncoder
import statsmodels.api as sm
import numpy as np

# Load the data
data_path = r'C:\Users\LOQ\OneDrive\Pose Health Care\MA (PPop)\สถิติ Pose Repairman.xlsx'  
sheet_name = 'ข้อมูลการใช้นำยา'
df = pd.read_excel(data_path, sheet_name=sheet_name)

# Preprocess the data (convert to numeric)
df['ระยะเวลาในใช้น้ำยา /แบต (วัน)'] = pd.to_numeric(df['ระยะเวลาในใช้น้ำยา /แบต (วัน)'], errors='coerce')

# Streamlit app
st.title("Machinery Maintenance Information and Prediction2")

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
encoded_features = encoder.fit_transform(df[[issue_column]])

# Create a DataFrame with the encoded features
encoded_feature_names = encoder.get_feature_names_out([issue_column])
encoded_df = pd.DataFrame(encoded_features, columns=encoded_feature_names, index=df.index)

# Store models for each department
models_sklearn = {}
models_statsmodels = {}

# Train separate models for each department
for department in df[department_column].unique():
    st.write(f"Training models for department: {department}")
    df_dept = df[df[department_column] == department]
    
    # Combine the encoded features with the numeric feature
    encoded_dept_df = encoded_df.loc[df_dept.index]
    X = pd.concat([df_dept[[machine_id_column]].reset_index(drop=True), encoded_dept_df.reset_index(drop=True)], axis=1)
    y = df_dept[maintenance_duration_column].reset_index(drop=True)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model using sklearn
    model_sklearn = LinearRegression()
    model_sklearn.fit(X_train, y_train)
    models_sklearn[department] = model_sklearn

    # Train the model using statsmodels
    X_train_sm = sm.add_constant(X_train)  # adding a constant
    ols_model = sm.OLS(y_train, X_train_sm).fit()
    models_statsmodels[department] = ols_model

# Display regression equations and coefficients
st.header("Regression Equations and Coefficients")

# Scikit-learn coefficients
st.write("**Scikit-learn Linear Regression Equations**")
for department, model in models_sklearn.items():
    sklearn_eq = f"Department {department}: y = {model.intercept_:.4f}"
    for coef, name in zip(model.coef_, X.columns):
        sklearn_eq += f" + ({coef:.4f} * {name})"
    st.write(sklearn_eq)

# Statsmodels coefficients
st.write("**Statsmodels OLS Regression Equations**")
for department, model in models_statsmodels.items():
    params = model.params
    statsmodels_eq = f"Department {department}: y = {params[0]:.4f}"
    for coef, name in zip(params[1:], X_train_sm.columns[1:]):
        statsmodels_eq += f" + ({coef:.4f} * {name})"
    st.write(statsmodels_eq)

# Predict for a new input
st.header("Predict Time Until Maintenance Issue")
selected_department = st.selectbox("Select Department for Prediction", df[department_column].unique())
machine_id = st.number_input("Enter Machine ID:", min_value=int(df[machine_id_column].min()), max_value=int(df[machine_id_column].max()))
selected_issue = st.selectbox("Select Issue", df[issue_column].unique())

# Prepare the input data for prediction
input_data = pd.DataFrame([[machine_id, selected_issue]], columns=[machine_id_column, issue_column])
input_encoded = encoder.transform(input_data[[issue_column]])
input_encoded_df = pd.DataFrame(input_encoded, columns=encoded_feature_names)
input_final = pd.concat([input_data[[machine_id_column]], input_encoded_df], axis=1)
input_final_sm = sm.add_constant(input_final)

if st.button("Predict"):
    # Prediction using sklearn model
    model_sklearn = models_sklearn[selected_department]
    prediction = model_sklearn.predict(input_final)
    st.write(f"Predicted Time Until Maintenance Issue (sklearn): {prediction[0]:.2f} days")

    # Prediction using statsmodels model
    model_statsmodels = models_statsmodels[selected_department]
    prediction_sm = model_statsmodels.predict(input_final_sm)
    st.write(f"Predicted Time Until Maintenance Issue (statsmodels): {prediction_sm[0]:.2f} days")

# Filter by machine type
st.sidebar.header("Filter by Machine Type")
selected_machine_type = st.sidebar.selectbox("Select Machine Type", df[department_column].unique())
filtered_data = df[df[department_column] == selected_machine_type]
st.header(f"Records for Machine Type: {selected_machine_type}")
st.table(filtered_data)
