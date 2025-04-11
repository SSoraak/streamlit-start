import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import plotly.graph_objects as go
import joblib
# ---------------------- CONFIG ----------------------
st.set_page_config(
    page_title="Machinery Maintenance System",
    layout="wide",  # Full width
    initial_sidebar_state="collapsed"
)

# ---------------------- STYLE -----------------------
hide_streamlit_style = """
    <style>

        .block-container {
            padding-top: 4rem;
            padding-bottom: 4rem;
            padding-left: 4rem;
            padding-right: 4rem;
        }
        .css-18e3th9 {
            padding-left: 1rem !important;
            padding-right: 1rem !important;
        }
    </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)



# แคชข้อมูล
@st.cache_data
def load_data():
    data_path = r'สถิติ Pose Repairman.xlsx'
    sheet_name = 'ข้อมูลการใช้นำยา'
    df = pd.read_excel(data_path, sheet_name=sheet_name, engine='openpyxl')
    df['ระยะเวลาในใช้น้ำยา /แบต (วัน)'] = pd.to_numeric(df['ระยะเวลาในใช้น้ำยา /แบต (วัน)'], errors='coerce')
    df['วันที่'] = pd.to_datetime(df['วันที่'], errors='coerce').dt.date
    return df

@st.cache_resource
def train_models(X_train, y_train):
    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(random_state=42),
        "Random Forest": RandomForestRegressor(random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42)
    }
    for name, model in models.items():
        model.fit(X_train, y_train)
    return models

@st.cache_resource
def load_trained_models():
    models = {
        "Linear Regression": joblib.load("LinearRegression.pkl"),
        "Decision Tree": joblib.load("DecisionTree.pkl"),
        "Random Forest": joblib.load("RandomForest.pkl"),
        "Gradient Boosting": joblib.load("GradientBoosting.pkl")
    }
    return models

@st.cache_resource
def get_encoder(df, department_column, issue_column):
    encoder = OneHotEncoder(sparse_output=False)
    encoder.fit(df[[department_column, issue_column]])
    return encoder

#@st.cache_data
def predict_all_data(_df, _encoder, _model):
    input_encoded = _encoder.transform(_df[[department_column, issue_column]])
    input_encoded_df = pd.DataFrame(input_encoded, columns=encoded_feature_names)
    input_final = pd.concat([_df[[machine_id_column]], input_encoded_df], axis=1)
    predictions = _model.predict(input_final)
    return np.round(predictions).astype(int)


# ---------------------
# Navigation
# ---------------------
pages = {
    "Pose maintenance": [
        st.Page("pose_maintenance/dashboard_bi.py", title="Dashboard", icon="📊"),
        st.Page("pose_maintenance/prediction.py", title="Prediction", icon="🔮")
    ],
    "page": [
        st.Page("pages/1.py", title="1", icon=":material/favorite:"),
        st.Page("pages/2.py", title="2"),
        st.Page("pages/3.py", title="3"),
        st.Page("pages/4.py", title="4"),
        st.Page("pages/5.py", title="5"),  
        st.Page("pages/6.py", title="6"),
        st.Page("pages/7.py", title="7"),
        st.Page("pages/8.py", title="8"),
        st.Page("pages/9.py", title="9"),
        
    ]
}

pg = st.navigation(pages, position="hidden")
pg.run()

#st.sidebar.markdown("<h3 style='text-align: left;'>Dashboard</h3>", unsafe_allow_html=True)
#st.sidebar.markdown("<h3 style='text-align: left;'>Predict</h3>", unsafe_allow_html=True)
page = st.sidebar.selectbox("Select Page", ["📊 Dashboard", "🔧 Predict Maintenance"])


st.markdown("<h1 style='text-align: left;'>Machinery Maintenance </h1>", unsafe_allow_html=True)



if page == "📊 Dashboard":
    st.header("📊 Dashboard Overview")
    dashboard_url = "https://your-dashboard-url.com"  # <== ใส่ URL dashboard จริงตรงนี้
    st.markdown(f"""
        <iframe title="ข้อมูลเครื่องจักร MA (ธีม)" width="100%" height="750" src="https://app.powerbi.com/reportEmbed?reportId=892b0a03-c03d-41d9-a3fd-79ac05cbca3d&autoAuth=true&ctid=1f23f2e8-f8b5-4438-a7d7-1d2d87dffcf3" frameborder="0" allowFullScreen="true"></iframe>
    """, unsafe_allow_html=True)

elif page == "🔧 Predict Maintenance":
    st.header("🔧 Predict Maintenance Information")

    df = load_data()
    department_column = 'แผนก'
    machine_id_column = 'หมายเลขเครื่อง'
    issue_column = 'ปัญหา'
    maintenance_duration_column = 'ระยะเวลาในใช้น้ำยา /แบต (วัน)'

    encoder = get_encoder(df, department_column, issue_column)
    encoded_features = encoder.transform(df[[department_column, issue_column]])
    encoded_feature_names = encoder.get_feature_names_out([department_column, issue_column])
    encoded_df = pd.DataFrame(encoded_features, columns=encoded_feature_names)

    X = pd.concat([df[[machine_id_column]], encoded_df], axis=1)
    y = df[maintenance_duration_column].fillna(df[maintenance_duration_column].median())

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    models = load_trained_models()

    # คำนวณประสิทธิภาพของโมเดลที่โหลดมา
    model_results = {}
    predictions = {}
    for name, model in models.items():
        y_pred = model.predict(X_test)
        model_results[name] = {
            "R Squared": r2_score(y_test, y_pred),
            "MAE": mean_absolute_error(y_test, y_pred),
            "MSE": mean_squared_error(y_test, y_pred),
            "RMSE": np.sqrt(mean_squared_error(y_test, y_pred))
        }
        predictions[name] = y_pred

    st.subheader("Model Performance")
    cols = st.columns(4)
    for i, (name, metrics) in enumerate(model_results.items()):
        with cols[i % 4]:
            st.info(f"**{name}**")
            st.write(f"R Squared: {metrics['R Squared']:.4f}")
            st.write(f"MAE: {metrics['MAE']:.2f}")
            st.write(f"MSE: {metrics['MSE']:.2f}")
            st.write(f"RMSE: {metrics['RMSE']:.2f}")

    # กราฟเปรียบเทียบ
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(range(len(y_test))), y=y_test, mode='lines', name='Actual Values'))
    colors = ['red', 'green', 'blue', 'orange']
    for idx, (name, y_pred) in enumerate(predictions.items()):
        fig.add_trace(go.Scatter(x=list(range(len(y_pred))), y=y_pred, mode='lines', 
                                name=f'{name} Predictions', line=dict(color=colors[idx])))
    fig.update_layout(title="Model Predictions vs Actual Values", 
                     xaxis_title="Sample Index", 
                     yaxis_title="Maintenance Duration (days)")
    st.plotly_chart(fig)

    # การทำนายสำหรับข้อมูลใหม่
    st.subheader("Predict for New Input")
    col1, col2, col3 = st.columns(3)

    with col1:
        machine_id = st.number_input(
            "Enter Machine ID:",
            min_value=int(df[machine_id_column].min()),
            max_value=int(df[machine_id_column].max())
        )

    with col2:
        selected_department = st.selectbox(
            "Select Department",
            df[department_column].unique()
        )

    with col3:
        selected_issue = st.selectbox(
            "Select Issue",
            df[issue_column].unique()
        )

    if st.button("Predict"):
        input_data = pd.DataFrame([[machine_id, selected_department, selected_issue]], 
                                columns=[machine_id_column, department_column, issue_column])
        input_encoded = encoder.transform(input_data[[department_column, issue_column]])
        input_encoded_df = pd.DataFrame(input_encoded, columns=encoded_feature_names)
        input_final = pd.concat([input_data[[machine_id_column]], input_encoded_df], axis=1)
        #st.write("Input for prediction:", input_final) # Show the input data
        new_predictions = {name: np.round(model.predict(input_final)[0]).astype(int) for name, model in models.items()}
        for name, prediction in new_predictions.items():
            st.success(f"**{name} Prediction**: {prediction:.0f} days")

    # Filter Sidebar
    st.sidebar.header("Select Model")
    selected_model = st.sidebar.selectbox("Select Model", list(models.keys()), index=list(models.keys()).index("Gradient Boosting")) # ตั้งค่าเริ่มต้น)
    st.sidebar.header("Filter Records")
    issue_types = ['All'] + list(df[issue_column].unique())
    selected_issue_type = st.sidebar.selectbox("Select Issue Type", issue_types)
    department_types = ['All'] + list(df[department_column].unique())
    selected_department_type = st.sidebar.selectbox("Select Department Type", department_types)
    machine_options = ['All'] + list(df[machine_id_column].unique()) if selected_department_type == 'All' else ['All'] + list(df[df[department_column] == selected_department_type][machine_id_column].unique())
    selected_machine_type = st.sidebar.selectbox("Select Machine Type", machine_options)

    # คำนวณการทำนายและวันที่ถัดไป
    for model_name, model in models.items():
        df[f'Predicted ({model_name})'] = predict_all_data(df, encoder, model)
        df[f'Next Date ({model_name})'] = pd.to_datetime(df['วันที่']) + pd.to_timedelta(df[f'Predicted ({model_name})'], unit='d')
        df[f'Next Date ({model_name})'] = df[f'Next Date ({model_name})'].dt.date

    df = df.sort_values(by='วันที่', ascending=False)
    
    # กรองข้อมูล
    filtered_data = df.copy()
    if selected_issue_type != 'All':
        filtered_data = filtered_data[filtered_data[issue_column] == selected_issue_type]
    if selected_department_type != 'All':
        filtered_data = filtered_data[filtered_data[department_column] == selected_department_type]
    if selected_machine_type != 'All':
        filtered_data = filtered_data[filtered_data[machine_id_column] == selected_machine_type]

    # เตรียมข้อมูลสำหรับแสดง
    display_data = filtered_data.rename(columns={
        'แผนก': 'Department',
        'หมายเลขเครื่อง': 'Machine ID',
        'ปัญหา': 'Issue',
        'วันที่': 'Date',
        'ระยะเวลาในใช้น้ำยา /แบต (วัน)': 'Duration (days)'
    })
    
    display_columns = ['Department', 'Machine ID', 'Issue', 'Date', 'Duration (days)', 
                      f'Predicted ({selected_model})', f'Next Date ({selected_model})']
    
    display_data = display_data[display_columns].reset_index(drop=True)
    display_data.index += 1

    display_data = display_data.rename(columns={
        f'Predicted ({selected_model})': 'Predicted Duration (days)',
        f'Next Date ({selected_model})': 'Next Predicted Date'
    })

    # [ส่วนสไตล์ตารางคงเดิม]

    st.header(f"Records for Machine: {selected_machine_type}  Issue: {selected_issue_type} Department: {selected_department_type}")
    st.info(f"Prediction By: {selected_model}")
    st.dataframe(display_data, use_container_width=True)