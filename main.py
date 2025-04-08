import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error

st.set_page_config(page_title="Predictive Maintenance", layout="wide")
st.title("📈 การพยากรณ์วันหมดอายุของน้ำยา / แบตเตอรี่")

# Load data and cache it
@st.cache_data
def load_data():
    data_path = 'สถิติ Pose Repairman.xlsx'
    sheet_name = 'ข้อมูลการใช้นำยา'
    df = pd.read_excel(data_path, sheet_name=sheet_name, engine='openpyxl')
    df.rename(columns={
        'แผนก': 'แผนก',
        'หมายเลขเครื่อง': 'หมายเลขเครื่อง',
        'ปัญหา': 'ปัญหา',
        'ระยะเวลาในใช้น้ำยา /แบต (วัน)': 'อัตราการใช้งาน (วัน)',
        'วันที่': 'วันที่เติมน้ำยา'
    }, inplace=True)
    df['วันที่เติมน้ำยา'] = pd.to_datetime(df['วันที่เติมน้ำยา'])
    return df

# Train multiple models and cache them
@st.cache_resource
def train_models(df):
    df = df.dropna(subset=['แผนก', 'หมายเลขเครื่อง', 'ปัญหา', 'อัตราการใช้งาน (วัน)'])
    encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
    X = encoder.fit_transform(df[['แผนก', 'หมายเลขเครื่อง', 'ปัญหา']])
    y = df['อัตราการใช้งาน (วัน)']

    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(random_state=42),
        "Random Forest": RandomForestRegressor(random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42)
    }

    for name in models:
        models[name].fit(X, y)

    return models, encoder

# Predict maintenance duration for new data
def predict_all_models(new_df, models, encoder):
    X_new = encoder.transform(new_df[['แผนก', 'หมายเลขเครื่อง', 'ปัญหา']])
    predictions = {}
    for name, model in models.items():
        predictions[name] = model.predict(X_new)
    return predictions

# Main logic
raw_df = load_data()
models, encoder = train_models(raw_df)

# User filters
with st.sidebar:
    st.header("🔍 ตัวกรองข้อมูล")
    selected_departments = st.multiselect("เลือกแผนก", raw_df['แผนก'].unique(), default=raw_df['แผนก'].unique())
    selected_machines = st.multiselect("เลือกหมายเลขเครื่อง", raw_df['หมายเลขเครื่อง'].unique(), default=raw_df['หมายเลขเครื่อง'].unique())

# Filtered data
filtered_df = raw_df[
    (raw_df['แผนก'].isin(selected_departments)) &
    (raw_df['หมายเลขเครื่อง'].isin(selected_machines))
]

# Prediction
if not filtered_df.empty:
    filtered_df = filtered_df.copy()
    all_predictions = predict_all_models(filtered_df, models, encoder)

    error_summary = []

    for name, preds in all_predictions.items():
        filtered_df[f'{name} (วัน)'] = np.round(preds).astype(int)
        filtered_df[f'{name} วันหมดอายุ'] = filtered_df['วันที่เติมน้ำยา'] + pd.to_timedelta(filtered_df[f'{name} (วัน)'], unit='D')

        # Calculate error metrics
        mae = mean_absolute_error(filtered_df['อัตราการใช้งาน (วัน)'], preds)
        rmse = np.sqrt(mean_squared_error(filtered_df['อัตราการใช้งาน (วัน)'], preds))
        error_summary.append({"Model": name, "MAE": mae, "RMSE": rmse})

    st.success(f"พบข้อมูล {len(filtered_df)} รายการ")
    st.dataframe(filtered_df[[
        'แผนก', 'หมายเลขเครื่อง', 'ปัญหา', 'วันที่เติมน้ำยา', 'อัตราการใช้งาน (วัน)'
    ] + [col for col in filtered_df.columns if any(m in col for m in models.keys())]]
    .sort_values(by='วันที่เติมน้ำยา'))

    st.subheader("📊 ค่าความคลาดเคลื่อนของแต่ละโมเดล")
    st.dataframe(pd.DataFrame(error_summary).sort_values(by="RMSE"))

    for name in all_predictions:
        st.subheader(f"📉 กราฟผลการพยากรณ์ ({name})")
        fig = px.timeline(
            filtered_df,
            x_start='วันที่เติมน้ำยา',
            x_end=f'{name} วันหมดอายุ',
            y='หมายเลขเครื่อง',
            color='แผนก',
            title=f'ผลการพยากรณ์ ({name})'
        )
        fig.update_layout(xaxis_title='วันที่', yaxis_title='หมายเลขเครื่อง', height=600)
        st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("ไม่พบข้อมูลที่ตรงกับเงื่อนไข")
