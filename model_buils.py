import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import OneHotEncoder
import joblib

# ====== STEP 1: LOAD DATA ======
data_path = 'streamlit-start\สถิติ Pose Repairman.xlsx'
sheet_name = 'ข้อมูลการใช้นำยา'
df = pd.read_excel(data_path, sheet_name=sheet_name, engine='openpyxl')
df['ระยะเวลาในใช้น้ำยา /แบต (วัน)'] = pd.to_numeric(df['ระยะเวลาในใช้น้ำยา /แบต (วัน)'], errors='coerce')
df['วันที่'] = pd.to_datetime(df['วันที่'], errors='coerce').dt.date

# ====== STEP 2: CLEANING ======
df['ระยะเวลาในใช้น้ำยา /แบต (วัน)'] = pd.to_numeric(df['ระยะเวลาในใช้น้ำยา /แบต (วัน)'], errors='coerce')
df['วันที่'] = pd.to_datetime(df['วันที่'], errors='coerce').dt.date

# ====== STEP 3: FEATURES ======
department_column = 'แผนก'
machine_id_column = 'หมายเลขเครื่อง'
issue_column = 'ปัญหา'
target_column = 'ระยะเวลาในใช้น้ำยา /แบต (วัน)'

# One-hot encoding
encoder = OneHotEncoder(sparse_output=False)
encoded = encoder.fit_transform(df[[department_column, issue_column]])
encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out([department_column, issue_column]))

# รวม features
X = pd.concat([df[[machine_id_column]].reset_index(drop=True), encoded_df.reset_index(drop=True)], axis=1)
y = df[target_column].fillna(df[target_column].median())

# ====== STEP 4: SPLIT DATA ======
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ====== STEP 5: TRAIN AND EXPORT MODELS ======
models = {
    "LinearRegression": LinearRegression(),
    "DecisionTree": DecisionTreeRegressor(random_state=42),
    "RandomForest": RandomForestRegressor(random_state=42),
    "GradientBoosting": GradientBoostingRegressor(random_state=42)
}

for name, model in models.items():
    model.fit(X_train, y_train)
    joblib.dump(model, f"{name}.pkl")
    print(f"✅ {name}.pkl saved.")
