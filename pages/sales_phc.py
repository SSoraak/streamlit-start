import streamlit as st
import mysql.connector
import pandas as pd
from mysql.connector import Error
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import os

# ฟังก์ชันแคชการเชื่อมต่อฐานข้อมูล
@st.cache_resource
def get_db_connection():
    try:
        db_config = st.secrets["db_config"]
        required_keys = ["host", "database", "user", "password"]
        if not all(key in db_config for key in required_keys):
            raise KeyError("Missing required keys in db_config")
        conn = mysql.connector.connect(**db_config)
        return conn
    except (Error, KeyError) as e:
        st.error(f"ไม่สามารถเชื่อมต่อฐานข้อมูล: {e}")
        return None

# ฟังก์ชันดึงข้อมูลจากฐานข้อมูล
def fetch_data(_conn, query, params=None):
    try:
        cursor = _conn.cursor(dictionary=True)
        if params:
            cursor.execute(query, params)
        else:
            cursor.execute(query)
        data = cursor.fetchall()
        cursor.close()
        return pd.DataFrame(data)
    except Error as e:
        st.error(f"เกิดข้อผิดพลาดในการดึงข้อมูล: {e}")
        return None

# ฟังก์ชันแคชข้อมูลล่าสุด (2 เดือน)
@st.cache_data(ttl=3600)  # แคช 1 ชั่วโมง
def fetch_recent_data(_conn, query, start_date, end_date):
    return fetch_data(_conn, query, (start_date, end_date))

# ฟังก์ชันแก้ไขปีพ.ศ. เป็น ค.ศ.
def convert_to_ad(date):
    current_year = datetime.now().year
    if pd.isna(date):
        return date
    year = date.year
    if year > (current_year + 2):  # ถ้าปีเกินปีปัจจุบัน +2 ปี ถือว่าเป็นพ.ศ.
        return date.replace(year=year - 543)
    if year < 2010: #ลบแถวนั้นออก
        return date.replace(year=year + 543) 
    return date

# ฟังก์ชันทำความสะอาดข้อมูล
def phc_clean_data(df):
    df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y', errors='coerce')
    numeric_cols = ["Qty", "Price", "Total", "WelfareB"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(subset=["Date"], inplace=True)
    df["Date"] = df["Date"].apply(convert_to_ad)
    return df

# ฟังก์ชันบันทึกข้อมูลลงไฟล์ Parquet
def save_to_parquet(df, filename="sales_data.parquet"):
    try:
        df.to_parquet(filename, index=False)
        st.success(f"บันทึกข้อมูลลงไฟล์ '{filename}' สำเร็จ")
    except Exception as e:
        st.error(f"เกิดข้อผิดพลาดในการบันทึกไฟล์: {e}")

# ฟังก์ชันโหลดข้อมูลจากไฟล์ Parquet
def load_from_parquet(filename="sales_data.parquet"):
    try:
        if os.path.exists(filename):
            df = pd.read_parquet(filename)
            return df
        return None
    except Exception as e:
        st.error(f"เกิดข้อผิดพลาดในการโหลดไฟล์: {e}")
        return None

# คำสั่ง SQL พื้นฐาน
base_query = """
SELECT
  dallycall.xDate AS Date,
  item.NameTH,
  dallycall.ItemCode,
  dallycall.Qty,
  dallycall.UnitCode,
  item_unit.Unit_Name,
  dallycall.Price,
  dallycall.Total,
  dallycall.WelfareB,
  dallycall.AreaCode,
  dallycall.Cus_Code,
  customer.FName AS Name_Cus,
  customer.Prefix_Code,
  customer.Cus_Type,
  customer.Cus_Type_Sub,
  customer.IsLetter,
  customer.BedCapacity,
  customer.IsBedConfirm,
  customer.IsActive,
  customer_type.CT_Name,
  th_province.Name_Th AS province 
FROM
  dallycall
  INNER JOIN customer ON dallycall.Cus_Code = customer.Cus_Code
  INNER JOIN customer_type ON customer.Cus_Type = customer_type.CT_Code
  INNER JOIN customer_type_sub ON customer_type.CT_Code = customer_type_sub.Customer_Type_Code
  INNER JOIN item ON dallycall.ItemCode = item.Item_Code
  INNER JOIN item_unit ON dallycall.UnitCode = item_unit.Unit_Code
  INNER JOIN th_province
  INNER JOIN area ON th_province.Pv_Code = area.Pv_Code 
  AND dallycall.AreaCode = area.`Code` 
WHERE
  dallycall.IsCancel = 0 
  AND dallycall.AreaCode2 = "-"
"""

# Query สำหรับข้อมูลเก่า (จนถึงก่อน 2 เดือน)
historical_query = base_query + " AND dallycall.xDate <= %s"

# Query สำหรับข้อมูลล่าสุด (2 เดือนล่าสุด)
recent_query = base_query + " AND dallycall.xDate BETWEEN %s AND %s"

def main():
    # คำนวณวันที่สำหรับแบ่งข้อมูล
    today = datetime.now().date()
    two_months_ago = (today - relativedelta(months=2)).replace(day=1) - timedelta(days=1)
    recent_start = two_months_ago + timedelta(days=1)

    # ดึงการเชื่อมต่อ
    conn = get_db_connection()
    if conn is None:
        st.stop()

    # ไฟล์สำหรับเก็บข้อมูล
    data_file = "sales_data.parquet"

    # โหลดข้อมูลก้อนแรก (ถ้ายังไม่มีไฟล์)
    if not os.path.exists(data_file):
        st.write("กำลังโหลดข้อมูลเก่า (ครั้งแรก)...")
        historical_df = fetch_data(conn, historical_query, (two_months_ago,))
        if historical_df is None or historical_df.empty:
            st.warning("ไม่พบข้อมูลเก่าหรือเกิดข้อผิดพลาด")
            historical_df = pd.DataFrame()
        else:
            historical_df = phc_clean_data(historical_df)
            save_to_parquet(historical_df, data_file)
    else:
        st.info("ใช้ข้อมูลเก่าจากไฟล์ Parquet")

    # โหลดข้อมูลจากไฟล์
    combined_df = load_from_parquet(data_file)
    if combined_df is None or combined_df.empty:
        combined_df = pd.DataFrame()

    # ดึงข้อมูลล่าสุด (แคช 1 ชั่วโมง)
    recent_df = fetch_recent_data(conn, recent_query, recent_start, today)
    if recent_df is None or recent_df.empty:
        st.warning("ไม่พบข้อมูลล่าสุดหรือเกิดข้อผิดพลาด")
        recent_df = pd.DataFrame()
    else:
        recent_df = phc_clean_data(recent_df)

    # รวมข้อมูลและอัปเดตไฟล์
    if not recent_df.empty:
        combined_df = pd.concat([combined_df, recent_df], ignore_index=True)
        # ลบแถวซ้ำ (ถ้ามี) โดยใช้คอลัมน์ที่เป็น unique identifier
        combined_df = combined_df.drop_duplicates(subset=["Date", "Cus_Code", "ItemCode"], keep="last")
        # บันทึกข้อมูลรวมลงไฟล์
        save_to_parquet(combined_df, data_file)

    # ทำความสะอาดข้อมูลรวม (ถ้ายังไม่ได้ทำ)
    if not combined_df.empty:
        combined_df = phc_clean_data(combined_df)

    # แสดงผลลัพธ์
    st.write("ข้อมูลทั้งหมดจากไฟล์:")
    st.write(f"จำนวนแถวทั้งหมด: {combined_df.shape[0].__format__(',.0f')}")
    st.dataframe(combined_df.head(10))

    # เพิ่มปุ่มรีเซ็ตข้อมูล
    if st.button("รีเซ็ตข้อมูลเก่า"):
        if os.path.exists(data_file):
            os.remove(data_file)
        st.experimental_rerun()

    # ไม่ปิด connection เพราะ st.cache_resource จัดการให้
    # Sidebar สำหรับตัวกรอง
    st.sidebar.header("🔍 Filter")
    # ตัวกรองช่วงวันที่
    min_date = combined_df["Date"].min().date() if not combined_df.empty else today - timedelta(days=365)
    max_date = combined_df["Date"].max().date() if not combined_df.empty else today
    date_range = st.sidebar.date_input(
        "เลือกช่วงวันที่",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
        format="DD-MM-YYYY",
        key="date_range_filter"
    )
    # ตัวกรองสินค้าและลูกค้า
    selected_item = st.sidebar.selectbox(
        "เลือกสินค้า",
        options=["ทั้งหมด"] + list(combined_df["NameTH"].unique()),
        key="item_filter"
    )
    selected_customer = st.sidebar.selectbox(
        "เลือกลูกค้า",
        options=["ทั้งหมด"] + list(combined_df["Name_Cus"].unique()),
        key="customer_filter"
    )

    # ใช้ตัวแปร filtered_df เพื่อเก็บข้อมูลที่กรองแล้ว
    filtered_df = combined_df.copy()

    # ใช้ตัวกรอง
    if len(date_range) == 2:  # ตรวจสอบว่าเลือกช่วงวันที่ครบ
        start_date, end_date = date_range
        filtered_df = filtered_df[(filtered_df["Date"].dt.date >= start_date) & 
                                (filtered_df["Date"].dt.date <= end_date)]
    if selected_item != "ทั้งหมด":
        filtered_df = filtered_df[filtered_df["NameTH"] == selected_item]
    if selected_customer != "ทั้งหมด":
        filtered_df = filtered_df[filtered_df["Name_Cus"] == selected_customer]
    
    # แสดงผลลัพธ์
    st.write("ข้อมูลทั้งหมดจากไฟล์:")
    st.write(f"จำนวนแถวทั้งหมด: {combined_df.shape[0].__format__(',.0f')}")
    st.dataframe( filtered_df)

    
if __name__ == "__main__":
    main()