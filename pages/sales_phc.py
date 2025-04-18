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
        if "is_initial_load" in st.session_state and st.session_state.is_initial_load:
            st.toast(f"ไม่สามารถเชื่อมต่อฐานข้อมูล: {e}", icon="❌")
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
        if "is_initial_load" in st.session_state and st.session_state.is_initial_load:
            st.toast(f"เกิดข้อผิดพลาดในการดึงข้อมูล: {e}", icon="❌")
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
    return date

# ฟังก์ชันทำความสะอาดข้อมูล
def phc_clean_data(df):
    df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y', errors='coerce')
    # ลบแถวที่ Date มีปีน้อนกว่า 2010
    df = df[df['Date'].dt.year >= 2010]
    # แปลงคอลัมน์ที่เป็นตัวเลขให้เป็นชนิดข้อมูลตัวเลข
    numeric_cols = ["Qty", "Price", "Total", "WelfareB"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(subset=["Date"], inplace=True)
    df["Date"] = df["Date"].apply(convert_to_ad)
    return df

# ฟังก์ชันบันทึกข้อมูลลงไฟล์ Parquet และ CSV
def save_to_files(df, parquet_filename="sales_data.parquet", csv_filename="sales_data.csv", show_notification=False):
    try:
        # บันทึกเป็น Parquet
        df.to_parquet(parquet_filename, index=False)
        # บันทึกเป็น CSV
        df.to_csv(csv_filename, index=False, encoding='utf-8-sig')
        if show_notification:
            st.toast(f"บันทึกข้อมูลลงไฟล์ '{parquet_filename}' และ '{csv_filename}' สำเร็จ", icon="✅")
    except Exception as e:
        if show_notification:
            st.toast(f"เกิดข้อผิดพลาดในการบันทึกไฟล์: {e}", icon="❌")

# ฟังก์ชันโหลดข้อมูลจากไฟล์ Parquet
def load_from_parquet(filename="sales_data.parquet"):
    try:
        if os.path.exists(filename):
            df = pd.read_parquet(filename)
            return df
        return None
    except Exception as e:
        if "is_initial_load" in st.session_state and st.session_state.is_initial_load:
            st.toast(f"เกิดข้อผิดพลาดในการโหลดไฟล์: {e}", icon="❌")
        return None

# คำสั่ง SQL พื้นฐาน
base_query = """
SELECT
  dallycall.DocNo,
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
    # ตรวจสอบว่าเป็นการโหลดครั้งแรกหรือไม่
    if "is_initial_load" not in st.session_state:
        st.session_state.is_initial_load = True
    else:
        st.session_state.is_initial_load = False

    # คำนวณวันที่สำหรับแบ่งข้อมูล
    today = datetime.now().date()
    two_months_ago = (today - relativedelta(months=2)).replace(day=1) - timedelta(days=1)
    recent_start = two_months_ago + timedelta(days=1)

    # ดึงการเชื่อมต่อ
    conn = get_db_connection()
    if conn is None:
        st.stop()

    # ไฟล์สำหรับเก็บข้อมูล
    parquet_file = "sales_data.parquet"
    csv_file = "sales_data.csv"

    # โหลดข้อมูลเก่าจาก Parquet (ถ้ามี)
    combined_df = load_from_parquet(parquet_file)
    if combined_df is None or combined_df.empty:
        combined_df = pd.DataFrame()

    # ตรวจสอบว่าต้องโหลดข้อมูลเก่าหรือไม่
    if combined_df.empty and st.session_state.is_initial_load:
        with st.status("กำลังโหลดข้อมูลเก่า (ครั้งแรก)...", expanded=False):
            historical_df = fetch_data(conn, historical_query, (two_months_ago,))
            if historical_df is None or historical_df.empty:
                st.toast("ไม่พบข้อมูลเก่าหรือเกิดข้อผิดพลาด", icon="⚠️")
                historical_df = pd.DataFrame()
            else:
                historical_df = phc_clean_data(historical_df)
                combined_df = historical_df
                save_to_files(combined_df, parquet_file, csv_file, show_notification=True)

    # ปุ่ม  สำหรับดึงข้อมูลใหม่
    if st.button("รีเฟรชข้อมูลล่าสุด", key="refresh_button"):
        with st.status("กำลังดึงข้อมูลล่าสุด...", expanded=False):
            recent_df = fetch_data(conn, recent_query, (recent_start, today))  # ดึงข้อมูลใหม่โดยไม่ใช้แคช
            if recent_df is None or recent_df.empty:
                st.toast("ไม่พบข้อมูลล่าสุดหรือเกิดข้อผิดพลาด", icon="⚠️")
                recent_df = pd.DataFrame()
            else:
                recent_df = phc_clean_data(recent_df)
                # ลบข้อมูลล่าสุด (2 เดือน) ออกจาก combined_df เพื่ออัปเดตด้วยข้อมูลใหม่
                if not combined_df.empty:
                    combined_df = combined_df[combined_df["Date"].dt.date <= two_months_ago]
                # รวมข้อมูลเก่าและข้อมูลล่าสุด
                combined_df = pd.concat([combined_df, recent_df], ignore_index=True)
                # ลบแถวซ้ำ (ถ้ามี) โดยใช้คอลัมน์ที่เป็น unique identifier
                combined_df = combined_df.drop_duplicates(subset=["DocNo", "Date", "Cus_Code", "ItemCode"], keep="last")
                # บันทึกข้อมูลรวมลงไฟล์ทั้ง Parquet และ CSV
                save_to_files(combined_df, parquet_file, csv_file, show_notification=True)
            st.rerun()

    # ดึงข้อมูลล่าสุด (แคช 1 ชั่วโมง)
    recent_df = fetch_recent_data(conn, recent_query, recent_start, today)
    if recent_df is None or recent_df.empty:
        if st.session_state.is_initial_load:
            st.toast("ไม่พบข้อมูลล่าสุดหรือเกิดข้อผิดพลาด", icon="⚠️")
        recent_df = pd.DataFrame()
    else:
        recent_df = phc_clean_data(recent_df)

    # รวมข้อมูลและอัปเดตไฟล์ (ถ้ามีข้อมูลล่าสุดใหม่)
    if not recent_df.empty:
        # ลบข้อมูลล่าสุด (2 เดือน) ออกจาก combined_df เพื่ออัปเดตด้วยข้อมูลใหม่
        if not combined_df.empty:
            combined_df = combined_df[combined_df["Date"].dt.date <= two_months_ago]
        # รวมข้อมูลเก่าและข้อมูลล่าสุด
        combined_df = pd.concat([combined_df, recent_df], ignore_index=True)
        # ลบแถวซ้ำ (ถ้ามี) โดยใช้คอลัมน์ที่เป็น unique identifier
        combined_df = combined_df.drop_duplicates(subset=["DocNo", "Date", "Cus_Code", "ItemCode"], keep="last")
        # บันทึกข้อมูลรวมลงไฟล์ทั้ง Parquet และ CSV
        save_to_files(combined_df, parquet_file, csv_file, show_notification=st.session_state.is_initial_load)

    # ทำความสะอาดข้อมูลรวม (ถ้ายังไม่ได้ทำ)
    if not combined_df.empty:
        combined_df = phc_clean_data(combined_df)

    # ให้ index เริ่มจาก 1 
    combined_df.index += 1
   
    # แสดงผลลัพธ์
    with st.expander("📊 ข้อมูลการขาย", expanded=True):
        st.write("ข้อมูลทั้งหมดจากไฟล์:")
        st.write(f"จำนวนแถวทั้งหมด: {combined_df.shape[0].__format__(',.0f')}")
        st.dataframe(combined_df.head(100))

    
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
    # ตัวกรองสินค้า
    selected_item = st.sidebar.selectbox(
        "เลือกสินค้า",
        options=["ทั้งหมด"] + list(combined_df["NameTH"].unique()),
        key="item_filter"
    )
    # ตัวกรองลูกค้า
    selected_customer = st.sidebar.selectbox(
        "เลือกลูกค้า",
        options=["ทั้งหมด"] + list(combined_df["Name_Cus"].unique()),
        key="customer_filter"
    )
    # ตัวกรองเขต (AreaCode)
    selected_area = st.sidebar.selectbox(
        "เลือกเขต",
        options=["ทั้งหมด"] + sorted(list(combined_df["AreaCode"].unique())),
        key="area_filter"
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
    if selected_area != "ทั้งหมด":
        filtered_df = filtered_df[filtered_df["AreaCode"] == selected_area]
    # ให้ index เริ่มจาก 1 นับใหม่
    filtered_df = filtered_df.reset_index(drop=True)
    filtered_df.index += 1
 
    # แสดงผลลัพธ์
    st.write("ข้อมูลทั้งหมดจากไฟล์:")
    st.write(f"จำนวนแถวทั้งหมด: {combined_df.shape[0].__format__(',.0f')} จำนวนแถวที่กรองแล้ว: {filtered_df.shape[0].__format__(',.0f')}")
    st.dataframe(filtered_df)

    # แสดงสรุปยอดขาย
    st.subheader("สรุปยอดขาย")
    a, b = st.columns(2)
    c, d = st.columns(2)
    e, f = st.columns(2)
    g, h = st.columns(2)

    # คำนวณเมตริกจาก filtered_df
    total_sales = filtered_df['Total'].sum() if not filtered_df.empty else 0
    total_orders = filtered_df['DocNo'].nunique() if not filtered_df.empty else 0
    unique_customers = filtered_df['Cus_Code'].nunique() if not filtered_df.empty else 0

    # คำนวณจำนวนลูกค้าใหม่
    if not filtered_df.empty and len(date_range) == 2:
        start_date, end_date = date_range
        # ลูกค้าที่มีใน filtered_df
        current_customers = set(filtered_df['Cus_Code'].unique())
        # ลูกค้าที่มีใน combined_df ก่อน start_date
        previous_customers = set(combined_df[combined_df["Date"].dt.date < start_date]['Cus_Code'].unique())
        # ลูกค้าใหม่ = ลูกค้าปัจจุบันที่ไม่มีในข้อมูลก่อนหน้า
        new_customers = len(current_customers - previous_customers)
    else:
        new_customers = 0

    # คำนวณ MTD, YTD, YoY, MoM จาก combined_df
    current_month_start = today.replace(day=1)
    previous_month_end = current_month_start - timedelta(days=1)
    previous_month_start = previous_month_end.replace(day=1)
    current_year = today.year
    previous_year = current_year - 1
    current_year_start = today.replace(year=current_year, month=1, day=1)
    previous_year_start = today.replace(year=previous_year, month=1, day=1)
    previous_year_end = today.replace(year=previous_year)

    # MTD
    mtd_sales = combined_df[
        (combined_df["Date"].dt.date >= current_month_start) &
        (combined_df["Date"].dt.date <= today)
    ]['Total'].sum() if not combined_df.empty else 0

    # YTD
    ytd_sales = combined_df[
        (combined_df["Date"].dt.date >= current_year_start) &
        (combined_df["Date"].dt.date <= today)
    ]['Total'].sum() if not combined_df.empty else 0

    # YoY
    previous_year_sales = combined_df[
        (combined_df["Date"].dt.date >= previous_year_start) &
        (combined_df["Date"].dt.date <= previous_year_end)
    ]['Total'].sum() if not combined_df.empty else 0
    if previous_year_sales != 0:
        yoy_growth = ((ytd_sales - previous_year_sales) / previous_year_sales) * 100
    else:
        yoy_growth = 100 if ytd_sales > 0 else 0

    # MoM
    previous_month_sales = combined_df[
        (combined_df["Date"].dt.date >= previous_month_start) &
        (combined_df["Date"].dt.date <= previous_month_end)
    ]['Total'].sum() if not combined_df.empty else 0
    mom_change = mtd_sales - previous_month_sales

    # แสดงเมตริก
    a.metric(
        label="ยอดขายรวม",
        value=f"{total_sales:,.2f} บาท",
        border=True
    )
    b.metric(
        label="จำนวนออเดอร์",
        value=f"{total_orders:,}",
        border=True
    )
    c.metric(
        label="จำนวนลูกค้าใหม่",
        value=f"{new_customers:,}",
        border=True
    )
    d.metric(
        label="จำนวนลูกค้า",
        value=f"{unique_customers:,}",
        border=True
    )
    e.metric(
        label="MTD",
        value=f"{mtd_sales:,.2f} บาท",
        border=True
    )
    f.metric(
        label="YTD",
        value=f"{ytd_sales:,.2f} บาท",
        border=True
    )
    g.metric(
        label="YoY",
        value=f"{yoy_growth:.2f}%",
        border=True
    )
    h.metric(
        label="MoM",
        value=f"{mtd_sales:,.2f} บาท",
        delta=f"{mom_change:,.2f} บาท",
        border=True
    )

if __name__ == "__main__":
    main()