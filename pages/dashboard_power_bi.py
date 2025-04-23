import streamlit as st

# st.set_page_config(page_title="Power BI Dashboard Viewer", layout="wide")

# ข้อมูล Dashboard
dashboards = [
    {
        "name": "Power BI Pose Health Care",
        "url": "https://app.powerbi.com/reportEmbed?reportId=33dfc85a-36fd-4a4c-88ab-1c945c17e810&autoAuth=true&ctid=1f23f2e8-f8b5-4438-a7d7-1d2d87dffcf3",
        "description": "Pose Health Care Overall Dashboard",
        "owner": ["Tongtang","Night"],
        "tags": ["Pose", "Pose Health Care", "PHC"],
        "order": 1.000,
        "show": True
    },
    {
        "name": "Pose Health Care",
        "url": "https://app.powerbi.com/reportEmbed?reportId=80ae4777-0974-4821-b0ee-8f0f2e8a9406&autoAuth=true&ctid=1f23f2e8-f8b5-4438-a7d7-1d2d87dffcf3",
        "description": "Pose Health Care Overall Dashboard",
        "owner": ["Night"],
        "tags": ["Pose", "Pose Health Care", "PHC"],
        "order": 1.001,
        "show": True
    },
    {
        "name": "PHC Sales per Area",
        "url": "https://app.powerbi.com/reportEmbed?reportId=8d0bd288-526a-4501-8ebb-779936ee6451&autoAuth=true&ctid=1f23f2e8-f8b5-4438-a7d7-1d2d87dffcf3",
        "description": "PHC Sales per Area and type of hospital",
        "owner": ["Night"],
        "tags": ["Pose", "Pose Health Care", "PHC"],
        "order": 1.002,
        "show": True
    },
    {
        "name": "PHC Sales BOI",
        "url": "https://app.powerbi.com/reportEmbed?reportId=4e5333e8-6aa3-4dd9-a83c-043d8cc6b8a3&autoAuth=true&ctid=1f23f2e8-f8b5-4438-a7d7-1d2d87dffcf3",
        "description": "PHC Sales products BOI",
        "owner": ["Night"],
        "tags": ["Pose", "Pose Health Care", "PHC", "BOI"],
        "order": 1.003,
        "show": True
    },
    {
        "name": "PHC Finished Goods",
        "url": "https://app.powerbi.com/reportEmbed?reportId=e5541fd6-4e3c-4279-87ed-8668e54e0d90&autoAuth=true&ctid=1f23f2e8-f8b5-4438-a7d7-1d2d87dffcf3",
        "description": "PHC Sales and Production Finished Goods",
        "owner": ["Night"],
        "tags": ["Pose", "Pose Health Care", "PHC", "Finished Goods" , "Production"],
        "order": 1.004,
        "show": True
    }, 
    {
        "name": "PHC Production Products",
        "url": "https://app.powerbi.com/reportEmbed?reportId=b013f1ef-bd9f-4cfd-977d-a21f78389b48&autoAuth=true&ctid=1f23f2e8-f8b5-4438-a7d7-1d2d87dffcf3",
        "description": "PHC Production Finished Goods details item",
        "owner": ["Night"],
        "tags": ["Pose", "Pose Health Care", "PHC", "Finished Goods", "Production", "Packaging"],
        "order": 1.005,
        "show": True
    }, 
    {
        "name": "PHC Follow up on sellers",
        "url": "https://app.powerbi.com/reportEmbed?reportId=c6945a86-1650-453f-b5e5-e76ed69b10e6&autoAuth=true&ctid=1f23f2e8-f8b5-4438-a7d7-1d2d87dffcf3",
        "description": "PHC details on sellers and hospital visit",
        "owner": ["Night"],
        "tags": ["Pose", "Pose Health Care", "PHC", "sellers", "hospital", "visit", "แบบสอบถาม"],
        "order": 1.006,
        "show": True
    }, 
    {
        "name": "PHC Sales Customer",
        "url": "https://app.powerbi.com/reportEmbed?reportId=2929578c-e414-4c27-8ca6-4b1466928bbf&autoAuth=true&ctid=1f23f2e8-f8b5-4438-a7d7-1d2d87dffcf3",
        "description": "PHC customer analysis",
        "owner": ["Night"],
        "tags": ["Pose", "Pose Health Care", "PHC", "sellers", "hospital", "Customer", "RMF Analysis"],
        "order": 1.007,
        "show": True
    }, 
    {
        "name": "PHC Sales P071",
        "url": "https://app.powerbi.com/reportEmbed?reportId=4b51764b-9441-4abf-b0e1-469c6c435d39&autoAuth=true&ctid=1f23f2e8-f8b5-4438-a7d7-1d2d87dffcf3",
        "description": "PHC Sales analysis",
        "owner": ["Night"],
        "tags": ["Pose", "Pose Health Care", "PHC",  "P071"],
        "order": 1.008,
        "show": True
    }, 
    {
        "name": "Linen Dashboard",
        "url": "https://app.powerbi.com/reportEmbed?reportId=c6945a86-1650-453f-b5e5-e76ed69b10e6&autoAuth=true&ctid=1f23f2e8-f8b5-4438-a7d7-1d2d87dffcf3",
        "description": "Linen management",
        "owner": ["Night"],
        "tags": ["Linen"],
        "order": 2.001,
        "show": True
    }, 
    {
        "name": "Linen Praram9",
        "url": "https://app.powerbi.com/reportEmbed?reportId=772ecc13-424c-43c8-97df-95be62296e97&autoAuth=true&ctid=1f23f2e8-f8b5-4438-a7d7-1d2d87dffcf3",
        "description": "Linen management on Praram9 hospital",
        "owner": ["Night"],
        "tags": ["Linen", "Praram9"],
        "order": 2.002,
        "show": True
    }, 
    {
        "name": "Linen RFID Management",
        "url": "https://app.powerbi.com/reportEmbed?reportId=9ed34ae5-f52f-46bc-99cb-b9243b8c0e00&autoAuth=true&ctid=1f23f2e8-f8b5-4438-a7d7-1d2d87dffcf3",
        "description": "Linen management",
        "owner": ["Night"],
        "tags": ["Linen", "RFID"],
        "order": 1.003,
        "show": True
    }, 
    {
        "name": "Linen G Nimman",
        "url": "https://app.powerbi.com/reportEmbed?reportId=a866fce9-0f02-45c3-9603-11c8dcc7ab73&autoAuth=true&ctid=1f23f2e8-f8b5-4438-a7d7-1d2d87dffcf3",
        "description": "Linen management on G Nimman hotel",
        "owner": ["Night"],
        "tags": ["Linen", "G Nimman"],
        "order": 2.004,
        "show": True
    }, 
    {
        "name": "ITSM",
        "url": "https://app.powerbi.com/reportEmbed?reportId=a866fce9-0f02-45c3-9603-11c8dcc7ab73&autoAuth=true&ctid=1f23f2e8-f8b5-4438-a7d7-1d2d87dffcf3",
        "description": "Project management for rama มหิดล",
        "owner": ["Night"],
        "tags": ["Linen", "ITSM"],
        "order": 3.001,
        "show": False
    }, 
    {
        "name": "Dental BHQ",
        "url": "https://app.powerbi.com/reportEmbed?reportId=12c2f73b-5c32-4540-ac3f-05c07267347c&autoAuth=true&ctid=1f23f2e8-f8b5-4438-a7d7-1d2d87dffcf3",
        "description": "Instrument management for Dental",
        "owner": ["Night"],
        "tags": ["Linen", "BHQ", "Dental"],
        "order": 4.001,
        "show": True
    }, 
    {
        "name": "CSSD Dental BHQ",
        "url": "https://app.powerbi.com/reportEmbed?reportId=d1fdd32b-12e7-4f9d-a6e7-09c293e45db5&autoAuth=true&ctid=1f23f2e8-f8b5-4438-a7d7-1d2d87dffcf3",
        "description": "Instrument management for Dental and CSSD",
        "owner": ["Night"],
        "tags": ["Linen", "BHQ", "Dental", "CSSD"],
        "order": 4.002,
        "show": True
    }, 
        {
        "name": "Inventory OR Sriphat",
        "url": "https://app.powerbi.com/reportEmbed?reportId=24f931e5-b790-4a71-ab6e-57300d50d33b&autoAuth=true&ctid=1f23f2e8-f8b5-4438-a7d7-1d2d87dffcf3",
        "description": "Instrument management for OR",
        "owner": ["Night"],
        "tags": [ "Inventory OR", "OR", "Sriphat"],
        "order": 5.001,
        "show": True
    }, 
]

st.markdown("<h1 style='text-align: left;'>📊 Power BI Dashboards</h1>", unsafe_allow_html=True)



# เตรียมค่าทั้งหมดพร้อม "ทั้งหมด" เป็นตัวเลือกแรก เรียงตามลำดับ
# ใช้ set เพื่อไม่ให้มีค่าซ้ำแล้วแปลงกลับเป็น list
# และเรียงลำดับใหม่ต่าม order
# แสดงเฉพาะข้อมูลที่มี show = True
dashboards = [dash for dash in dashboards if dash["show"]]
all_dashboard_names = ["ทั้งหมด"] + [dash["name"] for dash in sorted(dashboards, key=lambda x: x["order"])]
all_descriptions = ["ทั้งหมด"] + sorted({dash["description"] for dash in dashboards})
all_tags = ["ทั้งหมด"] + sorted({tag for dash in dashboards for tag in dash["tags"]})



# --- Sidebar Filters ---
st.sidebar.header("🔍 ค้นหา Dashboard")

selected_dashboard = st.sidebar.selectbox("Dashboard", all_dashboard_names, key="dashboard_selector")
selected_description = st.sidebar.selectbox("Description", all_descriptions, key="description_selector")
selected_tag = st.sidebar.selectbox("Tag", all_tags, key="tag_selector")


# กรองข้อมูลตามตัวเลือก
filtered_dashboards = []
for dash in dashboards:
    match_dashboard = selected_dashboard == "ทั้งหมด" or dash["name"] == selected_dashboard
    match_description = selected_description == "ทั้งหมด" or dash["description"] == selected_description
    match_tag = selected_tag == "ทั้งหมด" or selected_tag in dash["tags"]

    if match_dashboard and match_description and match_tag:
        filtered_dashboards.append(dash)

# แสดง Dashboard ที่กรองแล้ว
if filtered_dashboards:
    for dash in filtered_dashboards:
        st.markdown(f"### {dash['name']}")
        st.markdown(f"""
            <iframe title="{dash["name"]}" 
            width="100%" 
            height="750" 
            src="{dash["url"]}" 
            frameborder="0" 
            allowFullScreen="true"></iframe>
        """, unsafe_allow_html=True)
        st.markdown(f"**Description:** {dash['description']}  \n"
                    f"**Owner:** {', '.join(dash['owner'])}  \n"
                    f"**Tags:** {', '.join(dash['tags'])}")
else:
    st.warning("ไม่พบ Dashboard ที่ตรงกับเงื่อนไข")
