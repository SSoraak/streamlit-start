import streamlit as st


st.markdown("<h1 style='text-align: left;'>Machinery Maintenance</h1>", unsafe_allow_html=True)

st.header("📊 Dashboard Overview")
dashboard_url = "https://your-dashboard-url.com"  # <== ใส่ URL dashboard จริงตรงนี้

st.markdown(f"""
    <iframe title="ข้อมูลเครื่องจักร MA (ธีม)" width="100%" height="750" 
    src="https://app.powerbi.com/reportEmbed?reportId=892b0a03-c03d-41d9-a3fd-79ac05cbca3d&autoAuth=true&ctid=1f23f2e8-f8b5-4438-a7d7-1d2d87dffcf3" 
    frameborder="0" allowFullScreen="true"></iframe>
    """, unsafe_allow_html=True)
