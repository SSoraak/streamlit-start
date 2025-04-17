import streamlit as st

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

def home_content():
    st.write("This is the home page of the Streamlit app.")
    st.info("Home Page")
    
    
# เอาไว้ที่นี่เพื่อให้สามารถเรียกใช้ได้จากทุกหน้า 
pages = {
    "Home": [
        st.Page(home_content, title="Home", icon="🏠"),],
    "Pose maintenance": [
        st.Page("pose_maintenance/dashboard_bi.py", title="Dashboard", icon="📊"),
        st.Page("pose_maintenance/prediction.py", title="Prediction", icon="🔮")
    ],
    "Dashboard": [
        st.Page("pages/dashboard_power_bi.py", title="Dashboard (All)", icon="📊"),
        
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

pg = st.navigation(pages)
pg.run()



