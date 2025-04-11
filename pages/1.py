import streamlit as st


with st.sidebar:
    st.title("Navigation")
    #สร้าง expander สำหรับแต่ละหมวดหมู่




st.title("Page 1")
st.write("This is the first page of the Streamlit app.")
with st.expander("Expander"):  
    st.info("Page 1")
    st.success("This is a success message.")
    st.warning("This is a warning message.")
    st.error("This is an error message.")
    st.exception(Exception("This is an exception message."))