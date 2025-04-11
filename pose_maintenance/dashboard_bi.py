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


st.markdown("<h1 style='text-align: left;'>Machinery Maintenance </h1>", unsafe_allow_html=True)

st.header("📊 Dashboard Overview")
dashboard_url = "https://your-dashboard-url.com"  # <== ใส่ URL dashboard จริงตรงนี้
st.markdown(f"""
    <iframe title="ข้อมูลเครื่องจักร MA (ธีม)" width="100%" height="750" src="https://app.powerbi.com/reportEmbed?reportId=892b0a03-c03d-41d9-a3fd-79ac05cbca3d&autoAuth=true&ctid=1f23f2e8-f8b5-4438-a7d7-1d2d87dffcf3" frameborder="0" allowFullScreen="true"></iframe>
    """, unsafe_allow_html=True) 
