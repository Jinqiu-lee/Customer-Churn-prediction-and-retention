import streamlit as st
from sklearn import datasets
import os
import plotly.express as px
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import joblib
from model.preprocessing import preprocess_data
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from sklearn.decomposition import PCA
from sklearn.inspection import permutation_importance


import warnings
warnings.filterwarnings('ignore')

    
st.header("📘 Project Overview")

with st.expander("🧠 What is Customer Churn ?",expanded=False):
    st.markdown("""
    Customer churn is when customers **leaves a business or stops using its services**, such as subscription-based company like :
    - Telecoms --- True,AIS Thailand,China Mobile,Verizon,Vodafone, 
    - Streaming Services --- Netflix, HBO Max,spotify, Tencent Video,iQIYI, 
    - SaaS platforms --- Alibaba Cloud,Salesforce,Microsoft365,Kahoot(Norway) 

    Churn means "Lost Revenue" and "Customer Lifetime Value"
    """)

    
with st.expander("💡 Why Does Predicting Churn Matter?", expanded=False):
    st.markdown("""
    Acquiring new customers is 5x more expensive than keeping existing ones. Losing customers hurts revenue and growth.
    By predicting who is likely to leave, businesses can:
    - Target high-risk customers with personalized offers
    - **Improve customer satisfaction** by adressing pain points
    - **Save costs** and **increase revenues** by focusing retention efforts efficiently.
                
    """)
                
with st.expander(" 📊 About the Data and Limitations", expanded=False):
    st.markdown("""
    **Dataset**:
    The dataset used comes from a real-world telecom company and includes:
    - Customer demographics (e.g., gender, dependant, tenure, contract type)
    - Services used (e.g., internet, phone, streaming)
    - Billing & payment behavior
    - Whether the customer churned

    **Limitations**:
    - No behavioral or customer feedback data (e.g., complaints, satisfaction scores)
    - Class imbalance (e.g., the churned customers 1 has much less than stayed customers 0)
    - Some features may be outdated or specific to one company""")

with st.expander("🎯 Project Goal &  📈 Use Cases", expanded=False):
    st.markdown("""
    This streamlit dashboard allows businesses to :

    ✅ Explore the trained ML models, upload new customer data, predict churn risk

    ✅ Explore key trends with interactive visualizations

    ✅ Take actions with actionable insights (e.g., personalized retention campaigns)
    - Customer service teams can prioritize outreach to high-risk customers.
    - Marketing can run targeted retention campaigns.
    - Executives can visualize churn trends by segment and plan improvements.

    Perfect for: Telecom, SaaS, e-commerce, and subscription-based businesses!

    Turn insights into action —-- before customers walk away! 🚀
    """)
    