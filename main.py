import streamlit as st

st.set_page_config(layout="wide")

overview_page = st.Page("app/project_overview.py", title="Project Overview", icon="📽️")
data_page = st.Page("app/data_exploration.py", title="Data Exploration", icon="📊")
model_page = st.Page("app/model_evaluation.py", title="Model Prediction", icon="🤖")

pg = st.navigation([overview_page,data_page,model_page])
pg.run()




                