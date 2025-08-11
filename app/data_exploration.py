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


st.header("Data Exploration and Visualization")

df = pd.read_csv("./data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
raw_data = df.copy()
processed_data = preprocess_data(df)

train_X = processed_data[3]
train_y= processed_data[4]
test_X = processed_data[5]
test_y = processed_data[6]

X_all = pd.concat([train_X, test_X], axis=0)
y_all = pd.concat([train_y, test_y], axis=0)

X_all.reset_index(drop=True, inplace=True)
y_all.reset_index(drop=True, inplace=True)

df_processed = X_all.copy()
df_processed['Churn'] = y_all
print(df_processed)

raw_data = raw_data.drop(['customerID'],axis=1)
raw_data['Churn'] = raw_data['Churn'].str.strip()   
raw_data['Churn'] = raw_data['Churn'].map({'Yes':1,'No':0}) 

show_raw = st.checkbox("Show Raw Data")  
show_processed = st.checkbox("Show Processed Data and EDA conclusion")

if show_raw:
    st.dataframe(raw_data) 
    st.write("##### Data shape: ",raw_data.shape)
        
        
if show_processed:
        st.write(df_processed)
        st.write("##### Data Shape:",df_processed.shape)
        c = st.container()
        correlation = df_processed.corr()
        
        # Filter only columns that correlate with 'Churn' above a threshold (e.g., |corr| > 0.2)
        churn_corr = correlation['Churn'].sort_values(ascending=False)
        high_corr = churn_corr[churn_corr > 0.2].index
        filterd_corr = df_processed[high_corr].corr()
        
        fig ,ax = plt.subplots(figsize =(10,6))
        sns.heatmap(filterd_corr,annot=True,cmap='coolwarm',fmt=".2f",ax=ax)
        c.subheader(" 🔄  Correlation Heatmap (High Correlation with Churn)")
        c.pyplot(fig)
        with c.expander("🔍 Correlation Heatmap Insights (EDA Summary)"):
            st.markdown("""
                        ##### Observation :
                        - 📄 Contract_Month-to-month (0.41)
                        - 🔐 OnlineSecurity_No and 🛠️ TechSupport_No (both at 0.34)
                        - 🌐 InternetService_Fiber optic (0.31)
                        - 💳 PaymentMethod_Electronic check (0.30)
                        ##### Insights :
                        These findings show that a **Month-to-month** customer,with **Fiber optic InternetService** but **lacking security or support services**, or with **Eletronic check payments** are mostly likely to churn.
                        These findings combine with the raw data EDA, reinforces the importance of **service offerings and price dissatisfaction** and **payment flexibility** in customer retention strategies
                        """)
            
                
c = st.container()  
c.subheader("EDA --- Customer Churn")
churn_count = raw_data['Churn'].value_counts().reset_index()
churn_count.columns = ['Churn','Count']
col1, col2 = c.columns([2, 2]) 
with col1:
    st.write("##### Customer Churn Distribution 📊")
    fig =px.bar(churn_count,x="Churn",y="Count",color='Churn')
    st.plotly_chart(fig)
        
with col2:
    st.write("Churn Class Distribution (Raw Count)")
    st.dataframe(raw_data['Churn'].value_counts().rename_axis('Churn').reset_index(name='Count'))
    st.write("Churn Class Distribution (Percentage)")
    st.dataframe(raw_data['Churn'].value_counts(normalize=True).mul(100).round(2).rename_axis('Churn').reset_index(name='Percentage (%)'))

with c.expander("⚠️ ⚖️ Class Imbalance Problem："):
    st.markdown("""
                - In this dataset ,73% of non-churn("0" ), 27% churned("1"),the model might naively predict "0" everytime.
                        - The model becomes biased toward predicting the majority class, it will achieve **high accuracy**,but with **low recall** on churners  
                        - Hurt performance in detecting **actual churners**, which is often the **business goal** """)
        
with c.expander("🛠️ How I will address it ："):
    st.markdown("""
                        - Apply **SMOTE**, **resampling** and **class weights** 
                        - Focus on **Evaluation Metrics** like F1-score, Precision and Recall - not just Accuracy
                """)
            
c = st.container() 
c.subheader(" EDA  --- Features Impacting Customer Churn")
c.markdown("##### 👩🏻‍💻 High Priority Features for Churn --- Contract,MontlyCharges")
    
col1, col2,col3 = c.columns([2,2,2])  # wider right column  

filter_df1 = raw_data[raw_data['Contract'] == 'Month-to-month']
filter_df2 = raw_data[raw_data['Contract'] == 'One year']
filter_df3 = raw_data[raw_data['Contract'] == 'Two year']
churn_pct1= filter_df1['Churn'].value_counts(normalize=True).mul(100).round(2)
churn_pct2= filter_df2['Churn'].value_counts(normalize=True).mul(100).round(2)
churn_pct3= filter_df3['Churn'].value_counts(normalize=True).mul(100).round(2)

with col1:
    fig = px.histogram(raw_data,
                        x='Contract',color='Churn',barmode='stack',
                        title='Churn by Contract Type')
    st.plotly_chart(fig,use_container_width=True)
    with st.expander("Churn Percentage by contract type"):
        st.write(f"Month-to-month:{churn_pct1[1]}%")
        st.write(f"One year :{churn_pct2[1]}%")
        st.write(f"Two year:{churn_pct3[1]}%")
                            
with col2:
    fig = px.box(raw_data,x='Churn',y='MonthlyCharges',
                    title="MonthlyCharges vs. Churn")
    st.plotly_chart(fig,use_container_width=True)
            
with col3:
    fig = px.box(raw_data,x='Contract',y='MonthlyCharges',
                    title="MonthlyCharges & Contract Combo vs. Churn")
    st.plotly_chart(fig,use_container_width=True)
        
with c.expander(" 🔎 Insight and Strategies from Contract Type and MonthlyChagres: "):  
    st.markdown("""
            - 📝 **Month-to-month** tend to high churn(42.71%), **Two year** churn very few(2.83%),**One year** churn rate 11.27%
            - 📈 **Month-to-month** have the **highest median charges** (~$75–80)
            - 📉 **Two-year contracts** have the **lowest median charges** (~$65)
            - 📑 Non-churners monthly median charges 64.5, majority paid less than median, while Churners monthly median charges about 80, which means **price dissatisfaction** woule be very likely be the churn driver, and it happens mostly to **Month-to-month** customers
            - 🔄 Some Non-churners also pay very high, maybe they have **Internetservice or other service**
            """)


c = st.container()
c.markdown("#### 👩🏻‍💻 High Priority Features for Churn --- Tenure")
raw_data['TenureGroup'] = pd.cut(raw_data['tenure'], bins=[0, 12, 24, 48, 72], labels=['<1yr', '1-2yr', '2-4yr', '4-6yr'])      

tenure1_df = raw_data[raw_data['tenure']<12]
churn_count1 = tenure1_df['Churn'].value_counts()
churn_rate1 =  (churn_count1/churn_count1.sum()*100).round(2)

tenure2_df = raw_data[(raw_data['tenure'] >= 12) & (raw_data['tenure'] < 24)]
churn_count2 = tenure2_df['Churn'].value_counts()
churn_rate2 = (churn_count2/churn_count2.sum()*100).round(2)

tenure3_df = raw_data[(raw_data['tenure']>=24) & (raw_data['tenure']< 48)]
churn_count3 = tenure3_df['Churn'].value_counts()
churn_rate3 = (churn_count3 /churn_count3.sum()*100).round(2)

tenure4_df = raw_data[raw_data['tenure'] > 48]
churn_count4 = tenure4_df['Churn'].value_counts()
churn_rate4 = (churn_count4 /churn_count4.sum()*100).round(2)

col1, col2 = c.columns([2,2])
with col1:
    fig = px.box(raw_data,x='Churn',y='tenure',title="Tenure vs. Churn")
    st.plotly_chart(fig,use_container_width=True)

with col2:
    fig = px.histogram(raw_data,x='TenureGroup',color='Churn',barmode='group',
                        title='Churn by TenureGroup')
    st.plotly_chart(fig,use_container_width=True)
    with st.expander("Churn Rate by differentiate Tenure groups:"):
        st.write(f"< 1 yr : {churn_rate1[1]}%")
        st.write(f"1-2 yrs : {churn_rate2[1]}%")
        st.write(f"2-4 yrs: {churn_rate3[1]}%")
        st.write(f"4-6 yrs: {churn_rate4[1]}%")

describe_df = raw_data.groupby(['TenureGroup', 'Churn'])['MonthlyCharges'].describe().round(2)
c.markdown("##### Monthly Charges Distribution by Tenure and Churn ")
c.dataframe(describe_df)
with c.expander(" 🔎 Insights and Strategies from Tenure: "):
    st.markdown("""
                ###### Short Tenure cause high Churn, New customers churn early (<1 yr),need to foucus on **New customers(<1 yr)**
                - <1 yr , Churners pay 40% more than Non-churner.
                - 1-2yrs, Churners pay 44% more than Non-churner, Q1 Churners pay more than 3 times of the price than Non-churners
                - 2-4 yrs, Churners pay 37% more than Non-churner, Q1 Churners pay 3 times of the price than Non-churners
                - 4-6yrs, Churners pay 27% more than Non-churners, Q1 Churners pay almost double of the price than Non-churners
                ###### Strategies :  **Price** is a big churn drive,**Short-term and new Customer** is another churn drive
                - For < 1 yr New customers , might be commitment issue, price issue, service unsatisfied with competitors etc, focus on **increase their stickness and commmitment**, offer **discount for yearly contract** and provide automatic service as bonus for convenience/stickness 
                - For 4-6yrs Churners customers ,focus on why they pay much higher price (Q1), maybe other service they are using are not satisfied. Offer **customized service/support** to maintain their satifaction and loyalty 
                """)


c = st.container()
c.markdown("#### 👩🏻‍💻 High Priority Features for Churn --- InternetService")

# Create a new column for service combinations
raw_data['Service_combo']=raw_data.apply(lambda x :f"OS:{x['OnlineSecurity']},TS:{x['TechSupport']},DP:{x['DeviceProtection']}",axis=1)
# calculate combo churn 
combo_churn = raw_data.groupby('Service_combo')['Churn'].mean().reset_index()

col1,col2= c.columns([2,2])
with col1:
    fig = px.histogram(raw_data,color='Churn',x='InternetService',barmode='stack',title='InternetService vs.Churn')
    st.plotly_chart(fig,use_container_width=True)
    
with col2:
    fig = px.bar(combo_churn,x='Service_combo',y='Churn',
                    color='Churn',# Color bars by value
                    color_continuous_scale='Viridis',
                    title="Churn Rates vs. Service Combination")
        
    st.plotly_chart(fig,use_container_width=True)

describe_df_service = raw_data.groupby(['InternetService', 'Churn'],observed=True)['Service_combo'].describe()
describe_df_contract = raw_data.groupby(['InternetService', 'Churn'],observed=True)['Contract'].describe()
describe_df_charges = raw_data.groupby(['InternetService','Churn'],observed=True)['MonthlyCharges'].describe().round(2)

c.markdown("##### ％ Churn Rates with or without InternetService per contract and MonthlyCharges ")

col1,col2,col3=c.columns([1,1,1])
with col1:
    st.dataframe(describe_df_service)
    
with col2:
    st.dataframe(describe_df_contract)

with col3:
    st.dataframe(describe_df_charges)

with c.expander(" 🔎 Insight and Strategies with InternetService per MonthlyCharges and Contract: "):
    st.markdown("""
                ##### Observations:
                - No InternetService category has only 1 churn(7.4%), InternetService customers with 3 Service_combo also has very few churn (7.15%),InternetService but lacking all three services category has very high-risk to churn (52.6%), InternetService has at least two Services churn rate dropped immediately to (18%, 15%, 12%)
                - For No InternetService: Non-churners pay average 21.14＄, majority are Two year contract, Churners majority are **Month-to-month**, they pay even less 20.27＄
                - For DSL: Majority of both Churners and Non-churners are **Month-to-month**, Non-Churners pay average 60＄, more than Churners 49＄
                - For Fiber optic: Majority of both Churners and Non-churners **lack all three services**,and both of them are **Month-to-month** contract, both of them pay the highest average Non-churn(93.9＄),Churn (88.1＄)
                ##### Strategies :
                - For InternetService Customers: they are mostly **Month-to-month** customers, they uaually pay higher than one-year and two-year contract customers, they are not fully commited, they are also considering other competitor with better services and lower prices
                - For InternetService itself : DSL has very low churn, very reasonable price(60＄) with all three services , slightly higher than DSL without any service(49＄); Fiber optic, both Churners and non-churners mostly are lacking three services, the average price is too high.  
                - Need to focus on : 1️⃣ Investigate on the satisfaction and User Experience of Fiber Optic,focus on how to increase **Fiber optic Services and price satisfaction**, can provide 30% **discount for service_combo** ,instead of seperately choosing 1, 2 or 3 services; 2️⃣ Provide 20% **upgrade discount for Month-to-month customers** to upgrade their contract type for stickness, or upgrade Service_combo from the current Internet; 3️⃣ Provide a new **yearly contract plan** with InternetService and multiple service Package, add **customized services/support** to increase after-sale satisfaction. 
                """)

    

st.markdown("#### 👩🏻‍💻 High Priority Features for Churn --- PaymentMethod and PaperlessBilling")
c = st.container()  

col1,col2 = c.columns([2,2])
with col1:
    fig1= px.histogram(raw_data,x='Churn',y='PaymentMethod',title='PaymentMethod vs. Churn %')
    st.plotly_chart(fig1,use_container_width=True)
with col2:
    fig2 = px.histogram(raw_data,x='Churn',y='PaperlessBilling',title='PaperlessBilling vs. Churn %')
    st.plotly_chart(fig2,use_container_width=True)
    
col3,col4 = c.columns([2,2])
with col3:    
    fig3 = px.histogram(raw_data,x='Contract',color='PaymentMethod',title='Contract type vs. PaymentMethod')
    st.plotly_chart(fig3,use_container_width=True)
with col4:
    fig4 = px.histogram(raw_data,x='Contract',color='PaperlessBilling',title='Contract type vs. PaperlessBilling')
    st.plotly_chart(fig4,use_container_width=True)
    
col5,col6 = c.columns([2,2])
with col5:
    describe_df_pay= raw_data.groupby(['PaymentMethod', 'Churn'],observed=True)['Contract'].describe()
    st.dataframe(describe_df_pay)   
with col6:
    describe_df_paper = raw_data.groupby(['PaperlessBilling','Churn'],observed=True)['Contract'].describe()
    st.dataframe(describe_df_paper)
with c.expander(" 🔎 Insights and Strategies from PaperlessBilling and PaymentMethod per contract type"):
    st.markdown("""
                ###### Observations:
                - Eletronic check has the highest risk (45.3%)to churn, majority(47.7%) are **Month-to-month**. Credit card(automatic) and Bank trasnfer both have low risk to churn and majority are two-year contract.
                - PaperlessBilling has almost 3 times more churn than Non-PaperlessBilling customers,for both Churners and Non-churners with PaperlessBilling majority are **Month-to-month**, even with Paper bill, 86.56% of Month-to-month customers still churn.
                ##### Strategies:
                - Offer incentives for customers with non-PaperlessBilling(inconvenience) and auto-pay(convenience), add multiple payment reminder
                - Based on Monthlycharges,Tenure,InternetSerice, PaymentMethod and PaperlessBilling, it's clear that Month-to-month customers has the highest risk to churn, need to focus on **Month-to-month Customers**
                - Offer 20% **discount or additional InternetService package** for them to **upgrade to yearly contract**, offer incentives for **auto-pay and stay** (for convenience), add multiple **payment reminder**.
                """)


st.markdown("#### 👩🏻‍💻 High Priority Features for Churn --- Dependents")
c = st.container()

churn_rate_dependent = raw_data.groupby('Dependents')['Churn'].mean().reset_index()
churn_rate_dependent['Churn_rate'] = churn_rate_dependent['Churn']

col1,col2,col3= c.columns([2,2,2])
with col1:
    fig = px.histogram(churn_rate_dependent,x='Dependents',y='Churn_rate',title='Dependents vs. Churn rate')
    st.plotly_chart(fig,use_container_width=True)
    
with col2:
    fig = px.histogram(raw_data,x='Contract',color='Dependents',title='Dependents vs. Contract')
    st.plotly_chart(fig,use_container_width=True)

with col3:
    fig = px.box(raw_data,x='Dependents',y='MonthlyCharges',title='Dependents vs. MonthlyCharges')
    st.plotly_chart(fig,use_container_width=True)
    
    
with c.expander(" 🔍 Insights and Strategies for Dependents or Non-dependents"):
    st.markdown("""
                - Non-dependents churn twice more than Dependents, Majority of Non-dependents are **Month-to-month**, Non-dependents also pay 20% higher than Dependents. Many of Dependents pay less than median.
                - Dependents might rely on their parents to pay, which pays less and prefer to choose yearly contract plan(62.6%),it's clear to see from the plot, so the reason of churned dependents might still the **Month-to-month Contract with price dissatisfaction**
                ##### Strategies
                - For new Dependents : Provide promotion of **yearly family plan** with different service packages, to keep convenience and commitment.
                - For current Non-dependents: Offer 30% discount to upgrade to yearly plan with auto-pay, also provide discount for multiple service_combo, promote DSL.
                """) 
    
st.markdown("#### 👩🏻‍💻 High Priority Features for Churn --- Partner")
c = st.container()    
churn_rate_partner = raw_data.groupby('Partner')['Churn'].mean().reset_index()
churn_rate_partner['Churn_rate'] = churn_rate_partner['Churn']

col1,col2,col3= c.columns([2,2,2])
with col1:
    fig = px.histogram(churn_rate_partner,x='Partner',y='Churn_rate',title='Partner vs. Churn')
    st.plotly_chart(fig,use_container_width=True)    
with col2:
    fig = px.histogram(raw_data,x='Contract',color='Partner',title='Partner vs. Contract')
    st.plotly_chart(fig,use_container_width=True)
with col3:
    fig = px.box(raw_data,x='Partner',y='MonthlyCharges',title='Partner vs. MonthlyCharges')
    st.plotly_chart(fig,use_container_width=True)
        
with c.expander(" 🔍 Insights and Strategies for Customers with or without Partner"):
    st.markdown("""
                ##### Observations and Insights
                - Customers with partner churn 19.6%, without partner churn 32.9%. 60% of the Customers with partners choose yearly plan. Customers with partners pay slightly higher than customers without partners.
                - Customers with partner are more stable, they are able to pay higher price with good service,more likely to choose yearly plan, less likely to switch to competitiors if they are satisfied.  
                ##### Strategies 
                - For new customers with partner: Provide promotion of **yearly family plan** with multiple service choices, promote DSL
                - Still need to focus more on **Month-to-month Customers**: Offer 20% discount to upgrade to yearly plan with auto_pay, promote DSL for customer without InternetService, offer 30% discount Service_combo for Current Fiber optic customers
                """)

