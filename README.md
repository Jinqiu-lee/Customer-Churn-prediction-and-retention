# Customer Churn Prediction System

## Turning Customer Data into Retention Decisions


### 1. Determin the Business Problem .
### Declining subscriptions or customer decline: 
    - Which customers are likely to leave — and why?”
    - The relationship between subscriber profiles and subscription volume)

    
### 2. Business Impact
### This system helps businesses:
  - Identify customers most at risk of leaving
  - Understand why they might churn
  - Prioritize who to save
  - Allocate retention budget/strategy on them more effectively


### 3. The Solution
### I built an churn prediction system that:
  - Self-tune the parameters to choose the ML model based on customer profile, to predict which customers are likely to churn
  - Returns the churn probability
  - Explains what factors are driving their risk
  - Compares different models to ensure reliable decisions.
  - Retention strategy suggestions. 

This turns raw customer data into clear, actionable business insights, because gaining new customers are 5X more expensive than retent existed customer


### 4. What This System Delivers: 
For every customer, the system provide 
1. Churn probability ---- How likely a given customer is to leave
2. Churn reasons ---- Data Exploration page clearly showed the visulization and explanation. 
3. Main drivers ---- Why this customer is predicted to churn


### 5. How It Works ?
1. Customer data (contract type, tenure, service-type etc.) is analyzed
2. Data Analytic page explored patterns who's in high risk of churn, and model page(4 models) learned patterns from past customers who left or stayed
3. Models predicts churn risk for current customers
4. Explanations show which factors influence each prediction


### 6. What Makes This Project Better than Typical Models?
Most churn models only output: “This customer will churn.”
This system instead provides:
  - Four models (Logistic Regression & XGBoost & Ramdon Forest & MLP)
    - Probability scores, not just yes/no
    - Threshold calibration to match business risk tolerance
    - SHAP explanations to show why each prediction happens

This makes it suitable for real decision-making, not just experimentation.

### 7. Who It Is For ? 
1. Subscription businesses
2. SaaS companies
3. Telecom 
4. E-commerce & loyalty programs
5. Any company with recurring customers, of you have customer data, this system can turn it into a retention strategy.

### 8. Demo & How to Use ?
The project includes:
  - Trained machine learning models
  - A Streamlit dashboard with data exploration page and model page 
  - Data pipelines

You can:
  - See top 5 churn-driven factors
  - Choose customer's profile customer data
  - View churn risk
  - Know why they are at risk and what the retention strategy should take
