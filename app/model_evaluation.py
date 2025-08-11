import streamlit as st
from sklearn import datasets
import os
import pandas as pd
import numpy as np
import joblib
from model.preprocessing import preprocess_data,scaled
from model.train_models import train
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix,precision_score,recall_score,f1_score
from sklearn.inspection import permutation_importance
from config import init_config,reset_config
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv("./data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
num_cols, enc_cols,cat_cols,X_train,y_train, X_test,y_test = preprocess_data(df,save_test_csv=True)
X_scaled_train,X_scaled_test,y_train,y_test = scaled(save_test_csv=True)
    
col1, col2 = st.columns([2,1])

with col1:
    
    st.header("Model prediction and Evaluation")

    model_option = st.sidebar.selectbox("👇🏼 Select a model to load:",[
        "logistic_model.joblib",
        "xgb_model.joblib",
        "rf_model.joblib",
        "mlp_model.joblib",  
        ])

    model_path = f"./model/saved_model/{model_option}"

    init_config()

    # Load model
    @st.cache_resource
    def load_model(path):
        return joblib.load(path)

    model = load_model(model_path)
    st.success(f"Done Loaded{model_option}")

    max_depth = 5
    n_estimators = 500
    learning_rate = 0.1
    min_samples_split = 2

    if model_option == "xgb_model.joblib":
        st.sidebar.header("Tune paramater for xgb_model")
        max_depth = st.sidebar.slider("Max Depth", 1, 20,7)
        n_estimators = st.sidebar.slider("n_estimators", 100,800,500,step=100)
        learning_rate = st.sidebar.slider("Learning Rate",0.001,0.5,0.1)
    elif model_option =="rf_model.joblib":
        st.sidebar.header("Tuning parameters for rf_model")
        max_depth = st.sidebar.slider("Max Depth", 1, 20,5)
        n_estimators = st.sidebar.slider("n_estimators", 100,1000,800,step=100)
        min_samples_split = st.sidebar.slider("Min Samples Split", 2, 10,3)


    xgb_params = {
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "learning_rate":learning_rate,
    }

    rf_params = {
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "min_samples_split":min_samples_split,
    }

        

    # Upload test data 
    @st.cache_data
    def load_data():
        if model_option in ["logistic_model.joblib","mlp_model.joblib"]:
            test_data = pd.read_csv("./data/scaled_test_data.csv")
        else:
            test_data = pd.read_csv("./data/test_data.csv")
        return test_data
    test_data = load_data()

    X_test = test_data.drop("Churn",axis=1)
    y_true = test_data["Churn"]
    

    if model_option == "rf_model.joblib":
        model = RandomForestClassifier(**rf_params, random_state=42,n_jobs=-1)
        model.fit(X_train, y_train) 
    elif model_option == "xgb_model.joblib":
        model = XGBClassifier(**xgb_params, random_state=42,n_jobs=-1)
        model.fit(X_train, y_train)
    elif model_option == "logistic_model.joblib":
        model = LogisticRegression()
        model.fit(X_scaled_train, y_train)
    elif model_option == "mlp_model.joblib":
        model = MLPClassifier(random_state=42)
        model.fit(X_scaled_train, y_train)
     
    
    # Show Evaluation Metrics
    thresholds = np.arange(0.0, 1.0, 0.05)
    precisions = []
    recalls = []
    f1s = []
   
    if model_option in ["logistic_model.joblib","mlp_model.joblib"]:
        y_probs = model.predict_proba(X_scaled_test)[:,1] # get probability of class 1
    
        for t in thresholds:
            y_pred = (y_probs >= t).astype(int)
            precisions.append(precision_score(y_true, y_pred))
            recalls.append(recall_score(y_true, y_pred))
            f1s.append(f1_score(y_true, y_pred))
            
        selected_threshold = st.sidebar.slider("Select a threshold",0.0,1.0,0.5,0.05)  
        y_pred_sel = (y_probs >= selected_threshold).astype(int)
        cf = confusion_matrix(y_true,y_pred_sel,normalize='true')
        
        fig,ax = plt.subplots(1,2,figsize=(14,6))  # 1 row, 2 columns
        
        # left plot
        sns.heatmap(cf,annot=True,cbar=True,ax=ax[0])
        ax[0].set_xlabel('Prediction')
        ax[0].set_ylabel('Target')
        ax[0].set_title(f'{model_option} Confusion Matrix(Threshold{selected_threshold})')
    
        
        # right plot
        ax[1].plot(thresholds,precisions,label="Precision")
        ax[1].plot(thresholds, recalls, label='Recall')
        ax[1].plot(thresholds, f1s, label='F1 Score')
        ax[1].axvline(selected_threshold, color='gray', linestyle='--')  # show selected threshold
        ax[1].set_xlabel("Threshold")
        ax[1].set_ylabel("Score")
        ax[1].set_title("Threshold vs Precision, Recall, F1")
        ax[1].legend()
        
        st.pyplot(fig,use_container_width=False)
        
        report_dict = classification_report(y_true, y_pred_sel,output_dict=True)
        report_df = pd.DataFrame(report_dict).transpose()
        report_df = report_df.round(2)
        report_df = report_df.style.format("{:.2f}", na_rep="")
        report_df.index.name = 'Class / Avg'
        st.subheader("Classification Report (Detailed)")
        st.dataframe(report_df)
    
    else:
        y_probs = model.predict_proba(X_test)[:,1]
        for t in thresholds:
            y_pred = (y_probs >= t).astype(int)
            precisions.append(precision_score(y_true, y_pred))
            recalls.append(recall_score(y_true, y_pred))
            f1s.append(f1_score(y_true, y_pred))
        
        selected_threshold = st.sidebar.slider("Select a threshold",0.0,1.0,0.5,0.05)  
        y_pred_sel = (y_probs >= selected_threshold).astype(int)
      
        cf = confusion_matrix(y_true,y_pred_sel,normalize='true')
        fig,ax = plt.subplots(1,2,figsize=(16,5))
        sns.heatmap(cf,annot=True,cbar=True,ax=ax[0])
        ax[0].set_xlabel('Prediction')
        ax[0].set_ylabel('Target')
        ax[0].set_title(f'{model_option} Confusion Matrix(threshold{selected_threshold})')
        
        ax[1].plot(thresholds,precisions,label="Precision")
        ax[1].plot(thresholds, recalls, label='Recall')
        ax[1].plot(thresholds, f1s, label='F1 Score')
        ax[1].axvline(selected_threshold, color='gray', linestyle='--')  # show selected threshold
        ax[1].set_xlabel("Threshold")
        ax[1].set_ylabel("Score")
        ax[1].set_title("Threshold vs Precision, Recall, F1")
        ax[1].legend()
        
        st.pyplot(fig,use_container_width=False)

        report_dict = classification_report(y_true, y_pred_sel,output_dict=True)
        report_df = pd.DataFrame(report_dict).transpose()
        report_df = report_df.round(2)
        report_df = report_df.style.format("{:.2f}", na_rep="")
        report_df.index.name = 'Class / Avg'
        st.subheader("Classification Report (Detailed)")
        st.dataframe(report_df)
        
    with st.expander("Model Performance Analysis and Explaination"):
        st.markdown("""
                    - 📉 Precision : How many of the predicted churners are actually churned.
                    - 📈 Recall : How many of actual churners did the model catch.
                    - ✅ **RECALL** is the most important.
                    - ⚖️ Optimize **recall** to make sure the model catch as many churners as possible , **F1-score** as a secondary metric to balance not wrongly flagging loyal customers (precision).
                    - If recall is above 0.8,and F1-score is above 0.6, we're doing well.
                    """)
        
    st.markdown(f"🧪 Using a custom threshold of **{selected_threshold}** to classify churn based on predicted probabilities.")
    if model_option == "logistic_model.joblib" and selected_threshold == 0.2:
        st.write(" 🧠 When **threshold = 0.2**(recall = 0.87,f1-score = 0.61), Logistic Regression model performs the best to detect and predict churn")
    elif model_option == "xgb_model.joblib" and selected_threshold == 0.3:
        st.write(" 🧠 When max-depth = 7,n_estimators = 500,learning rate = 0.03,**threshold = 0.3**(recall=0.84,f1-score=0.60),XGBoost model performs the best and balanced,you can tune parameters to get higher recall based on business goal")
    elif model_option == "rf_model.joblib" and selected_threshold == 0.35:
        st.write(" 🧠 When max-depth = 5, n_estimators = 800,min-sample-split = 3, **threshold = 0.35**(recall = 0.89,f1-score = 0.60),random forest model performs the best and balanced, you can tune parameters to get higher based on business goal ")
    elif model_option == "mlp_model.joblib" and selected_threshold == 0.15:
        st.write(" 🧠 When threshold = 0.15(recall = 0.89, f1-score = 0.60),MLP model performs the best ")
    
    data = {
            "Scenario":["No model","Top 20% Risk Customers","Top 30% Risk Customers"],
            "Risk Customers ":[0,200,300],
            "Real Churners Saved":[0,148,252],
            "Estimated Revenue Saved":["$0","$14,800","$25,200"]
        }
    df = pd.DataFrame(data)
    
    st.subheader("🕴 Estimated Business Impact")
    st.markdown("""
                Even with the imperfect precision, if our model catches 70% of churners, company can offer **early retention deals** to the 30% churn-risk customers , that could:
                - Save 100s of customers per month(based on the dataset)
                - Improve revenue predictability (based on the recall result and assume 100$ revenue per customer)
                    """)
    st.dataframe(df)
   

    # Feature Importance
    if model_option in ["rf_model.joblib","xgb_model.joblib"]:
        st.subheader(f"Feature Importance of {model_option}")
        importance = pd.DataFrame({
            "Feature": X_test.columns,
            "Importance": model.feature_importances_
        }).sort_values("Importance", ascending=False)
        
        st.bar_chart(importance.set_index("Feature"))
        st.dataframe(importance.sort_values("Importance", ascending=False).head(10))

        
    elif model_option in ["logistic_model.joblib"]:
        st.subheader("Feature Importance of logistic model")
        importance = model.coef_[0]
        feature_importance = pd.DataFrame({
            'Feature':X_test.columns,
            'Importance':importance
        })
        feature_importance['AbsImportance']=np.abs(feature_importance['Importance'])
        feature_importance = feature_importance.sort_values(by='AbsImportance',ascending=False)
        
        st.bar_chart(feature_importance.set_index('Feature')['Importance'])
        st.dataframe(feature_importance.sort_values('AbsImportance',ascending=False).head(10))
    
        
    else:
        st.subheader("Feature Importance of mlp model")
        result = permutation_importance(model, X_test, y_true, n_repeats=10, random_state=42)
        mlp_importance = pd.DataFrame({
            'Feature':X_test.columns,
            'Importance':result.importances_mean,
            'Std':result.importances_std
        })
        
        mlp_importance['AbsImportance'] = mlp_importance['Importance'].abs()
        mlp_importance = mlp_importance.sort_values(by='AbsImportance',ascending=False)
        st.bar_chart(mlp_importance.set_index('Feature')['Importance'])
        st.dataframe(mlp_importance.sort_values('AbsImportance',ascending=False).head(10))
    st.markdown("""
                    ##### 🔍 Insights of📊 Correlation Heatmap &🤖 Model Performance
                    - 1. Combine with the Data Exploration(Correlation Heatmap), top correlated feature with churn:
                       - Contract_Month-to-month (0.41)
                       - OnlineSecurity and TechSupport(0.34)
                       - InternetService_Fiber optic(0.31)
                       - PaymentMethod_Eletronic check(0.30)
                    - It means the customers with **Month-to-month contract**,**Fiber optic without online security & tech suopport**,**Eletronic check payment**are most likely to churn 
                    - 2. Logistic / Random Forest / MLP  ➡️ top features: tenure, totalcharges 
                       - Good for long-term customers for trend detection 
                       - Good for customers where tenure and billing history are strong churn signals ,combine with dependants and internet services
                    - 3. XGBoost ➡️ top features: Contract_Month-to-month, InternetService_Fiber optic
                       - Better for behavior-pattern-based churn detection and monthly contract analysis
                       - Better for New customers with little tenure data
                    """)
    st.markdown("""
                    ##### 🔍 Strategies for a single customer
                    - Step1️⃣ → Model Selection based on profile
                    - Step2️⃣ → Risk Scoring 
                       - Run prediction from chosen model at a **threshold favoring high recall** to catch more potential churners
                       - Example - if the threshold/score curve and classfication report show recall > 0.8 at threshold 0.4, use that for churn alert !
                    - Step3️⃣ → Targeted Retention Actions
                       - if high churn risk + **Contract_Month-to-month** → Offer annual contract discount to upgrade
                       - if **InternerService_Fiber optic with OnlineSecurity_No & TechSupport_No** → Offer free trial for these services and discount for InternetService package
                       - if **Eletronic check payment** → Offer incentives to switch to auto-pay, add multiple payment reminder
                    """)





with col2:
    st.subheader("Customer Churn Prediction")
    st.markdown("##### 🛂 Enter single customer info:")
    
  
   # Full features expected by the encoder/model
    expected_features = ['tenure', 'MonthlyCharges', 'TotalCharges', 'gender','Partner' ,'Dependents', 'PhoneService', 'MultipleLines',
 'InternetService', 'OnlineSecurity' ,'OnlineBackup' ,'DeviceProtection',
 'TechSupport', 'StreamingTV' ,'StreamingMovies' ,'Contract',
 'PaperlessBilling', 'PaymentMethod']
    
    
     # Only ask user for these important ones
    user_inputs = {
    'gender':st.selectbox("Gender",["Male","Female"]),
    'Contract': st.selectbox("Contract", ["Month-to-month", "One year", "Two year"]),
    'tenure': st.slider("Tenure (months)", 0, 72, 12),
    'MonthlyCharges': st.number_input("Monthly Charges", min_value=0.0, value=120.0),
    'TotalCharges':st.number_input("Total Charges",min_value=0.0,max_value=10000.0),
    'InternetService': st.selectbox("Internet Service",["Fiber optic","DSL","No"]),
    'OnlineSecurity':st.selectbox("Online Security", ["Yes", "No", "No internet service"]),
    'TechSupport': st.selectbox("TechSupport",["Yes","No","No internet service"]),
    'PaymentMethod': st.selectbox("Payment Method", [
        "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
    ])
}
    
    default_values = {
    'Dependents':'No',
    'Partner': 'No',
    'PhoneService': 'Yes',
    'MultipleLines': 'No',
    'OnlineBackup': 'No',
    'DeviceProtection': 'No',
    'StreamingTV': 'No',
    'StreamingMovies': 'No',
    'PaperlessBilling': 'Yes'
}
    
    full_input_dict = {}
    for feature in expected_features:
        if feature in user_inputs:
            full_input_dict[feature] = user_inputs[feature]
        else:
            full_input_dict[feature] = default_values.get(feature, 'No')  # fallback

    input_df = pd.DataFrame([full_input_dict])
    
    # Load encoder ,model, and expected columns
    encoder = joblib.load("model/saved_model/encoder.joblib")
    scaler = joblib.load("model/saved_model/scaler.joblib")
    expected_cols = joblib.load("model/saved_model/encoded_cols.joblib")
    
    X_cat = input_df.select_dtypes(include='object')
    X_num = input_df.select_dtypes(exclude='object')
    
    X_encoded = encoder.transform(X_cat)
    X_scaled = scaler.transform(X_num)
    
    X_all = np.hstack((X_scaled, X_encoded))
    all_feature_names = np.concatenate([X_num.columns, encoder.get_feature_names_out()])
    X_df = pd.DataFrame(X_all, columns=all_feature_names)
    
    for col in expected_cols:
        if col not in X_df.columns: 
            X_df[col] = 0
    #Column names match,Column order matches exactly what the model expects
    X_df = X_df[model.feature_names_in_]    
    
    st.dataframe(X_df)
    
    contract_type = 1 if user_inputs['Contract'] == 'Month-to-month' else 0
    internet_service = 1 if user_inputs['InternetService'] == 'Fiber optic' else 0

    
    if contract_type == 1 or internet_service == 1:
        recommended_model = "XGBoost"
    elif user_inputs['tenure'] < 12 or user_inputs['TotalCharges'] < 500:
        recommended_model = "Logistic"
    else:
        recommended_model = "Random Forest"
    st.info(f"Based on this customer's profile, we recommend using the **{recommended_model}** model.")

    
    model_choice = st.selectbox(
    "Choose Model for Prediction:",
    ("Logistic", "Random Forest", "XGBoost", "MLP")
)
    model_paths = {
    "Logistic": "./model/saved_model/logistic_model.joblib",
    "Random Forest":"./model/saved_model/rf_model.joblib",
    "XGBoost": "./model/saved_model/xgb_model.joblib",
    "MLP": "./model/saved_model/mlp_model.joblib",
}
    
    model = joblib.load(model_paths[model_choice])
       
    if st.button("Predict Churn"):
        prediction = model.predict(X_df)[0]
        prob = model.predict_proba(X_df)[0][1]
        prob_single = model.predict_proba(X_df)[:,1][0]
        
        thresholds = 0.4
        
        # Predict with custom threshold
        pred_single = (prob_single >= thresholds).astype(int)
        st.write(f"**Selected Model:** {model_choice}")
        if pred_single.any() == 1:
            st.error(f"⚠️ This customer is likely to churn , Churn probability :{prob_single:.2f}")
        else:
            st.success(f"✅  This customer is likely to stay, Churn probability:{prob_single:.2f}")
        
    st.markdown("""
                
                """)

   
    
    
    
    




        
            
            


