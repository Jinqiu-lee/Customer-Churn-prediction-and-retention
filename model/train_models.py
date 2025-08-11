import argparse
import streamlit as st
import os
import pandas as pd
from joblib import dump
from model.models import logistic_model, rf_model, xgb_model,mlp_model
from model.preprocessing import preprocess_data,scaled
from model.model_utils import evaluate_model
from config import reset_config,init_config
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

print("📂 Current working directory:", os.getcwd())

    

model_dict = {
    "logistic": logistic_model,
    "rf": rf_model,
    "xgb": xgb_model,
    "mlp":mlp_model,
}

def train(model_name):
    config = st.session_state.model_config
    if model_name not in model_dict:
         raise ValueError(f"Invalid model name: {model_name}")
     
     # load data 
    df1 = pd.read_csv("./data/preprocessed_train.csv")
    df2 = pd.read_csv("./data/preprocessed_scaled_train.csv")
    
    # num_cols, enc_cols,cat_cols,X_train,y_train, X_test,y_test = preprocess_data(df,save_test_csv=True)
    # X_scaled_train,X_scaled_test,y_train,y_test = scaled(save_test_csv=True)
    
    X = df1
    y = df1["Churn"]
    scaled_X = df2
    scaled_y= df2["Churn"]
    
    
    X_train,X_test,y_train, y_test = train_test_split(X,y,test_size=0.2,stratify=y,random_state=42)
    X_train2,X_test2,y_train2, y_test2 = train_test_split(scaled_X,scaled_y,test_size=0.2,stratify=y,random_state=42)
     
    smote = SMOTE(random_state=42)
    X_train_smote,y_train_smote = smote.fit_resample(X_train,y_train)
    X_scaled_smote,y_scaled_smote = smote.fit_resample(X_train2,y_train2)


    # train model
    model = model_dict[model_name]()
    print(f"Training{model_name}model")
    
    if model_name == "logistic" or model_name == "mlp":
        model.fit(X_scaled_smote,y_train2)
    elif model_name == "rf" or model_name =="xgb":
        model.fit(X_train_smote,y_train)
        

    # Evaluate and save 
    if model_name == "logistic" or model_name =="mlp":
        evaluate_model(model,X_test2,y_test2)
    elif model_name == "rf" or model_name =="xgb":
        evaluate_model(model,X_test,y_test)
        
        
    os.makedirs("saved_model", exist_ok=True)
    
    # dump(X_train.columns.tolist(),"model/columns.joblib")

    dump(model,f"model/saved_model/{model_name}_model.joblib")
    print(f"Model saved to: model/saved_model/{model_name}_model.joblib")
    
    
# making the script executable from the command line    

if __name__ == "__main__":   # only run the followig code if this line is excuted directly 
    parser = argparse.ArgumentParser()  # using the argparse module, python built-in way to handle command-line arguments
    parser.add_argument(  # This adds an argument the script expects when run from the terminal.
        "--model", # command-line flag,like --model logistic
        type=str, # expect a string value ("rf" or "logistic")
        choices=model_dict.keys(), # limite to only the models I defined (e.g. rf, mlp)
        required=True   # making it mandatory - if you don't pass it, you'll get an error
        )
    args = parser.parse_args()  
    # This parses the command-line input into a variable named args
    # if run : python train_models.py --model logistic in bash , it shows args.model =="logistic" in python

    train(args.model)   # calls the train() function using the model specified from the command line 
    