import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from joblib import dump

df = pd.read_csv("./data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

def preprocess_data(df, save_test_csv=False, save_path="./data/test_data.csv"):
    df['Churn'] = df['Churn'].astype(str).str.strip().map({'Yes': 1, 'No': 0})
    missing_churn = df['Churn'].isna().sum()
    if missing_churn > 0:
      print(f" Warning: Found {missing_churn} rows with unknown/missing Churn values. Dropping them.")
      df = df[df['Churn'].notna()]
        
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.drop(['customerID'],axis=1)
    
    imputer = SimpleImputer(strategy='mean')
    imputer.fit(df[['TotalCharges']])
    df['TotalCharges'] = imputer.transform(df[['TotalCharges']])
    
     # train , test split
    train_df, test_df = train_test_split(df,test_size=0.2,random_state=42)

    # Identify Input columns and target col
    target_col = 'Churn'
    input_cols = list(df.drop(['Churn','SeniorCitizen'],axis=1))
    
    train_inputs = train_df[input_cols]
    train_target = train_df[target_col]
    test_inputs = test_df[input_cols]
    test_target = test_df[target_col]
        
    numerical_cols = train_inputs.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = train_inputs.select_dtypes('object').columns.tolist()
      
      # encode categorical date 
    encoder = OneHotEncoder(sparse_output=False,handle_unknown='ignore')
    encoder.fit(df[categorical_cols])
    
    dump(encoder, 'model/saved_model/encoder.joblib')
    
    encoded_cols = list(encoder.get_feature_names_out(categorical_cols))  
    all_feature_names = list(encoded_cols)+numerical_cols
    
    dump(all_feature_names,'model/saved_model/encoded_cols.joblib')
    
    train_inputs.loc[:,encoded_cols] = encoder.transform(train_inputs[categorical_cols])
    test_inputs.loc[:,encoded_cols] = encoder.transform(test_inputs[categorical_cols])
    
    train_inputs = train_inputs[numerical_cols + encoded_cols]
    test_inputs = test_inputs[numerical_cols+encoded_cols]
    
    if save_test_csv:
      test_df = X_test.copy()
      test_df['Churn'] = y_test.values
      test_df.to_csv(save_path,index=False)
      print(f"Processed test data saved to:{save_path}")
      
    
    return numerical_cols,encoded_cols,categorical_cols,train_inputs,train_target,test_inputs,test_target
     

num_cols, enc_cols,cat_cols,X_train,y_train, X_test,y_test = preprocess_data(df)

# unpack all return values
def scaled(save_test_csv=False,save_path="./data/scaled_test_data.csv"):
    # Scale numerical features
    num_imputer = SimpleImputer(strategy='mean')   
    X_train_num = num_imputer.fit_transform(X_train[num_cols])
    X_test_num = num_imputer.transform(X_test[num_cols])
    
    scaler = MinMaxScaler()
    X_train_scaled_num = scaler.fit(X_train_num)
    dump(scaler, 'model/saved_model/scaler.joblib')
    
    X_train_scaled_num = scaler.transform(X_train_num)
    X_test_scaled_num = scaler.transform(X_test_num)
    

    df_train_scaled_num = pd.DataFrame(X_train_scaled_num, columns=num_cols, index=X_train.index)
    df_test_scaled_num = pd.DataFrame(X_test_scaled_num, columns=num_cols, index=X_test.index)
    
    # Keep encoded categorical features
    X_train_encoded = X_train[enc_cols]
    X_test_encoded = X_test[enc_cols]
    
     # Combine scaled numerical and encoded categorical features
    X_scaled_train = pd.concat([df_train_scaled_num, X_train_encoded], axis=1)
    X_scaled_test = pd.concat([df_test_scaled_num, X_test_encoded], axis=1)
    
    if save_test_csv:
        # Combine features and target into one test dataframe
        scaled_test_df = X_scaled_test.copy()
        scaled_test_df['Churn'] = y_test.values
        scaled_test_df.to_csv(save_path, index=False)
        print(f"✅ Processed test data saved to: {save_path}")
    
    return X_scaled_train, X_scaled_test, y_train, y_test

X_scaled_train, X_scaled_test, y_train, y_test = scaled()

dump(X_train.columns.tolist(), "model/saved_model/columns.joblib")


# save them to csv file 
train_df = pd.DataFrame(X_train,columns=num_cols+enc_cols)
train_df["Churn"] = y_train.values

train_scaled_df = pd.DataFrame(X_scaled_train,columns=num_cols+enc_cols)
train_scaled_df["Churn"] = y_train.values

train_df.to_csv("./data/preprocessed_train.csv",index=False)
train_scaled_df.to_csv("./data/preprocessed_scaled_train.csv",index=False)