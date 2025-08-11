from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
import pandas as pd
import joblib
from joblib import dump 

df = pd.read_csv("./data/preprocessed_train.csv")

X = df
y = df['Churn']

X_train,X_test,y_train, y_test = train_test_split(X,y,test_size=0.2,stratify=y,random_state=42)

smote = SMOTE(random_state=42)
X_train_smote,y_train_smote = smote.fit_resample(X_train,y_train)



