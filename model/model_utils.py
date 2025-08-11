from sklearn.metrics import classification_report, confusion_matrix,accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import os
import joblib
from joblib import dump
import streamlit as st
from config import init_config,reset_config
"""
def saved_model(model,path):
    os.makedirs(os.path.dirname(path),exist_ok=True)
    joblib.dump(model,path)
"""


def evaluate_model(model,X,y,name=''):
    config = st.session_state.model_config
    y_pred = model.predict(X)
    accuracy = accuracy_score(y,y_pred)
    print("Accuracy:{:.2f}%".format(accuracy*100))
    print("Classification Report:\n", classification_report(y, y_pred))
    
    cf = confusion_matrix(y,y_pred,normalize='true')
    plt.figure(figsize=(6,3))
    sns.heatmap(cf,annot=True)
    plt.xlabel('Prediction')
    plt.ylabel('Target')
    plt.title('{} Confusion Matrix'.format(name))
    plt.show()
    
    return y_pred

    

    
    
    