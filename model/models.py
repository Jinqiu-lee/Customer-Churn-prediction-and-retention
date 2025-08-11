from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from config import reset_config,init_config
import streamlit as st


def logistic_model():
    return LogisticRegression(solver='liblinear', random_state=42)

def rf_model():
    return RandomForestClassifier(n_jobs=-1, 
                               random_state=42, 
                               n_estimators=st.session_state.model_config["n_estimators"],
                               max_features=10,
                               max_depth=st.session_state.model_config["max_depth"],
                               min_samples_split=st.session_state.model_config["min_samples_split"],
                               min_impurity_decrease=1e-4)

def xgb_model():
    return  XGBClassifier(random_state=42,n_jobs=-1,n_estimators=st.session_state.model_config["n_estimators"],max_depth=st.session_state.model_config["max_depth"])

def mlp_model():
    return MLPClassifier(
    hidden_layer_sizes=(64, 32),  # 2 hidden layers
    activation='relu',
    solver='adam',
    alpha=0.001,  # L2 regularization
    batch_size=16,
    learning_rate_init=0.001,
    max_iter=100,
    early_stopping=True,
)