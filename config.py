import streamlit as st 

# Default config values
default_config = {
    "n_estimators":500,
    "max_depth":5,
    "min_samples_split": 2
}

def init_config():
    if "model_config" not in st.session_state:
        st.session_state.model_config = default_config.copy()
        
def reset_config():
    st.session_state.model_config = default_config.copy()
    
