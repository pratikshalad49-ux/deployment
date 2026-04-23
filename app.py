import streamlit as st
import pandas as pd
import pickle
import streamlit_lottie as st_lottie
import requests

# --- CONFIGURATION ---
st.set_page_config(page_title="Model Predictor", page_icon="🤖", layout="centered")

def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# --- ASSETS ---
lottie_robot = load_lottieurl("https://assets5.lottiefiles.com/packages/lf20_prev_ui_v2.json")

# --- LOAD MODEL ---
@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

# --- CUSTOM CSS ---
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stButton>button {
        width: 100%;
        border-radius: 20px;
        height: 3em;
        background-color: #007bff;
        color: white;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #0056b3;
        transform: scale(1.02);
    }
    </style>
    """, unsafe_allow_html=True)

# --- UI LAYOUT ---
st.title("🚀 Machine Learning Deployment")
st_lottie.st_lottie(lottie_robot, height=200, key="coding")

st.write("---")
st.subheader("Enter Input Features")

# Replace these with your actual model features
col1, col2 = st.columns(2)

with col1:
    feature_1 = st.number_input("Feature 1", value=0.0)
    feature_2 = st.number_input("Feature 2", value=0.0)

with col2:
    feature_3 = st.number_input("Feature 3", value=0.0)
    feature_4 = st.selectbox("Feature 4 (Category)", options=[0, 1, 2])

# Prediction Logic
if st.button("Predict Result"):
    # Prepare the input array (ensure it matches your model's expected shape)
    input_data = [[feature_1, feature_2, feature_3, feature_4]]
    
    with st.spinner('Calculating...'):
        prediction = model.predict(input_data)
        
    st.balloons()
    st.success(f"### Result: {prediction[0]}")
