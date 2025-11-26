import streamlit as st
import requests
from PIL import Image
import os
import pandas as pd
import matplotlib.pyplot as plt

# Configuration
API_URL = "http://127.0.0.1:8000"
VIS_DIR = "visualizations"

st.set_page_config(page_title="Flower Prediction Dashboard", layout="wide")

st.title("🌸 Flower Prediction Dashboard")

# Sidebar for controls
st.sidebar.header("Controls")

# 1. Model Up-time / Status
st.header("1. System Status")
try:
    response = requests.get(f"{API_URL}/")
    if response.status_code == 200:
        status = response.json()
        st.success(f"API is Online. Message: {status['message']}")
        st.info(f"Model Status: {status['model_status']}")
    else:
        st.error("API is reachable but returned an error.")
except requests.exceptions.ConnectionError:
    st.error("API is Offline. Please start the API server.")

# 2. Retraining Trigger
st.sidebar.subheader("Model Management")
if st.sidebar.button("Trigger Retraining"):
    with st.spinner("Triggering training..."):
        try:
            response = requests.post(f"{API_URL}/train?force=true")
            if response.status_code == 200:
                st.sidebar.success("Training triggered successfully!")
                st.sidebar.json(response.json())
            else:
                st.sidebar.error(f"Failed to trigger training: {response.text}")
        except Exception as e:
            st.sidebar.error(f"Error: {e}")

# 3. Data Visualizations
st.header("2. Data Visualizations")
col1, col2 = st.columns(2)

with col1:
    st.subheader("Training History")
    history_path = os.path.join(VIS_DIR, "training_history.png")
    if os.path.exists(history_path):
        st.image(history_path, caption="Accuracy & Loss")
    else:
        st.warning("Training history plot not found.")

with col2:
    st.subheader("Confusion Matrix")
    cm_path = os.path.join(VIS_DIR, "confusion_matrix.png")
    if os.path.exists(cm_path):
        st.image(cm_path, caption="Confusion Matrix")
    else:
        st.warning("Confusion matrix plot not found.")

# 4. Inference / Prediction
st.header("3. Live Prediction")
uploaded_file = st.file_uploader("Choose a flower image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    col_img, col_pred = st.columns([1, 2])
    
    with col_img:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
    
    with col_pred:
        if st.button("Predict"):
            with st.spinner("Classifying..."):
                try:
                    # Reset file pointer
                    uploaded_file.seek(0)
                    files = {"file": uploaded_file}
                    response = requests.post(f"{API_URL}/predict", files=files)
                    
                    if response.status_code == 200:
                        result = response.json()
                        st.success(f"Prediction: **{result['class']}**")
                        st.metric("Confidence", result['confidence'])
                    else:
                        st.error(f"Prediction failed: {response.text}")
                except Exception as e:
                    st.error(f"Error: {e}")
