import streamlit as st
import numpy as np
import tensorflow as tf
from utils import predict_audio

# ================= LOAD MODEL =================
model = tf.keras.models.load_model("hybrid_model.h5")

# ================= UI =================
st.set_page_config(page_title="Speech Gender Classification", layout="centered")

st.title("🎤 Assamese Speech Gender Classification")
st.write("Upload a WAV file and the model will predict Gender (Male/Female)")

uploaded_file = st.file_uploader("Upload Audio File", type=["wav"])

if uploaded_file is not None:

    # save uploaded file temporarily
    file_path = f"/tmp/{uploaded_file.name}"
    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())

    st.audio(uploaded_file)

    st.info("Processing audio... please wait")

    label, confidence = predict_audio(file_path, model)

    st.success(f"🎯 Prediction: {label}")
    st.write(f"📊 Confidence: {confidence*100:.2f}%")
