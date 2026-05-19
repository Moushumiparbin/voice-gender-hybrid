import streamlit as st
import numpy as np
import tensorflow as tf
import tempfile
from utils import predict_audio

# ================= PAGE CONFIG (MUST BE FIRST) =================
st.set_page_config(page_title="Speech Gender Classification", layout="centered")

st.title("🎤 Assamese Speech Gender Classification")
st.write("Upload a WAV file and the model will predict Gender (Male/Female)")

# ================= LOAD MODEL (SAFE FIX) =================
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        "hybrid_model.keras",
        compile=False
    )
model = load_model()

# ================= FILE UPLOAD =================
uploaded_file = st.file_uploader("Upload Audio File", type=["wav"])

if uploaded_file is not None:

    # save temp file safely
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(uploaded_file.read())
        file_path = tmp.name

    st.audio(uploaded_file)

    st.info("Processing audio... please wait")

    try:
        label, confidence = predict_audio(file_path, model)

        st.success(f"🎯 Prediction: {label}")
        st.write(f"📊 Confidence: {confidence * 100:.2f}%")

    except Exception as e:
        st.error("Something went wrong while processing the audio.")
        st.exception(e)
