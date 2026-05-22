import streamlit as st
import numpy as np
import os
from pydub import AudioSegment
import tensorflow as tf
from collections import Counter
import librosa

# =========================
# FFmpeg setup
# =========================
AudioSegment.converter = "ffmpeg"

# =========================
# SETTINGS
# =========================
SR = 16000
MAX_LEN = 130
EPS = 1e-8

# 🔥 CHANGED HERE (IMPORTANT)
MODEL_PATH = "gender_model.keras"

# =========================
# FEATURE EXTRACTION
# =========================
def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=SR)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)

    features = np.vstack([mfcc, delta, delta2])

    # pad / trim
    if features.shape[1] < MAX_LEN:
        pad = MAX_LEN - features.shape[1]
        features = np.pad(features, ((0, 0), (0, pad)))
    else:
        features = features[:, :MAX_LEN]

    # normalization
    features = (features - np.mean(features, axis=1, keepdims=True)) / (
        np.std(features, axis=1, keepdims=True) + EPS
    )

    return features.astype(np.float32)

# =========================
# LOAD MODEL (SAFE)
# =========================
MODEL_PATH = "gender_model.h5"

@st.cache_resource
def load_model_safe():
    return tf.keras.models.load_model(MODEL_PATH, compile=False)
model = load_model_safe()

# =========================
# UI
# =========================
st.set_page_config(page_title="Voice Gender Detection", layout="centered")

st.title("🎤 Voice Gender Classification")
st.write("Upload a WAV file for prediction")

uploaded_file = st.file_uploader("Upload Audio", type=["wav"])

# =========================
# PREDICTION
# =========================
def predict(file_path, threshold=0.5):

    audio = AudioSegment.from_wav(file_path)
    predictions = []

    for i in range(0, len(audio), 3000):

        chunk = audio[i:i+3000]

        # skip silence
        if chunk.dBFS == float("-inf") or chunk.dBFS < -55:
            continue

        temp_file = "temp.wav"
        chunk.export(temp_file, format="wav")

        feat = extract_features(temp_file)
        feat = feat[np.newaxis, ..., np.newaxis]

        prob = model.predict(feat, verbose=0)[0][0]

        label = "MALE" if prob > threshold else "FEMALE"
        predictions.append(label)

    if len(predictions) == 0:
        return None, 0

    final_label = Counter(predictions).most_common(1)[0][0]
    confidence = Counter(predictions)[final_label] / len(predictions)

    return final_label, confidence

# =========================
# APP LOGIC
# =========================
if uploaded_file is not None:

    file_path = "temp_uploaded.wav"

    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())

    st.audio(uploaded_file)

    if st.button("Predict Gender"):

        with st.spinner("Analyzing audio..."):
            label, conf = predict(file_path)

        if label is None:
            st.warning("No speech detected in audio")
        else:
            st.success(f"🎯 Prediction: {label}")
            st.info(f"📊 Confidence: {conf:.3f}")
