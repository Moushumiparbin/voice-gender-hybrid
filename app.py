import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import streamlit as st
import numpy as np
import tensorflow as tf
from pydub import AudioSegment
import librosa

AudioSegment.converter = "ffmpeg"

SR = 16000
MAX_LEN = 130
EPS = 1e-8
MODEL_PATH = "cnn_gender_model.keras"


# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH, compile=False)

model = load_model()

st.title("🎤 Voice Gender Classification")


# =========================
# FEATURE EXTRACTION (IDENTICAL TO COLAB)
# =========================
def extract_features(file_path):

    y, sr = librosa.load(file_path, sr=SR)

    # normalize audio
    y = y / (np.max(np.abs(y)) + 1e-8)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)

    features = np.vstack([mfcc, delta, delta2])  # (39, T)

    # FIX LENGTH EXACTLY LIKE TRAINING
    if features.shape[1] < MAX_LEN:
        pad = MAX_LEN - features.shape[1]
        features = np.pad(features, ((0,0),(0,pad)))
    else:
        features = features[:, :MAX_LEN]

    # normalization SAME AS TRAINING
    features = (features - np.mean(features, axis=1, keepdims=True)) / (
        np.std(features, axis=1, keepdims=True) + EPS
    )

    return features.astype(np.float32)


# =========================
# CLEAN PREDICTION
# =========================
def predict(file_path):

    audio = AudioSegment.from_wav(file_path)

    probs = []

    for i in range(0, len(audio), 3000):

        chunk = audio[i:i+3000]

        # skip silence only (safe)
        if chunk.dBFS < -55:
            continue

        chunk.export("temp.wav", format="wav")

        feat = extract_features("temp.wav")
        feat = feat[np.newaxis, ..., np.newaxis]

        prob = float(model.predict(feat, verbose=0)[0][0])
        probs.append(prob)

    if len(probs) == 0:
        return "UNDETECTED"

    # =========================
    # FINAL CORRECT AGGREGATION
    # =========================
    probs = np.array(probs)

    final_prob = np.mean(probs)

    # FINAL DECISION ONLY
    if final_prob > 0.5:
        return "MALE"
    else:
        return "FEMALE"


# =========================
# UI
# =========================
uploaded_file = st.file_uploader("Upload WAV file", type=["wav"])

if uploaded_file is not None:

    with open("temp_uploaded.wav", "wb") as f:
        f.write(uploaded_file.read())

    st.audio(uploaded_file)

    if st.button("Predict Gender"):

        result = predict("temp_uploaded.wav")

        if result == "UNDETECTED":
            st.warning("No speech detected")
        else:
            st.success(f"Prediction: {result}")
