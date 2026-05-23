import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import streamlit as st
import numpy as np
import tensorflow as tf
from pydub import AudioSegment
import librosa
from collections import Counter

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

# =========================
# FEATURE EXTRACTION (UNCHANGED)
# =========================
def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=SR)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)

    feat = np.vstack([mfcc, delta, delta2])

    if feat.shape[1] < MAX_LEN:
        feat = np.pad(feat, ((0,0),(0, MAX_LEN - feat.shape[1])))
    else:
        feat = feat[:, :MAX_LEN]

    feat = (feat - np.mean(feat, axis=1, keepdims=True)) / (
        np.std(feat, axis=1, keepdims=True) + EPS
    )

    return feat.astype(np.float32)

# =========================
# FIXED PREDICTION LOGIC
# =========================
def predict(file_path):

    audio = AudioSegment.from_wav(file_path)

    probs = []

    for i in range(0, len(audio), 2000):  # smaller window helps stability
        chunk = audio[i:i+2000]

        if chunk.dBFS < -50:
            continue

        chunk.export("temp.wav", format="wav")

        feat = extract_features("temp.wav")
        feat = feat[np.newaxis, ..., np.newaxis]

        prob = float(model.predict(feat, verbose=0)[0][0])
        probs.append(prob)

    if len(probs) == 0:
        return None, 0

    # =========================
    # IMPORTANT FIX 1: REMOVE HARD THRESHOLD BIAS
    # =========================
    avg_prob = np.mean(probs)

    # =========================
    # IMPORTANT FIX 2: SOFT DECISION
    # =========================
    if avg_prob > 0.5:
        label = "MALE"
    else:
        label = "FEMALE"

    # =========================
    # IMPORTANT FIX 3: REAL CONFIDENCE
    # (distance from uncertainty)
    # =========================
    confidence = abs(avg_prob - 0.5) * 2  # scales to [0,1]

    return label, confidence

# =========================
# STREAMLIT UI
# =========================
st.title("🎤 Voice Gender Classification")

file = st.file_uploader("Upload WAV", type=["wav"])

if file is not None:

    with open("temp.wav", "wb") as f:
        f.write(file.read())

    st.audio(file)

    if st.button("Predict"):

        label, conf = predict("temp.wav")

        if label is None:
            st.warning("No speech detected")
        else:
            st.success(f"Prediction: {label}")
            st.info(f"Confidence: {conf:.3f}")
