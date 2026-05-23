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

st.title("🎤 Voice Gender Classification (FINAL STABLE VERSION)")


# =========================
# FEATURE EXTRACTION (LOCKED)
# =========================
def extract_features(file_path):

    y, sr = librosa.load(file_path, sr=SR)

    # normalize audio FIRST (important fix)
    y = y / (np.max(np.abs(y)) + 1e-8)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)

    features = np.vstack([mfcc, delta, delta2])

    # FIX LENGTH
    features = librosa.util.fix_length(features, size=MAX_LEN, axis=1)

    # GLOBAL NORMALIZATION
    features = (features - np.mean(features)) / (np.std(features) + EPS)

    return features.astype(np.float32)


# =========================
# STABLE PREDICTION ENGINE
# =========================
def predict(file_path):

    audio = AudioSegment.from_wav(file_path)

    probs = []

    for i in range(0, len(audio), 3000):

        chunk = audio[i:i+3000]

        # strict silence filter
        if chunk.dBFS < -45:
            continue

        chunk.export("temp.wav", format="wav")

        feat = extract_features("temp.wav")
        feat = feat[np.newaxis, ..., np.newaxis]

        prob = float(model.predict(feat, verbose=0)[0][0])

        st.write("Chunk prob:", prob)

        probs.append(prob)

    if len(probs) == 0:
        return None, 0

    probs = np.array(probs)

    # =========================
    # FINAL CALIBRATION FIX
    # =========================

    # remove extreme noise
    probs = probs[(probs > 0.15) & (probs < 0.85)]

    if len(probs) == 0:
        probs = np.array(probs)

    avg_prob = np.median(probs)  # IMPORTANT: median > mean

    # SOFT calibration shift (fix bias)
    calibrated = (avg_prob - 0.45) / 0.1
    calibrated = 1 / (1 + np.exp(-calibrated))  # sigmoid rescale

    label = "MALE" if calibrated > 0.5 else "FEMALE"

    confidence = max(calibrated, 1 - calibrated)

    return label, confidence


# =========================
# UI
# =========================
uploaded_file = st.file_uploader("Upload WAV file", type=["wav"])

if uploaded_file is not None:

    with open("temp_uploaded.wav", "wb") as f:
        f.write(uploaded_file.read())

    st.audio(uploaded_file)

    if st.button("Predict Gender"):

        label, conf = predict("temp_uploaded.wav")

        if label is None:
            st.warning("No speech detected")
        else:
            st.success(f"Prediction: {label}")
            st.info(f"Confidence: {conf:.3f}")
