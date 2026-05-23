import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import streamlit as st
import numpy as np
import tensorflow as tf
from pydub import AudioSegment
import librosa
import tempfile

AudioSegment.converter = "ffmpeg"

# =========================
# CONSTANTS (MUST MATCH COLAB)
# =========================
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
# FEATURE EXTRACTION (IDENTICAL TO COLAB)
# =========================
def extract_features(y, sr):

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)

    features = np.vstack([mfcc, delta, delta2])  # (39, T)

    # pad / trim
    if features.shape[1] < MAX_LEN:
        features = np.pad(features, ((0,0),(0, MAX_LEN - features.shape[1])))
    else:
        features = features[:, :MAX_LEN]

    # normalize
    features = (features - np.mean(features, axis=1, keepdims=True)) / (
        np.std(features, axis=1, keepdims=True) + EPS
    )

    return features.astype(np.float32)


# =========================
# CLEAN AUDIO SEGMENTATION (IMPORTANT FIX)
# =========================
def get_speech_segments(y, sr):
    intervals = librosa.effects.split(y, top_db=25)
    return [(y[start:end]) for start, end in intervals]


# =========================
# PREDICTION (FIXED VERSION)
# =========================
def predict(file_path):

    y, sr = librosa.load(file_path, sr=SR)

    segments = get_speech_segments(y, sr)

    probs = []

    if len(segments) == 0:
        return None, 0

    for seg in segments:

        if len(seg) < SR // 2:
            continue

        feat = extract_features(seg, sr)
        feat = feat[np.newaxis, ..., np.newaxis]

        prob = float(model.predict(feat, verbose=0)[0][0])

        probs.append(prob)

        # 🔍 DEBUG OUTPUT (YOU ASKED FOR THIS)
        st.write("Chunk prob:", prob)

    if len(probs) == 0:
        return None, 0

    avg_prob = np.mean(probs)

    label = "MALE" if avg_prob > 0.5 else "FEMALE"
    confidence = max(avg_prob, 1 - avg_prob)

    st.write("Average prob:", avg_prob)

    return label, confidence


# =========================
# STREAMLIT UI
# =========================
st.title("🎤 Voice Gender Classification (FIXED CNN)")

uploaded_file = st.file_uploader("Upload WAV file", type=["wav"])

if uploaded_file is not None:

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(uploaded_file.read())
        temp_path = tmp.name

    st.audio(uploaded_file)

    if st.button("Predict Gender"):

        label, conf = predict(temp_path)

        if label is None:
            st.warning("⚠️ No speech detected")
        else:
            st.success(f"Prediction: {label}")
            st.info(f"Confidence: {conf:.3f}")
