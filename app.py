import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import streamlit as st
import numpy as np
import tensorflow as tf
from pydub import AudioSegment
import librosa

# =========================
# CONFIG
# =========================
SR = 16000
MAX_LEN = 130
EPS = 1e-8
MODEL_PATH = "cnn_gender_model.keras"

AudioSegment.converter = "ffmpeg"

# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH, compile=False)

model = load_model()

# =========================
# FEATURE EXTRACTION (SAME AS COLAB)
# =========================
def extract_features(file_path):

    y, sr = librosa.load(file_path, sr=SR)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)

    features = np.vstack([mfcc, delta, delta2])  # (39, T)

    # pad / trim
    if features.shape[1] < MAX_LEN:
        pad = MAX_LEN - features.shape[1]
        features = np.pad(features, ((0,0),(0,pad)))
    else:
        features = features[:, :MAX_LEN]

    # normalization (same as training)
    features = features / (np.max(np.abs(features), axis=1, keepdims=True) + EPS)

    return features.astype(np.float32)

# =========================
# CLEAN PREDICTION (FIXED LOGIC)
# =========================
def predict(file_path):

    audio = AudioSegment.from_wav(file_path)

    probs = []

    # IMPORTANT: keep SAME chunk size as training
    for i in range(0, len(audio), 3000):

        chunk = audio[i:i+3000]

        # skip silence
        if chunk.dBFS == float("-inf") or chunk.dBFS < -55:
            continue

        chunk.export("temp.wav", format="wav")

        feat = extract_features("temp.wav")

        feat = feat[np.newaxis, ..., np.newaxis]

        prob = float(model.predict(feat, verbose=0)[0][0])

        # DEBUG VIEW (optional)
        st.write("Chunk prob:", prob)

        probs.append(prob)

    if len(probs) == 0:
        return None, 0

    # =========================
    # STABLE AGGREGATION
    # =========================
    probs = np.array(probs)

    avg_prob = np.median(probs)   # IMPORTANT: median is more stable than mean

    label = "MALE" if avg_prob > 0.5 else "FEMALE"

    # confidence = distance from 0.5 (fixes fake 0.5 confidence)
    confidence = abs(avg_prob - 0.5) * 2
    confidence = min(max(confidence, 0), 1)

    return label, confidence

# =========================
# STREAMLIT UI
# =========================
st.title("🎤 Voice Gender Classification (CNN - FIXED)")

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
