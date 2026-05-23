import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import streamlit as st
import numpy as np
import tensorflow as tf
from pydub import AudioSegment
from collections import Counter
import librosa

# =========================
# SETTINGS (MUST MATCH COLAB)
# =========================
SR = 16000
MAX_LEN = 130
EPS = 1e-8

MODEL_PATH = "cnn_gender_model.keras"

# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_model_safe():
    return tf.keras.models.load_model(MODEL_PATH, compile=False)

model = load_model_safe()

# =========================
# FEATURE EXTRACTION (EXACT COPY)
# =========================
def extract_features(file_path):

    y, sr = librosa.load(file_path, sr=SR, mono=True)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)

    features = np.vstack([mfcc, delta, delta2])  # (39, T)

    if features.shape[1] < MAX_LEN:
        pad = MAX_LEN - features.shape[1]
        features = np.pad(features, ((0,0),(0,pad)))
    else:
        features = features[:, :MAX_LEN]

    features = (features - np.mean(features, axis=1, keepdims=True)) / (
        np.std(features, axis=1, keepdims=True) + EPS
    )

    return features.astype(np.float32)

# =========================
# PREDICTION (IDENTICAL TO COLAB LOGIC)
# =========================
def predict(file_path):

    audio = AudioSegment.from_wav(file_path)

    predictions = []

    for i in range(0, len(audio), 3000):

        chunk = audio[i:i+3000]

        # SAME SILENCE FILTER AS COLAB
        if chunk.dBFS == float("-inf") or chunk.dBFS < -40:
            continue

        chunk.export("temp.wav", format="wav")

        feat = extract_features("temp.wav")
        feat = feat[np.newaxis, ..., np.newaxis]

        prob = float(model.predict(feat, verbose=0)[0][0])

        label = "MALE" if prob > 0.5 else "FEMALE"
        predictions.append(label)

    if len(predictions) == 0:
        return None, 0

    # MAJORITY VOTING (SAME AS COLAB)
    final_label = Counter(predictions).most_common(1)[0][0]
    confidence = Counter(predictions)[final_label] / len(predictions)

    return final_label, confidence

# =========================
# STREAMLIT UI
# =========================
st.title("🎤 Voice Gender Classification (Matched with Colab)")

uploaded_file = st.file_uploader("Upload WAV file", type=["wav"])

if uploaded_file is not None:

    file_path = "temp_uploaded.wav"

    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())

    st.audio(uploaded_file)

    if st.button("Predict Gender"):

        label, conf = predict(file_path)

        if label is None:
            st.warning("No speech detected")
        else:
            st.success(f"🎯 Prediction: {label}")
            st.info(f"📊 Confidence: {conf:.3f}")
