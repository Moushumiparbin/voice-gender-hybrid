import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import streamlit as st
import numpy as np
import tensorflow as tf
import librosa
from pydub import AudioSegment

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
# FEATURE EXTRACTION (IMPROVED FOR SEPARATION)
# =========================
def extract_features(file_path):

    y, sr = librosa.load(file_path, sr=SR)

    # 🔥 IMPROVEMENT 1: normalize + pre-emphasis
    y = librosa.util.normalize(y)
    y = librosa.effects.preemphasis(y)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)

    features = np.vstack([mfcc, delta, delta2])

    # pad / truncate
    if features.shape[1] < MAX_LEN:
        pad = MAX_LEN - features.shape[1]
        features = np.pad(features, ((0,0),(0,pad)))
    else:
        features = features[:, :MAX_LEN]

    # normalize per feature
    features = (features - np.mean(features, axis=1, keepdims=True)) / (
        np.std(features, axis=1, keepdims=True) + EPS
    )

    return features.astype(np.float32)

# =========================
# IMPROVED PREDICTION (FIXED SEPARATION)
# =========================
def predict_gender(file_path):

    audio = AudioSegment.from_wav(file_path)

    probs = []
    weights = []

    for i in range(0, len(audio), 3000):

        chunk = audio[i:i+3000]

        # 🔥 IMPROVEMENT 2: stricter silence removal
        if chunk.dBFS == float("-inf") or chunk.dBFS < -35:
            continue

        chunk.export("temp.wav", format="wav")

        feat = extract_features("temp.wav")
        feat = feat[np.newaxis, ..., np.newaxis]

        prob = float(model.predict(feat, verbose=0)[0][0])

        # 🔥 IMPROVEMENT 3: sharpen separation
        prob = (prob - 0.5) * 2
        prob = 1 / (1 + np.exp(-prob))

        probs.append(prob)

        # confidence weight (distance from 0.5)
        weights.append(abs(prob - 0.5))

        st.write("Chunk prob:", prob)

    if len(probs) == 0:
        return None, 0, 0

    # =========================
    # 🔥 IMPROVEMENT 4: weighted averaging
    # =========================
    probs = np.array(probs)
    weights = np.array(weights) + 1e-8

    final_prob = np.sum(probs * weights) / np.sum(weights)

    # =========================
    # FINAL DECISION (CLEAR SEPARATION)
    # =========================
    if final_prob > 0.6:
        label = "MALE"
    elif final_prob < 0.4:
        label = "FEMALE"
    else:
        label = "UNCERTAIN"

    confidence = abs(final_prob - 0.5) * 2

    return label, confidence, final_prob

# =========================
# STREAMLIT UI
# =========================
st.title("🎤 Voice Gender Classification (CNN - Final Version)")
st.write("Upload a WAV file for prediction")

uploaded_file = st.file_uploader("Upload WAV", type=["wav"])

if uploaded_file is not None:

    with open("temp_input.wav", "wb") as f:
        f.write(uploaded_file.read())

    st.audio(uploaded_file)

    if st.button("Predict Gender"):

        label, conf, final_prob = predict_gender("temp_input.wav")

        if label is None:
            st.warning("No speech detected")
        else:
            st.success(f"Prediction: {label}")

            st.info(f"Final Male Probability: {final_prob:.3f}")
            st.info(f"Confidence Score: {conf:.3f}")

            st.write("Decision Rule:")
            st.write("MALE > 0.6 | FEMALE < 0.4 | ELSE UNCERTAIN")
