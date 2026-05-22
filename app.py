import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import streamlit as st
import numpy as np
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

MODEL_PATH = "gender_model.h5"

# =========================
# FEATURE EXTRACTION
# =========================
def extract_features(file_path):

    y, sr = librosa.load(file_path, sr=SR)

    # MFCC
    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=13
    )

    # Delta
    delta = librosa.feature.delta(mfcc)

    # Delta-Delta
    delta2 = librosa.feature.delta(
        mfcc,
        order=2
    )

    # Combine -> 39 features
    features = np.vstack([mfcc, delta, delta2])

    # Padding / trimming
    if features.shape[1] < MAX_LEN:
        pad = MAX_LEN - features.shape[1]
        features = np.pad(
            features,
            ((0, 0), (0, pad))
        )
    else:
        features = features[:, :MAX_LEN]

    # Normalization
    features = (
        features - np.mean(features, axis=1, keepdims=True)
    ) / (
        np.std(features, axis=1, keepdims=True) + EPS
    )

    return features.astype(np.float32)

# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_model_safe():

    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )

    return model

# 🔥 IMPORTANT
model = load_model_safe()

# =========================
# STREAMLIT UI
# =========================
st.set_page_config(
    page_title="Voice Gender Detection",
    layout="centered"
)

st.title("🎤 Voice Gender Classification")
st.write("Upload a WAV audio file")

uploaded_file = st.file_uploader(
    "Upload Audio",
    type=["wav"]
)

# =========================
# PREDICTION FUNCTION
# =========================
def predict(file_path, threshold=0.5):

    audio = AudioSegment.from_wav(file_path)

    predictions = []

    # Process in 3-second chunks
    for i in range(0, len(audio), 3000):

        chunk = audio[i:i+3000]

        # Skip silence
        if (
            chunk.dBFS == float("-inf")
            or chunk.dBFS < -55
        ):
            continue

        # Temporary chunk file
        temp_file = "temp.wav"

        chunk.export(
            temp_file,
            format="wav"
        )

        # Extract features
        feat = extract_features(temp_file)

        # CNN input shape
        feat = feat[np.newaxis, ..., np.newaxis]

        # Prediction
        prob = model.predict(
            feat,
            verbose=0
        )[0][0]

        # Label
        label = (
            "MALE"
            if prob > threshold
            else "FEMALE"
        )

        predictions.append(label)

    # No speech found
    if len(predictions) == 0:
        return None, 0

    # Majority voting
    final_label = Counter(
        predictions
    ).most_common(1)[0][0]

    confidence = (
        Counter(predictions)[final_label]
        / len(predictions)
    )

    return final_label, confidence

# =========================
# APP LOGIC
# =========================
if uploaded_file is not None:

    file_path = "temp_uploaded.wav"

    # Save uploaded file
    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())

    st.audio(uploaded_file)

    if st.button("Predict Gender"):

        with st.spinner("Analyzing audio..."):

            label, conf = predict(file_path)

        if label is None:

            st.warning(
                "No speech detected in audio"
            )

        else:

            st.success(
                f"🎯 Prediction: {label}"
            )

            st.info(
                f"📊 Confidence: {conf:.3f}"
            )
