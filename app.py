import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import streamlit as st
import numpy as np
import tensorflow as tf
from pydub import AudioSegment
import librosa
from collections import Counter

SR = 16000
MAX_LEN = 130
EPS = 1e-8

MODEL_PATH = "cnn_gender_model.keras"
model = tf.keras.models.load_model(MODEL_PATH, compile=False)

def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=SR, mono=True)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)

    features = np.vstack([mfcc, delta, delta2])

    if features.shape[1] < MAX_LEN:
        pad = MAX_LEN - features.shape[1]
        features = np.pad(features, ((0,0),(0,pad)))
    else:
        features = features[:, :MAX_LEN]

    features = (features - np.mean(features, axis=1, keepdims=True)) / (
        np.std(features, axis=1, keepdims=True) + EPS
    )

    return features

def predict(file_path):

    audio = AudioSegment.from_wav(file_path)

    probs = []

    for i in range(0, len(audio), 3000):

        chunk = audio[i:i+3000]

        if chunk.dBFS < -55 or chunk.dBFS == float("-inf"):
            continue

        chunk.export("temp.wav", format="wav")

        feat = extract_features("temp.wav")
        feat = feat[np.newaxis, ..., np.newaxis]

        prob = model.predict(feat, verbose=0)[0][0]

        st.write("Chunk prob:", prob)   # DEBUG LINE

        probs.append(prob)

    if len(probs) == 0:
        return None, 0

    avg = np.mean(probs)

    st.write("Average prob:", avg)

    label = "MALE" if avg > 0.5 else "FEMALE"
    confidence = max(avg, 1 - avg)

    return label, confidence


st.title("🎤 Gender Classification")

file = st.file_uploader("Upload WAV", type=["wav"])

if file:
    with open("temp.wav", "wb") as f:
        f.write(file.read())

    st.audio(file)

    if st.button("Predict"):
        label, conf = predict("temp.wav")
        st.success(label)
        st.info(conf)
