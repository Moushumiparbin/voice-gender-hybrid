import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import streamlit as st
import numpy as np
import librosa
from pydub import AudioSegment

import tensorflow as tf
from tensorflow.keras.models import load_model

tf.keras.backend.clear_session()

model = load_model("cnn_gender_model.keras", compile=False)

SR = 16000
CHUNK = 3000

def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=SR)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)

    feat = np.vstack([mfcc, delta, delta2])

    if feat.shape[1] < 130:
        feat = np.pad(feat, ((0,0),(0,130-feat.shape[1])))
    else:
        feat = feat[:, :130]

    feat = (feat - np.mean(feat)) / (np.std(feat) + 1e-8)

    return feat.astype(np.float32)


def predict_audio(file):

    audio = AudioSegment.from_file(file)

    preds = []

    for i in range(0, len(audio), CHUNK):
        chunk = audio[i:i+CHUNK]

        if chunk.dBFS == float("-inf"):
            continue

        chunk.export("temp.wav", format="wav")

        feat = extract_features("temp.wav")
        feat = feat[np.newaxis, ..., np.newaxis]

        prob = model.predict(feat, verbose=0)[0][0]

        preds.append(prob > 0.5)

    if len(preds) == 0:
        return "No speech detected"

    return "MALE" if sum(preds) > len(preds)/2 else "FEMALE"


st.title("🎤 Speech Gender Classification (CNN)")

audio_file = st.file_uploader("Upload WAV file", type=["wav"])

if audio_file is not None:
    st.audio(audio_file)

    if st.button("Predict Gender"):
        result = predict_audio(audio_file)
        st.success(f"Prediction: {result}")
