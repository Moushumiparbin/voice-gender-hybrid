import numpy as np
import librosa
from pydub import AudioSegment
import uuid
from collections import Counter

SR = 16000
N_MFCC = 13
MAX_LEN = 130

def extract_features(file_path):

    y, sr = librosa.load(file_path, sr=SR)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)

    features = np.vstack([mfcc, delta, delta2])

    if features.shape[1] < MAX_LEN:
        features = np.pad(
            features,
            ((0, 0), (0, MAX_LEN - features.shape[1])),
            mode='constant'
        )
    else:
        features = features[:, :MAX_LEN]

    return features


def chunk_audio(file_path):
    audio = AudioSegment.from_wav(file_path)
    chunks = []

    for i in range(0, len(audio), 3000):
        chunk = audio[i:i+3000]

        if chunk.dBFS == float('-inf') or chunk.dBFS < -45:
            continue

        temp_path = f"/tmp/{uuid.uuid4().hex}.wav"
        chunk.export(temp_path, format="wav")
        chunks.append(temp_path)

    return chunks


def predict_audio(file_path, model):
    chunks = chunk_audio(file_path)
    results = []

    for chunk_path in chunks:
        features = extract_features(chunk_path)

        # CNN-LSTM input format
        features = np.transpose(features, (1, 0))
        features = features[np.newaxis, ...]

        pred = model.predict(features, verbose=0)

        label = "Female" if pred[0][0] > 0.5 else "Male"
        results.append(label)

    if len(results) == 0:
        return "No valid audio detected", 0

    counts = Counter(results)
    final = counts.most_common(1)[0][0]
    confidence = counts[final] / len(results)

    return final, confidence
