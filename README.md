# 🎤 Voice Gender Classification using CNN

This project classifies **male vs female voice** using a CNN model trained on MFCC features.

---

## 🚀 Features
- CNN-based deep learning model
- MFCC + Delta + Delta-Delta features
- Streamlit web app
- Real-time audio prediction

---

## 📊 Model Performance
- CNN Accuracy: **99.06%**
- LSTM Accuracy: **96.5%**

---

## 🧠 How it works
1. Audio is split into chunks
2. MFCC features extracted
3. CNN predicts each chunk
4. Majority voting gives final result

---

## ▶️ Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
