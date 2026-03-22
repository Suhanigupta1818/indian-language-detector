%%writefile app.py
import streamlit as st
import librosa
import numpy as np
import joblib
import tempfile
import os

st.set_page_config(
    page_title="Indian Language Detector",
    page_icon="🎤",
    layout="centered"
)

@st.cache_resource
def load_model():
    if not os.path.exists("language_model_v3.pkl"):
        st.error("❌ language_model_v3.pkl not found!")
        st.stop()
    return joblib.load("language_model_v3.pkl")

model = load_model()

def extract_features(file_path):
    signal, sr = librosa.load(file_path, sr=16000, duration=3)
    mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=40)
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)
    chroma = librosa.feature.chroma_stft(y=signal, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)
    mel = librosa.feature.melspectrogram(y=signal, sr=sr)
    mel_mean = np.mean(mel, axis=1)[:20]
    zcr = np.mean(librosa.feature.zero_crossing_rate(signal))
    rms = np.mean(librosa.feature.rms(y=signal))
    return np.concatenate([mfcc_mean, mfcc_std, chroma_mean, mel_mean, [zcr, rms]])

col1, col2 = st.columns([1, 5])
with col1:
    st.markdown("## 🎙️")
with col2:
    st.markdown("## Indian Language Detector")
    st.caption("Powered by MFCC + SVM · 86% accuracy")

st.markdown("---")

c1, c2, c3 = st.columns(3)
c1.metric("Languages", "10")
c2.metric("Accuracy", "86%")
c3.metric("Model", "SVM")

st.markdown("---")

uploaded_file = st.file_uploader(
    "📁 Upload audio file (.wav or .mp3)",
    type=["wav", "mp3"]
)

if uploaded_file is not None:
    st.audio(uploaded_file)

    with st.spinner("🔍 Analysing audio..."):
        suffix = ".wav" if uploaded_file.name.endswith(".wav") else ".mp3"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        try:
            features = extract_features(tmp_path).reshape(1, -1)
            prediction = model.predict(features)[0]

            st.markdown("---")
            st.success(f"✅ Detected language: **{prediction}**")

            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(features)[0]
                classes = model.classes_
                top5_idx = np.argsort(proba)[::-1][:5]
                st.markdown("#### 📊 Top 5 predictions")
                for idx in top5_idx:
                    lang = classes[idx]
                    conf = round(proba[idx] * 100, 1)
                    st.progress(int(conf), text=f"{lang} — {conf}%")

        except Exception as e:
            st.error(f"❌ Error: {e}")
        finally:
            os.unlink(tmp_path)

st.markdown("---")
st.caption("Indian Languages Audio Dataset · MFCC + Chroma + Mel · SVM classifier")
