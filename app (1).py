%%writefile app.py
import streamlit as st
import librosa
import numpy as np
import joblib
import tempfile
import os
import soundfile as sf

st.set_page_config(
    page_title="Indian Language Detector",
    page_icon="🎤",
    layout="centered"
)

@st.cache_resource
def load_model():
    if not os.path.exists("language_model_v2.pkl"):
        st.error("❌ language_model_v2.pkl not found!")
        st.stop()
    return joblib.load("language_model_v2.pkl")

model = load_model()

def extract_features_advanced(file_path):
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

def predict_language(file_path):
    features = extract_features_advanced(file_path).reshape(1, -1)
    prediction = model.predict(features)[0]
    proba = model.predict_proba(features)[0]
    classes = model.classes_
    top5_idx = np.argsort(proba)[::-1][:5]
    return prediction, [(classes[i], round(proba[i]*100, 1)) for i in top5_idx]

# ── Header ──────────────────────────────────────────────
col1, col2 = st.columns([1, 5])
with col1:
    st.markdown("## 🎙️")
with col2:
    st.markdown("## Indian Language Detector")
    st.caption("Powered by MFCC + SVM · 87.5% accuracy")

st.markdown("---")

c1, c2, c3 = st.columns(3)
c1.metric("Languages", "10")
c2.metric("Accuracy", "87.5%")
c3.metric("Model", "SVM")

st.markdown("---")

# ── Tabs: Upload vs Live Mic ─────────────────────────────
tab1, tab2 = st.tabs(["📁 Upload Audio", "🎤 Record Live"])

# ── Tab 1: Upload ────────────────────────────────────────
with tab1:
    uploaded_file = st.file_uploader("Upload .wav or .mp3 file", type=["wav", "mp3"])

    if uploaded_file is not None:
        st.audio(uploaded_file)

        with st.spinner("🔍 Analysing..."):
            suffix = ".wav" if uploaded_file.name.endswith(".wav") else ".mp3"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name

            try:
                prediction, top5 = predict_language(tmp_path)
                st.markdown("---")
                st.success(f"✅ Detected language: **{prediction}**")
                st.markdown("#### 📊 Top 5 predictions")
                for lang, conf in top5:
                    st.progress(int(conf), text=f"{lang} — {conf}%")
            except Exception as e:
                st.error(f"❌ Error: {e}")
            finally:
                os.unlink(tmp_path)

# ── Tab 2: Live Mic ──────────────────────────────────────
with tab2:
    st.info("🎤 Record your voice below — speak for 3-5 seconds in any Indian language")

    audio_data = st.audio_input("Press to record")

    if audio_data is not None:
        st.audio(audio_data)

        with st.spinner("🔍 Analysing your voice..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tmp.write(audio_data.getvalue())
                tmp_path = tmp.name

            try:
                prediction, top5 = predict_language(tmp_path)
                st.markdown("---")
                st.success(f"✅ Detected language: **{prediction}**")
                st.markdown("#### 📊 Top 5 predictions")
                for lang, conf in top5:
                    st.progress(int(conf), text=f"{lang} — {conf}%")
            except Exception as e:
                st.error(f"❌ Error: {e}")
            finally:
                os.unlink(tmp_path)

st.markdown("---")
st.caption("Indian Languages Audio Dataset · MFCC + Chroma + Mel features · SVM classifier")
