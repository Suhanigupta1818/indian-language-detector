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

st.markdown("""
<style>
[data-testid="stAppViewContainer"] { max-width: 660px; margin: auto; }
.stat-card { background: #f8f9fa; border-radius: 10px; padding: 12px 16px; text-align: center; }
.lang-label { font-size: 13px; font-weight: 500; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    if not os.path.exists("language_model.pkl"):
        st.error("❌ language_model.pkl not found!")
        st.stop()
    return joblib.load("language_model.pkl")

model = load_model()

def extract_features(file_path):
    signal, sr = librosa.load(file_path, sr=16000, duration=2)
    mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=13)
    return np.mean(mfcc, axis=1)

col1, col2 = st.columns([1, 4])
with col1:
    st.markdown("###  🎙️")
with col2:
    st.markdown("## Indian Language Detector")
    st.caption("Powered by MFCC + Random Forest")

st.markdown("---")

c1, c2, c3 = st.columns(3)
c1.metric("Languages", "10")
c2.metric("Accuracy", "73.7%")
c3.metric("Model size", "1.6 MB")

st.markdown("---")

uploaded_file = st.file_uploader("📁 Upload audio file (.wav or .mp3)", type=["wav", "mp3"])

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
            st.success(f"✅  Detected language: **{prediction}**")

            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(features)[0]
                classes = model.classes_
                top5_idx = np.argsort(proba)[::-1][:5]

                st.markdown("#### 📊 All predictions")
                for idx in top5_idx:
                    lang = classes[idx]
                    conf = round(proba[idx] * 100, 1)
                    st.progress(int(conf), text=f"{lang}  —  {conf}%")

        except Exception as e:
            st.error(f"❌ Error: {e}")
        finally:
            os.unlink(tmp_path)

st.markdown("---")
st.caption("Indian Languages Audio Dataset · MFCC features · Random Forest classifier")
