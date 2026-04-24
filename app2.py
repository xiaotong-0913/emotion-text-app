import os

# 解决 OpenMP 冲突（云端/本地都稳）
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import tempfile
import hashlib
import traceback
from typing import Dict, Tuple

import streamlit as st
from faster_whisper import WhisperModel
from transformers import pipeline
from streamlit_mic_recorder import mic_recorder


# =========================
# 页面设置
# =========================
st.set_page_config(
    page_title="Voice Emotion Analyzer",
    page_icon="🎙️",
    layout="centered",
)

st.title("🎙️ Voice Emotion Analyzer")
st.write("Speak into your microphone and detect your emotional state in real time.")
st.caption("Recording will automatically trigger analysis.")


# =========================
# Session State
# =========================
if "last_record_hash" not in st.session_state:
    st.session_state.last_record_hash = None

if "last_result" not in st.session_state:
    st.session_state.last_result = None


# =========================
# 模型加载
# =========================
@st.cache_resource
def load_asr_model():
    return WhisperModel(
        "tiny",
        device="cpu",
        compute_type="int8",
        cpu_threads=1,
        num_workers=1,
    )


@st.cache_resource
def load_emotion_pipeline():
    return pipeline(
        task="text-classification",
        model="j-hartmann/emotion-english-distilroberta-base",
        framework="pt",
        device=-1,
        top_k=None,
        truncation=True,
    )


# =========================
# Emotion + Emoji
# =========================
def map_emotion_label(label: str) -> Tuple[str, str]:
    mapping = {
        "anger": ("Anger", "😡"),
        "disgust": ("Disgust", "🤢"),
        "fear": ("Fear", "😨"),
        "joy": ("Joy", "😊"),
        "neutral": ("Neutral", "😐"),
        "sadness": ("Sadness", "😢"),
        "surprise": ("Surprise", "😲"),
        "love": ("Love", "❤️"),
    }
    key = label.lower()
    return mapping.get(key, (label, "❓"))


# =========================
# ASR
# =========================
def transcribe_audio(audio_path: str) -> str:
    model = load_asr_model()

    segments, _ = model.transcribe(
        audio_path,
        beam_size=1,
        vad_filter=True,
    )

    texts = [seg.text.strip() for seg in segments if seg.text]
    return " ".join(texts).strip()


# =========================
# Emotion Prediction
# =========================
def predict_emotion(text: str):
    clf = load_emotion_pipeline()
    text = text[:500]
    results = clf(text)

    scores = results[0] if isinstance(results[0], list) else results

    score_dict = {}
    for item in scores:
        label, emoji = map_emotion_label(item["label"])
        score_dict[label] = float(item["score"])

    best_label = max(score_dict, key=score_dict.get)
    best_score = score_dict[best_label]

    label, emoji = map_emotion_label(best_label)
    return label, emoji, best_score, score_dict


# =========================
# Feedback
# =========================
def generate_feedback(label: str) -> str:
    feedback = {
        "Joy": "You're feeling positive. Keep it up!",
        "Neutral": "Emotion is stable. Observe further context.",
        "Sadness": "You seem down. Consider emotional support.",
        "Anger": "You may be frustrated. Take a pause.",
        "Fear": "You may feel anxious. Try to relax.",
        "Surprise": "Unexpected reaction detected.",
        "Disgust": "Strong negative reaction detected.",
        "Love": "Positive emotional attachment detected.",
    }
    return feedback.get(label, "Emotion detected.")


# =========================
# UI: Recording
# =========================
st.subheader("🎤 Record your voice")

audio = mic_recorder(
    start_prompt="Start Recording",
    stop_prompt="Stop Recording",
    just_once=True,
    use_container_width=True,
    format="wav",
)


# =========================
# 自动触发
# =========================
if audio and "bytes" in audio:
    current_hash = hashlib.md5(audio["bytes"]).hexdigest()

    if current_hash != st.session_state.last_record_hash:
        st.session_state.last_record_hash = current_hash

        try:
            with st.spinner("Analyzing..."):
                # 保存音频
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                    tmp.write(audio["bytes"])
                    temp_path = tmp.name

                # 转写
                text = transcribe_audio(temp_path)

                # 情绪分析
                label, emoji, score, score_dict = predict_emotion(text)

                st.session_state.last_result = {
                    "text": text,
                    "label": label,
                    "emoji": emoji,
                    "score": score,
                    "scores": score_dict,
                }

                os.remove(temp_path)

        except Exception as e:
            st.error(f"Error: {type(e).__name__}")
            st.code(traceback.format_exc())


# =========================
# 展示结果
# =========================
if st.session_state.last_result:
    result = st.session_state.last_result

    st.success("Analysis Complete")

    st.subheader("📝 Transcription")
    st.write(result["text"])

    st.subheader("🎯 Emotion")
    st.markdown(f"## {result['emoji']} {result['label']}")

    st.metric("Confidence", f"{result['score']:.2%}")

    st.subheader("💡 Feedback")
    st.info(generate_feedback(result["label"]))

    st.subheader("📊 Emotion Distribution")
    sorted_scores = sorted(result["scores"].items(), key=lambda x: x[1], reverse=True)

    for label, score in sorted_scores:
        st.progress(score, text=f"{label}: {score:.2%}")