import os

# Reduce OpenMP-related runtime conflicts
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import tempfile
import hashlib
import traceback
from typing import Dict, Tuple

import streamlit as st
from faster_whisper import WhisperModel
from transformers import pipeline


# =========================
# Page config
# =========================
st.set_page_config(
    page_title="Voice Emotion Analyzer",
    page_icon="🎙️",
    layout="centered",
)

st.title("🎙️ Voice Emotion Analyzer")
st.write("Speak into your microphone and detect your emotional state in real time.")
st.caption("Recording will automatically trigger analysis.")
st.caption("Please speak clearly for 2–5 seconds after clicking Record.")


# =========================
# Session state
# =========================
if "last_record_hash" not in st.session_state:
    st.session_state.last_record_hash = None

if "last_result" not in st.session_state:
    st.session_state.last_result = None


# =========================
# Model loading
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
# Emotion mapping
# =========================
def map_emotion_label(label: str) -> Tuple[str, str]:
    mapping = {
        "anger": ("Anger", "😡"),
        "angry": ("Anger", "😡"),
        "disgust": ("Disgust", "🤢"),
        "fear": ("Fear", "😨"),
        "joy": ("Joy", "😊"),
        "happy": ("Joy", "😊"),
        "neutral": ("Neutral", "😐"),
        "sadness": ("Sadness", "😢"),
        "sad": ("Sadness", "😢"),
        "surprise": ("Surprise", "😲"),
        "love": ("Love", "❤️"),
    }
    return mapping.get(label.lower(), (label, "❓"))

def transcribe_audio(audio_path: str) -> str:
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    file_size = os.path.getsize(audio_path)
    if file_size == 0:
        raise ValueError("The recorded audio file is empty.")

    model = load_asr_model()

    last_error = None
    for use_vad in [True, False]:
        try:
            segments, _ = model.transcribe(
                audio_path,
                beam_size=1,
                vad_filter=use_vad,
            )
            texts = [seg.text.strip() for seg in segments if seg.text and seg.text.strip()]
            transcript = " ".join(texts).strip()

            if transcript:
                return transcript

        except Exception as e:
            last_error = e

    raise ValueError(
        "No speech could be transcribed. Please record a longer and clearer voice sample."
    ) from last_error

# =========================
# Emotion prediction
# =========================
def predict_emotion(text: str) -> Tuple[str, str, float, Dict[str, float]]:
    clf = load_emotion_pipeline()
    text = text[:500]
    results = clf(text)

    scores = results[0] if len(results) > 0 and isinstance(results[0], list) else results

    score_dict: Dict[str, float] = {}
    for item in scores:
        mapped_label, _ = map_emotion_label(item["label"])
        score = float(item["score"])
        score_dict[mapped_label] = max(score_dict.get(mapped_label, 0.0), score)

    best_label = max(score_dict, key=score_dict.get)
    best_score = score_dict[best_label]
    _, best_emoji = map_emotion_label(best_label)

    return best_label, best_emoji, best_score, score_dict


# =========================
# Feedback
# =========================
def generate_feedback(label: str) -> str:
    feedback = {
        "Joy": "You sound positive and upbeat. Keep it up.",
        "Neutral": "Your emotion appears stable. More context may help interpret it better.",
        "Sadness": "You may be feeling low. Consider offering emotional support and patience.",
        "Anger": "You may sound frustrated or upset. Slowing down and taking a pause may help.",
        "Fear": "You may sound anxious or uneasy. Try to relax and speak again if needed.",
        "Surprise": "A surprised emotional response was detected.",
        "Disgust": "A strong negative reaction was detected.",
        "Love": "A warm and positive emotional tone was detected.",
    }
    return feedback.get(label, "An emotion was detected from your voice.")


# =========================
# Result display
# =========================
def show_result(result: dict):
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
        emoji = map_emotion_label(label)[1]
        st.progress(float(score), text=f"{emoji} {label}: {score:.2%}")


# =========================
# UI
# =========================
st.subheader("🎤 Record your voice")

audio_file = st.file_uploader(
    "Upload an audio file",
    type=["wav", "mp3"]
)

if audio_file is not None:
    st.audio(audio_file)


# =========================
# Auto trigger after recording
# =========================
if audio_file is not None:
    audio_bytes = audio_file.getvalue()
    current_hash = hashlib.md5(audio_bytes).hexdigest()

    if current_hash != st.session_state.last_record_hash:
        st.session_state.last_record_hash = current_hash

        temp_path = None
        try:
            with st.spinner("Analyzing..."):
                if not audio_bytes:
                    raise ValueError("No audio data was recorded.")

                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                    tmp.write(audio_bytes)
                    temp_path = tmp.name

                # very short recordings often fail
                if os.path.getsize(temp_path) < 2000:
                    raise ValueError(
                        "The recording is too short. Please speak for at least 2–3 seconds."
                    )

                text = transcribe_audio(temp_path)
                label, emoji, score, score_dict = predict_emotion(text)

                st.session_state.last_result = {
                    "text": text,
                    "label": label,
                    "emoji": emoji,
                    "score": score,
                    "scores": score_dict,
                }

        except ValueError as e:
            st.warning(str(e))

        except Exception as e:
            st.error(f"Error: {type(e).__name__}")
            st.code(traceback.format_exc())

        finally:
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)


# =========================
# Show latest result
# =========================
if st.session_state.last_result is not None:
    show_result(st.session_state.last_result)