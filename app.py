import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
import torch
import tempfile
import hashlib
import traceback 
from typing import Dict, Tuple

import streamlit as st
from faster_whisper import WhisperModel
from transformers import pipeline
from streamlit_mic_recorder import mic_recorder


# =========================
# 页面基础设置
# =========================
st.set_page_config(
    page_title="Speech-to-Text Emotion Demo",
    page_icon="🎙️",
    layout="centered",
)

st.title("🎙️ Speech-to-Text Emotion Demo")
st.write("支持文本输入、音频上传和麦克风录音。音频会先转写，再进行文本情绪识别。")
st.caption("录音结束后将自动开始分析；文本和上传音频需要点击按钮提交。")


# =========================
# Session State 初始化
# =========================
if "last_record_hash" not in st.session_state:
    st.session_state.last_record_hash = None

if "last_result" not in st.session_state:
    st.session_state.last_result = None


# =========================
# 模型加载：只加载一次
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
        "text-classification",
        model="j-hartmann/emotion-english-distilroberta-base",
        return_all_scores=True,
        truncation=True,
    )


# =========================
# 标签映射
# =========================
def map_emotion_label(raw_label: str) -> str:
    mapping = {
        "anger": "Anger",
        "angry": "Anger",
        "disgust": "Disgust",
        "fear": "Fear",
        "joy": "Joy",
        "happy": "Joy",
        "neutral": "Neutral",
        "sadness": "Sadness",
        "sad": "Sadness",
        "surprise": "Surprise",
        "love": "Positive",
    }
    return mapping.get(raw_label.lower(), raw_label)


# =========================
# ASR：音频转写
# =========================
def transcribe_audio(audio_path: str) -> str:
    model = load_asr_model()

    segments, _ = model.transcribe(
        audio_path,
        beam_size=1,
        vad_filter=True,
    )

    texts = []
    for seg in segments:
        if seg.text and seg.text.strip():
            texts.append(seg.text.strip())

    return " ".join(texts).strip()


# =========================
# 文本情绪识别
# =========================
def predict_emotion_from_text(text: str) -> Tuple[str, float, Dict[str, float]]:
    clf = load_emotion_pipeline()
    text = text[:500]
    results = clf(text)
    scores = results[0]

    score_dict: Dict[str, float] = {}
    for item in scores:
        label = map_emotion_label(item["label"])
        score = float(item["score"])
        score_dict[label] = max(score_dict.get(label, 0.0), score)

    best_label = max(score_dict, key=score_dict.get)
    best_score = score_dict[best_label]
    return best_label, best_score, score_dict


# =========================
# 反馈建议
# =========================
def generate_feedback(label: str, confidence: float) -> str:
    if label == "Joy":
        return "情绪整体偏积极，可以继续保持当前交流状态。"
    if label == "Neutral":
        return "情绪较为平稳，建议结合上下文进一步观察。"
    if label == "Sadness":
        return "检测到偏低落情绪，建议给予更多耐心回应与情感支持。"
    if label == "Anger":
        return "检测到较强负面情绪，建议降低刺激、放缓交流节奏。"
    if label == "Fear":
        return "检测到紧张或不安情绪，建议优先安抚并确认需求。"
    if label == "Surprise":
        return "检测到惊讶情绪，建议结合具体语境判断其正负倾向。"
    if label == "Disgust":
        return "检测到明显排斥或反感情绪，建议谨慎调整互动内容。"
    if label == "Positive":
        return "整体情绪较积极，适合继续推进当前互动。"
    return f"当前识别为 {label}，建议结合具体场景进一步判断。"


# =========================
# 统一分析逻辑
# =========================
def analyze_text(text: str):
    label, confidence, score_dict = predict_emotion_from_text(text)
    feedback = generate_feedback(label, confidence)
    return {
        "transcript": text,
        "emotion": label,
        "confidence": confidence,
        "scores": score_dict,
        "feedback": feedback,
    }


def analyze_audio_path(audio_path: str):
    transcript = transcribe_audio(audio_path)
    if not transcript:
        raise ValueError("未识别到有效语音内容。")

    label, confidence, score_dict = predict_emotion_from_text(transcript)
    feedback = generate_feedback(label, confidence)

    return {
        "transcript": transcript,
        "emotion": label,
        "confidence": confidence,
        "scores": score_dict,
        "feedback": feedback,
    }


def save_uploaded_file_to_temp(uploaded_file) -> str:
    suffix = os.path.splitext(uploaded_file.name)[-1]
    if not suffix:
        suffix = ".wav"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.read())
        return tmp.name


def save_recorded_audio_to_temp(audio_dict) -> str:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(audio_dict["bytes"])
        return tmp.name


def get_audio_hash(audio_bytes: bytes) -> str:
    return hashlib.md5(audio_bytes).hexdigest()


# =========================
# 结果展示
# =========================
def show_result(result: dict):
    st.success("分析完成")

    st.subheader("结果")
    st.write("**转写文本 / 输入文本：**")
    st.write(result["transcript"])

    col1, col2 = st.columns(2)
    with col1:
        st.metric("情绪标签", result["emotion"])
    with col2:
        st.metric("置信度", f'{result["confidence"]:.2%}')

    st.write("**建议反馈：**")
    st.info(result["feedback"])

    st.write("**Top 情绪分数：**")
    sorted_scores = sorted(
        result["scores"].items(),
        key=lambda x: x[1],
        reverse=True
    )

    for label, score in sorted_scores[:5]:
        st.progress(float(score), text=f"{label}: {score:.2%}")


# =========================
# 页面输入区
# =========================
st.subheader("输入方式")

text_input = st.text_area(
    "文本输入",
    placeholder="请输入一句话，例如：I feel really tired today, I want to stop.",
    height=130,
)

uploaded_audio = st.file_uploader(
    "音频上传",
    type=["wav", "mp3", "m4a", "mp4", "mpeg4"],
    help="上传语音文件后，系统会先转写，再进行情绪分析。",
)

st.write("或使用麦克风录音：")
recorded_audio = mic_recorder(
    start_prompt="开始录音",
    stop_prompt="停止录音",
    just_once=True,
    use_container_width=True,
    format="wav",
)

analyze_button = st.button("识别并分析情绪", type="primary")


# =========================
# 触发逻辑
# 优先级：新录音自动分析 > 上传音频按钮分析 > 文本按钮分析
# =========================
has_text = bool(text_input and text_input.strip())
has_upload = uploaded_audio is not None
has_record = recorded_audio is not None and "bytes" in recorded_audio

auto_record_trigger = False
current_record_hash = None

if has_record:
    current_record_hash = get_audio_hash(recorded_audio["bytes"])
    if current_record_hash != st.session_state.last_record_hash:
        auto_record_trigger = True
        st.session_state.last_record_hash = current_record_hash

should_analyze = analyze_button or auto_record_trigger

if should_analyze:
    if not has_text and not has_upload and not has_record:
        st.warning("请至少输入文本、上传音频或录制一段语音。")
    else:
        temp_path = None
        try:
            with st.spinner("处理中，请稍候..."):
                if auto_record_trigger:
                    temp_path = save_recorded_audio_to_temp(recorded_audio)
                    result = analyze_audio_path(temp_path)

                elif has_upload:
                    temp_path = save_uploaded_file_to_temp(uploaded_audio)
                    result = analyze_audio_path(temp_path)

                elif has_text:
                    result = analyze_text(text_input.strip())

                else:
                    raise ValueError("未检测到可分析的输入。")

            st.session_state.last_result = result
        except Exception as e:
            st.error(f"处理失败：{type(e).__name__}: {e}")
            st.code(traceback.format_exc())
        

        finally:
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)


# =========================
# 展示最近一次结果
# =========================
if st.session_state.last_result is not None:
    show_result(st.session_state.last_result)