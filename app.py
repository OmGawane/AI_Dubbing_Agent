"""Bhaav: a Streamlit app for speaker-aware semantic video localization."""
import shutil
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

import gnani
import localize as loc
import mock
import pipeline as pl

st.set_page_config(page_title="Bhaav · Video Localization", page_icon="🎬", layout="wide")
st.markdown("""
<style>
.block-container {padding-top: 2rem; max-width: 1250px;}
h1 {letter-spacing: -0.02em; margin-bottom: 0;}
.sub {color: #6b7280; margin-bottom: 1.2rem;}
</style>
""", unsafe_allow_html=True)

st.title("🎬 Bhaav")
st.markdown(
    '<div class="sub">Meaning-aware dubbing with speaker diarization and distinct character voices.</div>',
    unsafe_allow_html=True,
)

state = st.session_state
for key, value in {"work": None, "segments": None, "result": None, "media": None,
                   "has_video": False, "voice_map": {}}.items():
    state.setdefault(key, value)

with st.sidebar:
    st.header("Options")
    mock_mode = st.toggle(
        "Mock mode",
        value=False,
        help="Runs offline with sample transcripts, translations, and synthetic tones.",
    )
    stt_model = st.selectbox("Prisma model", ["prisma-v3.0", "prisma-v2.5"])
    max_seconds = st.slider("Maximum clip length (seconds)", 10, 180, 60, 10)
    bed_volume = st.slider("Original audio underneath (%)", 0, 40, 0, 5)

if not shutil.which("ffmpeg"):
    st.error("FFmpeg is required for audio extraction and timing. Install it and add it to PATH.")
    st.stop()

uploaded = st.file_uploader(
    "Upload a video or audio clip",
    type=["mp4", "mov", "mkv", "webm", "mp3", "wav", "m4a"],
)
source_col, target_col = st.columns(2)
with source_col:
    source_language = st.selectbox(
        "Source language", list(gnani.LANGS), format_func=gnani.LANGS.get, index=3,
    )
with target_col:
    target_language = st.selectbox(
        "Target language", list(gnani.LANGS), format_func=gnani.LANGS.get, index=1,
    )
scene = st.text_area(
    "Scene context (optional)",
    placeholder="For example: Two old friends arguing at a tea stall; sarcastic but affectionate.",
    height=80,
)

gnani_key = gnani.get_gnani_key()
evon_url = gnani.get_setting("EVON_BASE_URL", "http://localhost:8000/v1")
evon_key = gnani.get_evon_key()
evon_model = gnani.get_setting("EVON_MODEL", "Qwen/Qwen3-1.7B")
if not mock_mode and not gnani_key:
    st.info("Configure GNANI_API_KEY in Streamlit secrets or the environment to enable Prisma and Timbre.")
if not mock_mode and not gnani.get_setting("EVON_BASE_URL"):
    st.info("Configure EVON_BASE_URL in Streamlit secrets or the environment for your model API.")


def run_pipeline():
    previous_work = state.get("work")
    if previous_work and Path(previous_work).is_dir():
        shutil.rmtree(previous_work)
    work = Path(tempfile.mkdtemp(prefix="bhaav_"))
    state.work = str(work)
    state.result = None

    media = work / ("input" + Path(uploaded.name).suffix.lower())
    media.write_bytes(uploaded.getbuffer())
    info = pl.probe(media)
    if not info["has_audio"]:
        raise RuntimeError("The uploaded file has no audio track.")
    duration = min(info["duration"], float(max_seconds))
    audio = work / "audio16k.wav"
    progress = st.progress(0.0, text="Extracting audio…")
    pl.extract_audio(media, audio, duration)
    spans = pl.speech_spans(audio, duration)
    segments = pl.build_segments(spans)
    if not segments:
        raise RuntimeError("No speech was detected in the processed clip.")

    mock._state["i"] = 0

    def transcribe_fn(path):
        if mock_mode:
            return mock.stt(path)
        return gnani.transcribe_diarized(path, source_language, model=stt_model)

    segments = pl.transcribe_all(
        segments, audio, work, transcribe_fn,
        lambda done, total, message: progress.progress(0.1 + 0.3 * done / total, text=message),
    )
    if not segments:
        raise RuntimeError("Prisma returned no transcript. Check the source language and audio.")

    def llm(messages):
        if mock_mode:
            return mock.llm(messages)
        return gnani.chat(messages, evon_url, evon_model, api_key=evon_key)

    progress.progress(0.45, text="Localizing dialogue and preserving speaker identities…")
    loc.localize(segments, source_language, target_language, llm, scene=scene, emotion_tags=True)
    voice_map = pl.assign_voices(segments, gnani.V3_VOICES)

    def synthesize(text, voice):
        if mock_mode:
            return mock.tts(text)
        return gnani.synthesize(text, voice, "timbre-v3.0", target_language)

    condense = None if mock_mode else lambda text, chars: loc.condense(
        llm, text, chars, target_language,
    )
    progress.progress(0.55, text="Synthesizing speaker-matched voices and rendering…")
    result = pl.dub(
        segments, duration, media, work, info["has_video"], synthesize, voice_map,
        condense=condense, bed_volume=bed_volume / 100.0,
        progress=lambda done, total, message: progress.progress(
            0.55 + 0.44 * done / total, text=message,
        ),
    )
    progress.progress(1.0, text="Dub complete")
    state.segments = segments
    state.media = str(media)
    state.has_video = info["has_video"]
    state.voice_map = voice_map
    state.result = result


run_disabled = uploaded is None or (
    not mock_mode and (not gnani_key or not gnani.get_setting("EVON_BASE_URL"))
)
if st.button("Localize and dub", type="primary", disabled=run_disabled, use_container_width=True):
    try:
        run_pipeline()
    except Exception as error:
        st.error(f"{type(error).__name__}: {error}")

if state.result:
    st.divider()
    st.subheader("Dubbed result")
    original_col, dubbed_col = st.columns(2)
    with original_col:
        st.markdown("**Original**")
        (st.video if state.has_video else st.audio)(state.media)
    with dubbed_col:
        st.markdown("**Localized dub**")
        (st.video if state.has_video else st.audio)(state.result["out"])

    st.markdown("**Speaker voice mapping**")
    st.write(", ".join(
        f"{speaker}: {voice} ({gnani.V3_VOICES[voice]})"
        for speaker, voice in state.voice_map.items()
    ))
    if len(state.voice_map) == 1 and any(
        not segment.get("speaker_id_observed", True) for segment in state.segments
    ):
        st.warning(
            "The transcript did not contain reliable speaker labels, and dialogue context yielded only one speaker. "
            "Check the script if this scene contains multiple characters."
        )
    report = pd.DataFrame([{
        "Speaker": segment["speaker_id"],
        "Voice": segment.get("voice", ""),
        "Time": f"{segment['start']:.1f}–{segment['end']:.1f}s",
        "Original": segment["text"],
        "Localized": segment["text_final"],
        "Voice style": segment.get("voice_style", ""),
        "Sync": segment["status"],
    } for segment in state.segments])
    st.dataframe(report, hide_index=True, use_container_width=True)
    st.download_button(
        "Download dubbed output",
        Path(state.result["out"]).read_bytes(),
        file_name=Path(state.result["out"]).name,
        use_container_width=True,
    )
    st.download_button(
        "Download subtitles (.srt)",
        state.result["srt"],
        file_name="dub.srt",
        use_container_width=True,
    )
    script = pd.DataFrame([{
        "speaker": segment["speaker_id"],
        "voice": segment.get("voice", ""),
        "start": segment["start"],
        "end": segment["end"],
        "original": segment["text"],
        "literal": segment.get("literal", ""),
        "localized": segment["text_final"],
        "tone": segment.get("tone", ""),
        "voice_style": segment.get("voice_style", ""),
        "note": segment.get("note", ""),
    } for segment in state.segments])
    st.download_button(
        "Download localized script (.csv)",
        script.to_csv(index=False),
        file_name="localized_script.csv",
        use_container_width=True,
    )
