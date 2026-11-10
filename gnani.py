"""Thin clients for Gnani Prisma (STT), Gnani Timbre (TTS) and OpenAI-compatible Evon."""
import json
import os
import re
import time

import requests
import streamlit as st

STT_URL = "https://api.vachana.ai/stt/v3"
TTS_URL = "https://api.vachana.ai/api/v1/tts/inference"


def get_setting(name, default=""):
    """Read configuration from the environment first, then Streamlit secrets."""
    value = os.environ.get(name)
    if value:
        return value
    try:
        return st.secrets.get(name, default)
    except Exception:
        return default


def get_gnani_key():
    return get_setting("GNANI_API_KEY")


def get_evon_key():
    return get_setting("EVON_API_KEY")


LANGS = {
    "hi-IN": "Hindi", "en-IN": "English", "ta-IN": "Tamil", "te-IN": "Telugu",
    "kn-IN": "Kannada", "ml-IN": "Malayalam", "mr-IN": "Marathi",
    "bn-IN": "Bengali", "gu-IN": "Gujarati", "pa-IN": "Punjabi",
}

V3_VOICES = {
    "Shivani": "Female · warm, empathetic", "Lakshmi": "Female · expressive storyteller",
    "Kalpana": "Female · composed, empathetic", "Chetana": "Female · gentle",
    "Vasudha": "Female · expressive, composed", "Riyaa": "Female · confident, lively",
    "Tarika": "Female · warm, reassuring", "Avinash": "Male · professional, firm",
    "Shubham": "Male · energetic", "Prakash": "Male · steady",
    "Devansh": "Male · young, conversational", "Yashraj": "Male · warm narrative",
    "Harshad": "Male · professional",
}

AUDIO_TAGS = ["<laugh>", "<giggle>", "<sigh>", "<whispers>", "<pauses>", "<excited>",
              "<shouting>", "<slow>", "<exhales>", "<clears throat>"]


class GnaniError(RuntimeError):
    pass


def _err(response):
    try:
        data = response.json()
    except Exception:
        return response.text[:300]
    if isinstance(data, dict):
        detail = data.get("detail")
        if isinstance(detail, dict):
            return detail.get("message") or str(detail)
        if isinstance(detail, list):
            return json.dumps(detail, ensure_ascii=False)[:300]
        error = data.get("error")
        if isinstance(error, dict):
            return error.get("message") or str(error)
        return data.get("message") or str(data)[:300]
    return str(data)[:300]


def _post(url, retries=4, timeout=120, **kwargs):
    response = None
    for attempt in range(retries):
        response = requests.post(url, timeout=timeout, **kwargs)
        if response.status_code in (429, 502, 503, 504) and attempt < retries - 1:
            time.sleep(1.5 * (2 ** attempt))
            continue
        break
    return response


def _transcription(path, language_code, model, api_key, diarization):
    key = api_key or get_gnani_key()
    if not key:
        raise GnaniError("GNANI_API_KEY is not configured in Streamlit secrets or the environment.")
    with open(path, "rb") as audio_file:
        audio = audio_file.read()
    data = {"language_code": language_code, "format": "transcribe"}
    if model == "prisma-v3.0":
        data["model"] = model
    if diarization:
        data["speaker_diarization"] = "true"
    response = _post(
        STT_URL,
        headers={"X-API-Key-ID": key},
        files={"audio_file": (os.path.basename(path), audio, "audio/wav")},
        data=data,
    )
    if response.status_code != 200:
        raise GnaniError(f"Prisma STT {response.status_code}: {_err(response)}")
    data = response.json()
    if data.get("success") is False:
        raise GnaniError(f"Prisma STT failed: {_err(response)}")
    segments = data.get("segments") or data.get("utterances") or data.get("speaker_segments") or []
    transcript = data.get("transcript") or ""
    if isinstance(transcript, list):
        segments = segments or transcript
        transcript = " ".join(
            str(item.get("text", "")) for item in transcript if isinstance(item, dict)
        )
    return {
        "text": str(transcript).strip(),
        "language": data.get("detected_language_code"),
        "segments": segments,
        "speakers": data.get("speakers") or [],
    }


def transcribe_diarized(path, language_code, model="prisma-v2.5", api_key=None):
    """Transcribe with Prisma speaker diarization and return utterance metadata."""
    return _transcription(path, language_code, model, api_key, diarization=True)


def transcribe(path, language_code, model="prisma-v2.5", api_key=None):
    """Return a backward-compatible transcript and detected-language pair."""
    result = _transcription(path, language_code, model, api_key, diarization=False)
    return result["text"], result["language"]


def synthesize(text, voice, model, language, speed=1.0, sample_rate=24000, api_key=None):
    """Timbre text-to-speech using configured credentials."""
    key = api_key or get_gnani_key()
    if not key:
        raise GnaniError("GNANI_API_KEY is not configured in Streamlit secrets or the environment.")
    body = {
        "text": text,
        "voice": voice,
        "model": model,
        "language": language,
        "audio_config": {
            "sample_rate": sample_rate, "num_channels": 1, "sample_width": 2,
            "encoding": "linear_pcm", "container": "wav",
        },
    }
    if abs(speed - 1.0) > 1e-3:
        body["speed"] = max(0.85, min(1.15, speed))
    response = _post(
        TTS_URL,
        headers={"X-API-Key-ID": key, "Content-Type": "application/json"},
        json=body,
    )
    if response.status_code != 200:
        raise GnaniError(f"Timbre TTS {response.status_code}: {_err(response)}")
    if len(response.content) < 100:
        raise GnaniError("Timbre TTS returned an empty audio body")
    return response.content


def _strip_think(text):
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    if "</think>" in text:
        text = text.split("</think>")[-1]
    return text.strip()


def chat(messages, base_url, model, temperature=0.3, max_tokens=4096,
         disable_thinking=True, api_key=None):
    """OpenAI-compatible chat completion for Evon."""
    url = base_url.rstrip("/") + "/chat/completions"
    key = api_key or get_evon_key()
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = "Bearer " + key
    body = {
        "model": model, "messages": messages, "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if disable_thinking:
        body["reasoning"] = {"effort": "none", "exclude": True}
    response = _post(url, headers=headers, json=body, timeout=180)
    if response.status_code == 400 and disable_thinking:
        body.pop("chat_template_kwargs", None)
        response = _post(url, headers=headers, json=body, timeout=180)
    if response.status_code != 200:
        raise GnaniError(f"Evon endpoint {response.status_code}: {_err(response)}")
    try:
        data = response.json()
    except ValueError as error:
        raise GnaniError(f"Evon returned invalid JSON: {response.text[:300]}") from error
    if not isinstance(data, dict):
        raise GnaniError(f"Unexpected Evon response type: {type(data).__name__}")
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise GnaniError(f"Evon response has no choices; response fields: {sorted(data)}")
    choice = choices[0]
    message = choice.get("message")
    if isinstance(message, dict):
        content = message.get("content")
        if isinstance(content, list):
            content = "".join(
                part if isinstance(part, str) else (
                    part.get("text", "") if isinstance(part.get("text"), str) else ""
                )
                for part in content if isinstance(part, (str, dict))
            )
        if isinstance(content, str) and content.strip():
            return _strip_think(content)
        message_fields = sorted(message)
    else:
        message_fields = []
    raise GnaniError(
        "Evon returned no text content "
        f"(finish_reason={choice.get('finish_reason')!r}, "
        f"message_fields={message_fields}, response_fields={sorted(data)}, "
        f"usage={data.get('usage')!r}). "
        "Check the selected model and its OpenRouter provider settings."
    )
