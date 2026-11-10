"""Offline stand-ins so the full pipeline can be exercised without API keys. Clearly labelled in the UI."""
import io
import json
import math
import re
import wave

import numpy as np

_DEMO = [
    "यार, तुम तो बिल्कुल ऊँची दुकान फीका पकवान निकले",
    "अब पछताने से क्या होगा, जब चिड़िया चुग गई खेत",
    "चलो, कल बात करते हैं, आज बहुत थक गया हूँ",
    "ये तो बाएँ हाथ का खेल है, चिंता मत करो",
]
_state = {"i": 0}


def stt(_path):
    t = _DEMO[_state["i"] % len(_DEMO)]
    speaker = f"Speaker {_state['i'] % 2 + 1}"
    _state["i"] += 1
    return {"text": t, "speaker_id": speaker}


def llm(messages):
    payload = json.loads(messages[-1]["content"])
    if "Rewrite the line shorter" in messages[0]["content"]:
        return messages[-1]["content"].split("Line:")[-1].strip()[:30]
    out = []
    for ln in payload["lines"]:
        out.append({"id": ln["id"], "speaker_id": ln.get("speaker_id", "Speaker 1"),
                    "voice_style": "warm conversational", "literal": "[mock literal] " + ln["text"],
                    "localized": f"[mock {payload['target_language']}] " + ln["text"][: max(6, ln["max_chars"] - 20)],
                    "tone": "neutral", "note": "mock mode: no real adaptation"})
    return json.dumps(out, ensure_ascii=False)


def tts(text, seconds_per_char=0.065, sr=24000):
    text = re.sub(r"<[^>]+>", "", text)
    dur = max(0.4, len(text) * seconds_per_char)
    t = np.arange(int(dur * sr)) / sr
    x = 0.2 * np.sin(2 * math.pi * 220 * t) * (0.6 + 0.4 * np.sin(2 * math.pi * 3 * t))
    pcm = (x * 32767).astype(np.int16)
    b = io.BytesIO()
    with wave.open(b, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())
    return b.getvalue()