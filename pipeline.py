"""Audio/video plumbing: segmentation, timing-aware dubbing, mixing, muxing. Needs ffmpeg on PATH."""
import json
import math
import re
import subprocess
import wave
from pathlib import Path

import numpy as np

SR = 24000


def _run(cmd, data=None):
    p = subprocess.run(cmd, input=data, capture_output=True)
    if p.returncode != 0:
        raise RuntimeError(f"{cmd[0]} failed: {p.stderr[-600:].decode(errors='ignore')}")
    return p


def probe(path):
    p = _run(["ffprobe", "-v", "error", "-print_format", "json", "-show_streams", "-show_format", str(path)])
    j = json.loads(p.stdout)
    st = j.get("streams", [])
    has_video = any(s.get("codec_type") == "video" and not s.get("disposition", {}).get("attached_pic") for s in st)
    return {"duration": float(j["format"].get("duration", 0)), "has_video": has_video,
            "has_audio": any(s.get("codec_type") == "audio" for s in st)}


def extract_audio(src, out, max_seconds):
    _run(["ffmpeg", "-y", "-v", "error", "-i", str(src), "-t", f"{max_seconds:.3f}", "-vn",
          "-ac", "1", "-ar", "16000", str(out)])


def speech_spans(wav, duration, noise_db=-35, min_silence=0.35):
    """Speech intervals = complement of ffmpeg silencedetect output."""
    p = subprocess.run(["ffmpeg", "-v", "info", "-i", str(wav), "-af",
                        f"silencedetect=noise={noise_db}dB:d={min_silence}", "-f", "null", "-"],
                       capture_output=True, text=True)
    starts = [max(0.0, float(x)) for x in re.findall(r"silence_start: (-?[\d.]+)", p.stderr)]
    ends = [float(x) for x in re.findall(r"silence_end: ([\d.]+)", p.stderr)]
    sil = []
    for i, s in enumerate(starts):
        sil.append((s, ends[i] if i < len(ends) else duration))
    spans, cur = [], 0.0
    for s, e in sil:
        if s - cur >= 0.4:
            spans.append([cur, s])
        cur = max(cur, e)
    if duration - cur >= 0.4:
        spans.append([cur, duration])
    return spans


def build_segments(spans, max_len=8.0, max_gap=0.7, hard_cap=25.0):
    """Merge neighbouring speech spans into dubbable units (<= max_len), split anything beyond the 30 s STT limit."""
    merged = []
    for s, e in spans:
        if merged and s - merged[-1][1] <= max_gap and e - merged[-1][0] <= max_len:
            merged[-1][1] = e
        else:
            merged.append([s, e])
    out = []
    for s, e in merged:
        n = max(1, math.ceil((e - s) / hard_cap))
        step = (e - s) / n
        out += [[s + i * step, s + (i + 1) * step] for i in range(n)]
    return [{"idx": i, "start": round(s, 3), "end": round(e, 3)} for i, (s, e) in enumerate(out)]


def cut(wav, start, end, out, pad=0.15):
    ss = max(0.0, start - pad)
    _run(["ffmpeg", "-y", "-v", "error", "-ss", f"{ss:.3f}", "-t", f"{(end - start) + 2 * pad:.3f}",
          "-i", str(wav), "-ac", "1", "-ar", "16000", str(out)])


def _speaker_id(value):
    if value is None or value == "":
        return "Speaker 1"
    text = str(value).strip()
    match = re.fullmatch(r"(speaker[_ -]?)?(\d+)", text, flags=re.I)
    if match:
        number = int(match.group(2))
        if not match.group(1) or number == 0:
            number += 1
        return f"Speaker {number}"
    return text if text.lower().startswith("speaker ") else f"Speaker {text}"


def _utterances(result):
    if isinstance(result, dict):
        utterances = result.get("segments") or result.get("utterances") or []
        if isinstance(utterances, dict):
            utterances = [utterances]
        text = str(result.get("text") or result.get("transcript") or "").strip()
        if not utterances and text:
            utterances = [{"text": text, "speaker_id": result.get("speaker_id")}]
        speakers = result.get("speakers") or []
        metadata = {}
        for speaker in speakers:
            if isinstance(speaker, dict):
                sid = speaker.get("speaker_id", speaker.get("id"))
                if sid is not None:
                    metadata[_speaker_id(sid)] = speaker
        return utterances, metadata
    text = str(result or "").strip()
    return ([{"text": text}] if text else []), {}


def transcribe_all(segs, wav, workdir, stt_fn, progress=None):
    """Transcribe clips and preserve diarized utterance timing and speaker metadata."""
    diarized = []
    for i, s in enumerate(segs):
        clip = Path(workdir) / f"seg_{s['idx']:03d}.wav"
        cut(wav, s["start"], s["end"], clip)
        utterances, speaker_metadata = _utterances(stt_fn(str(clip)))
        usable = [u for u in utterances if isinstance(u, dict) and str(u.get("text", "")).strip()]
        total_chars = sum(len(str(u["text"])) for u in usable)
        cursor = s["start"]
        consumed_chars = 0
        parent_duration = s["end"] - s["start"]
        clip_start = max(0.0, s["start"] - 0.15)
        for u_index, utterance in enumerate(usable):
            text = str(utterance["text"]).strip()
            local_start = utterance.get("start", utterance.get("start_time"))
            local_end = utterance.get("end", utterance.get("end_time"))
            if local_start is None or local_end is None:
                start = s["start"] + parent_duration * consumed_chars / total_chars if total_chars else cursor
                consumed_chars += len(text)
                end = s["end"] if u_index == len(usable) - 1 else s["start"] + (
                    parent_duration * consumed_chars / total_chars if total_chars else parent_duration / len(usable)
                )
            else:
                start = max(s["start"], clip_start + float(local_start))
                end = min(s["end"], clip_start + float(local_end))
                if end <= start:
                    start, end = cursor, s["end"] if u_index == len(usable) - 1 else cursor
            cursor = max(cursor, end)
            raw_speaker = utterance.get("speaker_id", utterance.get("speaker"))
            speaker_observed = raw_speaker is not None and raw_speaker != ""
            speaker = _speaker_id(raw_speaker) if speaker_observed else "Unassigned"
            diarized.append({
                **s,
                "start": round(start, 3),
                "end": round(end, 3),
                "text": text,
                "speaker_id": speaker,
                "speaker_id_observed": speaker_observed,
                "speaker_metadata": speaker_metadata.get(speaker, {}),
            })
        if progress:
            progress(i + 1, len(segs), f"Transcribed {i + 1}/{len(segs)}")
    for i, s in enumerate(diarized):
        s["idx"] = i
    return diarized


def assign_voices(segs, voice_profiles):
    """Assign distinct Timbre profiles, honoring diarization gender metadata when present."""
    speakers = {}
    for segment in segs:
        sid = segment.get("speaker_id", "Speaker 1")
        entry = speakers.setdefault(sid, {"metadata": {}, "style": ""})
        entry["metadata"].update(segment.get("speaker_metadata") or {})
        if not entry["style"] and segment.get("voice_style"):
            entry["style"] = segment["voice_style"]

    assigned = {}
    used = set()
    for speaker_id, details in speakers.items():
        metadata = details["metadata"]
        gender = str(metadata.get("gender") or metadata.get("voice_gender") or "").lower()
        candidates = [
            (name, description) for name, description in voice_profiles.items()
            if not gender or gender in description.lower()
        ]
        if not candidates:
            candidates = list(voice_profiles.items())
        style_words = set(re.findall(r"[a-z]+", details["style"].lower()))
        ranked = sorted(
            candidates,
            key=lambda item: (
                len(style_words & set(re.findall(r"[a-z]+", item[1].lower()))),
                item[0] not in used,
            ),
            reverse=True,
        )
        selected = next((name for name, _ in ranked if name not in used), ranked[0][0])
        assigned[speaker_id] = selected
        used.add(selected)
    return assigned


# ---------- dubbing ----------

def decode(audio_bytes):
    """Any audio container -> mono float32 at SR (ffmpeg handles odd WAV headers from streaming TTS)."""
    p = _run(["ffmpeg", "-v", "error", "-i", "pipe:0", "-f", "s16le", "-ac", "1", "-ar", str(SR), "pipe:1"],
             data=audio_bytes)
    return np.frombuffer(p.stdout, dtype=np.int16).astype(np.float32) / 32768.0


def tempo_shift(pcm, tempo):
    raw = (np.clip(pcm, -1, 1) * 32767).astype(np.int16).tobytes()
    p = _run(["ffmpeg", "-v", "error", "-f", "s16le", "-ar", str(SR), "-ac", "1", "-i", "pipe:0",
              "-filter:a", f"atempo={tempo:.4f}", "-f", "s16le", "-ar", str(SR), "-ac", "1", "pipe:1"], data=raw)
    return np.frombuffer(p.stdout, dtype=np.int16).astype(np.float32) / 32768.0


def trim_silence(x, thr=0.01, keep=0.02):
    idx = np.where(np.abs(x) > thr)[0]
    if len(idx) == 0:
        return x
    k = int(keep * SR)
    return x[max(0, idx[0] - k): idx[-1] + k]


def fade_out(x, ms=40):
    n = min(len(x), int(ms * SR / 1000))
    if n:
        x = x.copy()
        x[-n:] *= np.linspace(1, 0, n)
    return x


def plan_slots(segs, media_dur):
    """slot = comfortable time budget; limit = hard ceiling before the next line starts."""
    for i, s in enumerate(segs):
        nxt = segs[i + 1]["start"] if i + 1 < len(segs) else media_dur
        s["limit"] = max(0.3, nxt - s["start"])
        s["slot"] = min((s["end"] - s["start"]) + 0.4, s["limit"])


def dub(segs, media_dur, src, workdir, has_video, synth, voice_map, condense=None, bed_volume=0.0,
        progress=None, max_tempo=1.35):
    """synth(text, voice)->audio bytes; returns paths, subtitles and per-line sync data."""
    workdir = Path(workdir)
    plan_slots(segs, media_dur)
    buf = np.zeros(int((media_dur + 1.0) * SR), dtype=np.float32)
    for i, s in enumerate(segs):
        text = s["localized"].strip()
        voice = voice_map[s.get("speaker_id", "Speaker 1")]
        pcm = trim_silence(decode(synth(text, voice)))
        dur = len(pcm) / SR
        status = "ok"
        if condense and dur > s["slot"] * max_tempo:
            plain_len = len(re.sub(r"<[^>]+>", "", text))
            target = max(6, int(plain_len * (s["slot"] * 1.1) / dur))
            shorter = condense(text, target)
            if shorter and shorter != text:
                text = shorter
                pcm = trim_silence(decode(synth(text, voice)))
                dur = len(pcm) / SR
                status = "condensed"
        tempo = 1.0
        if dur > s["slot"]:
            tempo = min(dur / s["slot"], max_tempo)
            pcm = tempo_shift(pcm, tempo)
            dur = len(pcm) / SR
            if status == "ok":
                status = "sped up"
        if dur > s["limit"]:
            pcm = fade_out(pcm[: int(s["limit"] * SR)])
            dur = len(pcm) / SR
            status = "overrun (clipped)"
        a = int(s["start"] * SR)
        b = min(len(buf), a + len(pcm))
        buf[a:b] += pcm[: b - a]
        s.update(text_final=text, voice=voice, tempo=round(tempo, 2), out_dur=round(dur, 2), status=status)
        if progress:
            progress(i + 1, len(segs), f"Dubbed {i + 1}/{len(segs)}")
    track = workdir / "dub_track.wav"
    pcm16 = (np.clip(buf, -1, 1) * 32767).astype(np.int16)
    with wave.open(str(track), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm16.tobytes())
    out = workdir / ("dubbed.mp4" if has_video else "dubbed.wav")
    mux(src, track, out, has_video, bed_volume, media_dur)
    return {"out": str(out), "track": str(track), "srt": make_srt(segs)}


def mux(src, track, out, has_video, bed_volume, max_dur):
    if not has_video:
        _run(["ffmpeg", "-y", "-v", "error", "-i", str(track), "-t", f"{max_dur:.3f}", str(out)])
        return
    try:
        from moviepy import AudioFileClip, CompositeAudioClip, VideoFileClip
    except ImportError:
        from moviepy.editor import AudioFileClip, CompositeAudioClip, VideoFileClip

    video = VideoFileClip(str(src))
    dubbed = AudioFileClip(str(track))
    clips_to_close = [video, dubbed]
    try:
        if hasattr(video, "subclipped"):
            video = video.subclipped(0, min(max_dur, video.duration))
            dubbed = dubbed.subclipped(0, min(max_dur, dubbed.duration))
            clips_to_close.extend([video, dubbed])
            if bed_volume > 0 and video.audio is not None:
                background = video.audio.subclipped(0, min(max_dur, video.audio.duration))
                background = background.with_volume_scaled(bed_volume)
                clips_to_close.append(background)
                audio = CompositeAudioClip([background, dubbed]).with_duration(max_dur)
            else:
                audio = dubbed.with_duration(max_dur)
            rendered = video.with_audio(audio)
        else:
            video = video.subclip(0, min(max_dur, video.duration))
            dubbed = dubbed.subclip(0, min(max_dur, dubbed.duration))
            clips_to_close.extend([video, dubbed])
            if bed_volume > 0 and video.audio is not None:
                background = video.audio.subclip(0, min(max_dur, video.audio.duration)).volumex(bed_volume)
                clips_to_close.append(background)
                audio = CompositeAudioClip([background, dubbed]).set_duration(max_dur)
            else:
                audio = dubbed.set_duration(max_dur)
            rendered = video.set_audio(audio)
        clips_to_close.append(rendered)
        clips_to_close.append(audio)
        rendered.write_videofile(
            str(out), codec="libx264", audio_codec="aac", temp_audiofile=str(Path(out).with_suffix(".m4a")),
            remove_temp=True, logger=None,
        )
    finally:
        for clip in reversed(clips_to_close):
            clip.close()


def _ts(t):
    ms = int(round(t * 1000))
    return f"{ms // 3600000:02d}:{ms // 60000 % 60:02d}:{ms // 1000 % 60:02d},{ms % 1000:03d}"


def make_srt(segs):
    """Subtitles carry the exact dubbed line, at the exact time the dub is spoken."""
    rows = []
    for i, s in enumerate(segs, 1):
        t = re.sub(r"<[^>]+>", "", s["text_final"]).strip()
        rows.append(f"{i}\n{_ts(s['start'])} --> {_ts(s['start'] + s['out_dur'])}\n{t}\n")
    return "\n".join(rows)