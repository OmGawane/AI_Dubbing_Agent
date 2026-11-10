"""Semantic localization agent: Evon as a film-localization scriptwriter."""
import json
import re

import gnani

# Comfortable spoken characters per second (native script, incl. spaces).
CPS = {"hi-IN": 13, "en-IN": 15, "ta-IN": 11, "te-IN": 11.5, "kn-IN": 11.5, "ml-IN": 10.5,
       "mr-IN": 12.5, "bn-IN": 12.5, "gu-IN": 12.5, "pa-IN": 12.5}

SYSTEM = """You are a senior film localization scriptwriter and dubbing director for Indian cinema and OTT.
You adapt dialogue so it sounds native in the target language. You never translate word for word.

For every input line:
- Preserve intent, emotion and register (sarcasm, humor, anger, tenderness, formality).
- Replace idioms, metaphors, slang and cultural references with natural target-language equivalents. Keep proper nouns.
- Keep speaker voice, gender and politeness level consistent across lines; use the surrounding lines as context.
- Preserve every provided speaker_id exactly.
- For lines marked "Unassigned", infer speaker turns conservatively from the dialogue and context. Assign stable
  labels such as "Speaker 1" and "Speaker 2", reusing labels from previous_lines; do not infer gender from dialogue.
- Do not create a new speaker unless the conversational context supports a distinct speaker.
- Give each speaker a concise voice_style (for example: warm, restrained, brisk, authoritative), consistent across lines.
  Infer style from dialogue and available speaker metadata only; do not infer gender from dialogue.
- Write in the native script of the target language, as people actually speak it. Write numbers, dates and currency as spoken words.
- Respect "max_chars" strictly: the dub must fit the time window of the original shot. Shorten by rephrasing, never by dropping meaning.
- Return exactly one output object per input id, including the original speaker_id.

Output ONLY a JSON array, no prose, no code fences:
[{"id": 0, "speaker_id": "Speaker 1", "literal": "<plain word-for-word translation>",
  "localized": "<natural dubbed line>", "tone": "<1-2 words>", "voice_style": "<1-3 words>",
  "note": "<=12 words on the cultural or idiom adaptation, or empty"}]"""


def plain(text):
    return re.sub(r"<[^>]+>", "", text or "").strip()


def max_chars(lang, seconds):
    return max(8, int(CPS.get(lang, 12) * seconds))


def _extract_json(s):
    s = re.sub(r"^```(?:json)?|```$", "", s.strip(), flags=re.M).strip()
    a, b = s.find("["), s.rfind("]")
    if a < 0 or b < a:
        raise ValueError("no JSON array in model output")
    return json.loads(s[a:b + 1])


def localize(segments, src_lang, tgt_lang, llm, scene="", emotion_tags=False, batch=20):
    """Fill literal/localized/tone/note on each segment. `llm` is a callable(messages)->str."""
    src, tgt = gnani.LANGS.get(src_lang, src_lang), gnani.LANGS[tgt_lang]
    tag_rule = ""
    if emotion_tags:
        tag_rule = ("\nYou may insert at most ONE audio tag in a line, only where the emotion genuinely calls for it, "
                    "from: " + " ".join(gnani.AUDIO_TAGS) + ". Tags do not count toward max_chars.")
    todo = [s for s in segments if s["text"]]
    prev = []
    for k in range(0, len(todo), batch):
        chunk = todo[k:k + batch]
        lines = [{
            "id": s["idx"],
            "speaker_id": s.get("speaker_id", "Speaker 1"),
            "speaker_id_observed": s.get("speaker_id_observed", True),
            "speaker_metadata": s.get("speaker_metadata", {}),
            "text": s["text"],
            "seconds": round(s["end"] - s["start"], 1),
            "max_chars": max_chars(tgt_lang, s["end"] - s["start"]),
        } for s in chunk]
        payload = {"source_language": src, "target_language": tgt, "scene_context": scene or "not provided",
                   "previous_lines": prev[-3:], "lines": lines}
        msgs = [{"role": "system", "content": SYSTEM + tag_rule},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}]
        out = None
        for attempt in range(2):
            try:
                out = _extract_json(llm(msgs))
                break
            except Exception:
                if attempt == 1:
                    raise
        by_id = {int(o["id"]): o for o in out}
        for s in chunk:
            o = by_id.get(s["idx"])
            if not o or not str(o.get("localized", "")).strip():
                raise ValueError(f"Evon returned no localized line for segment {s['idx']}")
            if not s.get("speaker_id_observed", True):
                inferred_speaker = str(o.get("speaker_id", "")).strip()
                s["speaker_id"] = (
                    inferred_speaker if inferred_speaker and inferred_speaker != "Unassigned"
                    else "Speaker 1"
                )
            s["literal"] = str(o.get("literal", "")).strip()
            s["localized"] = str(o["localized"]).strip()
            s["tone"] = str(o.get("tone", "")).strip()
            s["voice_style"] = str(o.get("voice_style", "")).strip()
            s["note"] = str(o.get("note", "")).strip()
            prev.append({"speaker_id": s["speaker_id"], "source": s["text"],
                         "localized": plain(s["localized"]), "voice_style": s["voice_style"]})
    return segments


def condense(llm, text, target_chars, tgt_lang):
    """Rewrite one dubbed line shorter while keeping meaning and tone."""
    msgs = [
        {"role": "system", "content": "You are a dubbing scriptwriter. Rewrite the line shorter so it fits the shot, "
                                      "keeping meaning, emotion and register, in the same language and script. "
                                      "Output only the rewritten line."},
        {"role": "user", "content": f"Language: {gnani.LANGS[tgt_lang]}\nMaximum characters: {target_chars}\nLine: {text}"},
    ]
    out = llm(msgs).strip().strip('"').splitlines()[0].strip()
    return out or text