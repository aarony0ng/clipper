"""Stage 3a: word-level transcript.

Default engine is faster-whisper (a Whisper reimplementation that runs well on
CPU). If the VOD came with YouTube auto-captions, `engine="youtube"` uses
those instead: lower quality, but free and instant.
"""
from __future__ import annotations

import json
from pathlib import Path

from . import db
from .config import Settings


def transcribe(settings: Settings, source_id: str, engine: str = "whisper", force: bool = False) -> Path:
    con = db.connect(settings.work_dir)
    src = db.get_source(con, source_id)
    sdir = Path(src["dir"])
    out = sdir / "transcript.json"
    if out.exists() and not force:
        print(f"[transcribe] {source_id}: already done")
        return out

    meta = json.loads((sdir / "meta.json").read_text())
    if engine == "youtube":
        segments = _from_youtube_json3(sdir, meta.get("offset", 0.0), src["duration"])
        label = "youtube-auto-captions"
    else:
        segments = _whisper(sdir / "audio.wav", settings)
        label = f"faster-whisper:{settings.whisper_model}"

    out.write_text(json.dumps({"engine": label, "segments": segments}, ensure_ascii=False))
    db.set_status(con, "source", source_id, "transcribed")
    n_words = sum(len(s["words"]) for s in segments)
    print(f"[transcribe] {source_id}: {len(segments)} segments, {n_words} words ({label})")
    return out


def _whisper(wav: Path, settings: Settings) -> list[dict]:
    from faster_whisper import WhisperModel

    device = settings.whisper_device
    if device == "auto":
        try:
            import ctranslate2

            device = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
        except Exception:
            device = "cpu"
    compute = "float16" if device == "cuda" else "int8"
    model = WhisperModel(settings.whisper_model, device=device, compute_type=compute)
    # Decode the 16 kHz mono wav ourselves: faster-whisper 1.2.x passes an argument
    # to PyAV that PyAV 19 removed, so handing it a path crashes.
    import wave
    import numpy as np
    with wave.open(str(wav), "rb") as wf:
        pcm = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)
    audio = pcm.astype(np.float32) / 32768.0
    seg_iter, info = model.transcribe(
        audio, word_timestamps=True, vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 500}, beam_size=5,
    )
    print(f"[transcribe] language={info.language} p={info.language_probability:.2f} "
          f"audio={info.duration:.0f}s on {device}")
    segments = []
    for seg in seg_iter:
        words = [{"s": round(w.start, 3), "e": round(w.end, 3), "w": w.word.strip()}
                 for w in (seg.words or []) if w.word.strip()]
        if words:
            segments.append({"s": words[0]["s"], "e": words[-1]["e"], "text": seg.text.strip(), "words": words})
            if len(segments) % 200 == 0:
                print(f"[transcribe] ... {seg.end:.0f}s")
    return segments


def _from_youtube_json3(sdir: Path, offset: float, duration: float | None) -> list[dict]:
    files = sorted(p for p in sdir.glob("source.*.json3") if "live_chat" not in p.name)
    if not files:
        raise FileNotFoundError("No YouTube captions were downloaded for this source; use --engine whisper.")
    pref = [p for p in files if ".en-orig." in p.name] or [p for p in files if ".en." in p.name] or files
    data = json.loads(pref[0].read_text())
    segments = []
    for ev in data.get("events", []):
        segs = [s for s in ev.get("segs", []) or [] if s.get("utf8", "").strip()]
        if not segs:
            continue
        t0 = ev["tStartMs"] / 1000 - offset
        end = t0 + ev.get("dDurationMs", 0) / 1000
        words = []
        for i, s in enumerate(segs):
            ws = t0 + s.get("tOffsetMs", 0) / 1000
            we = t0 + segs[i + 1].get("tOffsetMs", 0) / 1000 if i + 1 < len(segs) else end
            words.append({"s": round(ws, 3), "e": round(max(we, ws + 0.05), 3), "w": s["utf8"].strip()})
        words = [w for w in words if w["s"] >= 0 and (duration is None or w["e"] <= duration)]
        if words:
            segments.append({"s": words[0]["s"], "e": words[-1]["e"],
                             "text": " ".join(w["w"] for w in words), "words": words})
    return segments


def load(sdir: Path) -> list[dict]:
    return json.loads((Path(sdir) / "transcript.json").read_text())["segments"]
