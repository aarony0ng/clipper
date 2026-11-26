"""Stage 3b: per-second signals that hint at a viral moment.

- loudness: RMS level in dBFS (shouting, laughter, hype)
- chat: chat-replay messages per second (YouTube live_chat only)
- scene: scene-cut count per second (optional; decodes the whole video at low res)
"""
from __future__ import annotations

import json
import re
import subprocess
import wave
from pathlib import Path

import numpy as np

from . import db
from .config import Settings


def compute(settings: Settings, source_id: str, scenes: bool = False, force: bool = False) -> Path:
    con = db.connect(settings.work_dir)
    src = db.get_source(con, source_id)
    sdir = Path(src["dir"])
    out = sdir / "signals.json"
    if out.exists() and not force:
        return out
    meta = json.loads((sdir / "meta.json").read_text())
    loud = loudness(sdir / "audio.wav")
    n = len(loud)
    chat = chat_velocity(sdir, meta.get("offset", 0.0), n)
    scene = scene_cuts(sdir / "source.mp4", n) if scenes else None
    out.write_text(json.dumps({
        "hop": 1.0,
        "loudness_db": [round(float(x), 2) for x in loud],
        "chat": chat.tolist() if chat is not None else None,
        "scene": scene.tolist() if scene is not None else None,
    }))
    print(f"[signals] {source_id}: {n}s loudness"
          f"{', chat' if chat is not None else ''}{', scenes' if scene is not None else ''}")
    return out


def loudness(wav: Path, hop: float = 1.0) -> np.ndarray:
    with wave.open(str(wav), "rb") as w:
        sr = w.getframerate()
        step = int(sr * hop)
        vals = []
        while True:
            buf = w.readframes(step)
            if not buf:
                break
            x = np.frombuffer(buf, dtype=np.int16).astype(np.float32) / 32768.0
            # Max of 4 sub-windows so a short shout is not averaged away.
            q = max(1, len(x) // 4)
            rms = max(float(np.sqrt(np.mean(x[i:i + q] ** 2) + 1e-12)) for i in range(0, len(x), q))
            vals.append(20 * np.log10(rms + 1e-9))
    return np.array(vals)


def chat_velocity(sdir: Path, offset: float, n: int) -> np.ndarray | None:
    files = list(sdir.glob("*.live_chat.json"))
    if not files:
        return None
    counts = np.zeros(n)
    with open(files[0], encoding="utf-8") as f:
        for line in f:
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            act = item.get("replayChatItemAction")
            if not act:
                continue
            t = int(act.get("videoOffsetTimeMsec", 0)) / 1000 - offset
            if 0 <= t < n:
                counts[int(t)] += len(act.get("actions", []))
    return counts


def scene_cuts(video: Path, n: int, threshold: float = 0.35) -> np.ndarray:
    proc = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(video), "-an", "-vf",
         f"fps=4,scale=192:-2,select='gt(scene,{threshold})',metadata=print:file=-",
         "-f", "null", "-"],
        capture_output=True, text=True,
    )
    counts = np.zeros(n)
    for m in re.finditer(r"pts_time:([\d.]+)", proc.stdout):
        t = float(m.group(1))
        if t < n:
            counts[int(t)] += 1
    return counts


def load(sdir: Path) -> dict:
    return json.loads((Path(sdir) / "signals.json").read_text())
