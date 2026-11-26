"""Multi-POV moment scan: find the most engaging moments across a group stream.

When several creators stream the same session, each VOD is one point of view
(POV). This pass reads all of them at low resolution and lines them up on
wall-clock time:

1. transcribe each POV's eligible window (Whisper `base`, enough for timing
   and hype words);
2. per second and per POV: loudness, sudden loudness jumps, words per
   second, hype words, laughter and exclamations;
3. viewer clips (the strongest signal: people chose to clip it) are aligned
   to the exact second by matching their audio against the POV's audio;
4. a moment scores higher when several POVs spike together (everyone reacts),
   and each moment records the POV with the strongest reaction;
5. peaks become a shortlist of windows with every POV's transcript, written to
   work/scan/<name>/shortlist.md for picking (by Claude or by hand), then
   rendered from full-resolution sections of the chosen POV.

Config is a JSON file:
{"name": "...", "campaign": "...", "povs": [{"name": "sypherpk", "vod": "2929935152",
  "vod_start_ts": 1794774338, "start": 2750, "end": 9830,
  "video": "work/scan/sypherpk.mp4", "wav": "work/scan/sypherpk.wav"}],
 "viewer_clips": "work/scan/vod_clips.json"}
"""
from __future__ import annotations

import dataclasses
import json
import re
import subprocess
import wave
from pathlib import Path

import numpy as np

from .config import Settings
from .moments import HYPE, LAUGH, _robust_z, _smooth

EXCLAIM = re.compile(r"!")


def _load_wav(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as w:
        sr = w.getframerate()
        x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
    return x, sr


def _envelope(x: np.ndarray, sr: int, hop: float) -> np.ndarray:
    n = int(sr * hop)
    m = len(x) // n
    frames = x[: m * n].reshape(m, n)
    return 20 * np.log10(np.sqrt((frames ** 2).mean(axis=1)) + 1e-9)


def transcript(settings: Settings, wav: Path, out: Path, model: str = "base") -> list[dict]:
    """mlx-whisper (large-v3-turbo) on Apple Silicon; else faster-whisper `model` for speed."""
    if out.exists():
        return json.loads(out.read_text())
    from . import transcribe
    segs, label = transcribe.asr(wav, dataclasses.replace(settings, whisper_model=model))
    out.write_text(json.dumps(segs))
    return segs


def per_second(segs: list[dict], wav: Path) -> dict[str, np.ndarray]:
    x, sr = _load_wav(wav)
    # loudness: max of 4 sub-windows per second so a short shout isn't averaged away
    sub = _envelope(x, sr, 0.25)
    n = len(sub) // 4
    loud = sub[: n * 4].reshape(n, 4).max(axis=1)
    words, hype, laugh, excl = (np.zeros(n) for _ in range(4))
    for s in segs:
        t = int(s["s"])
        if 0 <= t < n:
            hype[t] += len(HYPE.findall(s["text"]))
            laugh[t] += len(LAUGH.findall(s["text"]))
            excl[t] += len(EXCLAIM.findall(s["text"]))
        for w in s["words"]:
            if 0 <= int(w["s"]) < n:
                words[int(w["s"])] += 1
    lz = _robust_z(_smooth(loud, 2))
    # sudden rise above the recent level: reactions, not constant loud music
    base = np.array([np.median(lz[max(0, i - 20):i]) if i else lz[0] for i in range(n)])
    jump = lz - base
    return {"loud_z": lz, "jump": jump, "words": _smooth(words, 3), "hype": _smooth(hype, 5),
            "laugh": _smooth(laugh, 5), "excl": _smooth(excl, 5)}


def align_clip(clip_wav: Path, pov_wav: Path, pov_start: float, guess_end: float,
               search: tuple[float, float] = (-150, 15)) -> tuple[float, float]:
    """Where a viewer clip sits in the POV (VOD seconds) and how confident (0-1)."""
    c, sr = _load_wav(clip_wav)
    p, sr2 = _load_wav(pov_wav)
    assert sr == sr2
    hop = 0.05
    ce = _envelope(c, sr, hop)
    a = max(0.0, guess_end - pov_start + search[0])
    b = min(len(p) / sr, guess_end - pov_start + search[1])
    pe = _envelope(p[int(a * sr):int(b * sr)], sr, hop)
    if len(pe) <= len(ce):
        return -1.0, 0.0
    ce = (ce - ce.mean()) / (ce.std() + 1e-6)
    best, best_i = -9.0, 0
    for i in range(len(pe) - len(ce)):
        seg = pe[i:i + len(ce)]
        r = float(np.dot(ce, (seg - seg.mean()) / (seg.std() + 1e-6)) / len(ce))
        if r > best:
            best, best_i = r, i
    return pov_start + a + best_i * hop, best


def fetch_clip_audio(url: str, out: Path) -> Path | None:
    if out.exists():
        return out
    tmp = out.with_suffix(".mp4")
    r = subprocess.run(["yt-dlp", "-q", "--no-warnings", "-f", "worst", "-o", str(tmp), url], capture_output=True)
    if r.returncode or not tmp.exists():
        return None
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(tmp), "-vn", "-ac", "1", "-ar", "16000", str(out)],
                   check=True)
    tmp.unlink(missing_ok=True)
    return out


def run(settings: Settings, cfg_path: Path, top: int = 24) -> Path:
    cfg = json.loads(cfg_path.read_text())
    root = Path(__file__).resolve().parents[1]  # repo root; config paths are relative to it
    out_dir = settings.work_dir / "scan" / cfg["name"]
    out_dir.mkdir(parents=True, exist_ok=True)
    povs = cfg["povs"]
    w0 = min(p["vod_start_ts"] + p["start"] for p in povs)
    w1 = max(p["vod_start_ts"] + p["end"] for p in povs)
    N = int(w1 - w0)

    feats, segs_by = {}, {}
    for p in povs:
        wav = root / p["wav"]
        segs = transcript(settings, wav, out_dir / f"{p['name']}.transcript.json")
        segs_by[p["name"]] = segs
        f = per_second(segs, wav)
        shift = int(p["vod_start_ts"] + p["start"] - w0)  # wall-clock index of this POV's second 0
        g = {}
        for k, v in f.items():
            arr = np.full(N, np.nan)
            m = min(len(v), N - shift)
            arr[shift:shift + m] = v[:m]
            g[k] = arr
        feats[p["name"]] = g
        print(f"[scan] {p['name']}: {len(segs)} segments, {len(f['loud_z'])}s")

    names = [p["name"] for p in povs]
    jump = np.vstack([np.nan_to_num(feats[n]["jump"], nan=0) for n in names])
    react = np.clip(jump, 0, 6) + 1.5 * np.vstack([np.nan_to_num(feats[n]["laugh"]) for n in names]) \
        + 0.6 * np.vstack([np.nan_to_num(feats[n]["hype"]) for n in names]) \
        + 0.4 * np.vstack([np.nan_to_num(feats[n]["excl"]) for n in names])
    srt = np.sort(react, axis=0)[::-1]
    together = srt[:2].mean(axis=0) if len(names) > 1 else srt[0]   # several POVs at once
    talk = np.nanmean(np.vstack([feats[n]["words"] for n in names]), axis=0)
    score = 1.2 * together + 0.6 * srt[0] + 0.3 * np.clip(np.nan_to_num(talk) - 2, 0, 3)

    # viewer clips: align each to its POV and add a strong boost over the clipped span
    clips = []
    vc_path = cfg.get("viewer_clips")
    if vc_path and (root / vc_path).exists():
        byname = {p["name"]: p for p in povs}
        for c in json.loads((root / vc_path).read_text()):
            p = byname.get(c["channel"])
            if not p or not c.get("eligible"):
                continue
            aud = fetch_clip_audio(c["url"], out_dir / f"clip_{c['id']}.wav")
            if not aud:
                print(f"[scan] could not fetch viewer clip {c['url']}")
                continue
            at, conf = align_clip(aud, root / p["wav"], p["start"], c["created_off"])
            if conf < 0.4:  # fall back to "it ends when it was created"
                at, conf = c["created_off"] - c["dur"], conf
            wall = p["vod_start_ts"] + at - w0
            a, b = int(max(0, wall)), int(min(N, wall + c["dur"]))
            score[a:b] += 3.0 + np.log10(max(10, c["views"]))
            clips.append({**c, "vod_start": round(at, 1), "align_conf": round(conf, 2), "wall": a})
            print(f"[scan] viewer clip {c['title']!r} ({c['views']} views): {c['channel']} "
                  f"{at:.1f}s in VOD (match {conf:.2f})")

    sm = _smooth(np.nan_to_num(score), 5)
    order = np.argsort(sm)[::-1]
    peaks = []
    for i in order:
        if all(abs(i - j) > 45 for j in peaks):
            peaks.append(int(i))
        if len(peaks) >= top:
            break

    def pov_time(p, wall_i):
        return wall_i + w0 - p["vod_start_ts"]

    rows, md = [], [f"# Shortlist: {cfg['name']}", "",
                    f"{len(names)} POVs aligned on wall-clock time, {N}s scanned. Top {len(peaks)} peaks "
                    f"(about {len(peaks) * 100 / max(1, N / 30):.1f}% of 30-s windows). Times are VOD seconds per POV.", ""]
    for rank, i in enumerate(peaks, 1):
        per = {n: float(react[k, max(0, i - 3):i + 4].max()) for k, n in enumerate(names)}
        best = max(per, key=per.get)
        spiking = [n for n in names if per[n] > 1.5]
        vc = [c for c in clips if c["wall"] - 10 <= i <= c["wall"] + c["dur"] + 10]
        row = {"rank": rank, "wall": i, "score": round(float(sm[i]), 2), "best_pov": best,
               "pov_react": {k: round(v, 2) for k, v in per.items()}, "spiking": spiking,
               "viewer_clips": [{"title": c["title"], "views": c["views"]} for c in vc],
               "pov_seconds": {p["name"]: round(pov_time(p, i), 1) for p in povs}}
        rows.append(row)
        md += [f"## #{rank} score {row['score']} · best POV {best} · spiking: {', '.join(spiking) or '-'}"
               + (f" · viewer clips: {', '.join(repr(c['title']) + ' ' + str(c['views']) for c in vc)}" if vc else ""), ""]
        for p in povs:
            t = pov_time(p, i)
            md.append(f"**{p['name']}** (peak at {t:.0f}s)")
            for s in segs_by[p["name"]]:
                vt = s["s"] + p["start"]
                if t - 40 <= vt <= t + 15:
                    md.append(f"- [{vt:.1f}] {s['text']}")
            md.append("")
    (out_dir / "shortlist.json").write_text(json.dumps({"w0": w0, "povs": povs, "peaks": rows,
                                                        "viewer_clips": clips}, indent=1))
    (out_dir / "shortlist.md").write_text("\n".join(md))
    np.save(out_dir / "score.npy", sm)
    print(f"[scan] {out_dir / 'shortlist.md'}")
    return out_dir / "shortlist.md"
