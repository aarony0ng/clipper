"""Fast-cut editing: jump cuts over dead air and punch-in zooms.

The top short-form clips cut about every 3 s and never sit on a pause. A
stream clip is one continuous take, so the editor fakes that rhythm:

- jump cuts: quiet pauses between words longer than `max_gap_s` are cut
  down to `keep_gap_s`. Loud pauses (laughter, game sounds) are kept, since
  they are usually the reaction or the payoff.
- punch-in zooms: the kept footage is split into pieces of about
  `zoom_every_s` at word boundaries, and the zoom alternates between 1.0 and
  `zoom_amount` on every cut. Pieces that are much louder than the rest of
  the clip (shouts, hype) get the stronger `reaction_zoom`.

The plan is a list of pieces in clip time (0 = moment start). Captions are
remapped onto the edited timeline with `remap`.
"""
from __future__ import annotations

import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .style import EditStyle


@dataclass
class Piece:
    s: float      # clip time (seconds from the moment's start)
    e: float
    zoom: float = 1.0


def _db(x: np.ndarray) -> float:
    return 20 * np.log10(float(np.sqrt(np.mean(x ** 2))) + 1e-9) if x.size else -90.0


def _audio(wav: Path, start: float, end: float) -> tuple[np.ndarray, int]:
    with wave.open(str(wav), "rb") as w:
        sr = w.getframerate()
        w.setpos(min(w.getnframes(), max(0, int(start * sr))))
        buf = w.readframes(max(0, int((end - start) * sr)))
    return np.frombuffer(buf, dtype=np.int16).astype(np.float32) / 32768.0, sr


def plan(words: list[dict], clip_len: float, wav: Path | None, start: float, st: EditStyle,
         min_len: float = 0.0) -> list[Piece]:
    """Pieces to keep, with a zoom for each. `words` are in clip time."""
    if not st.enabled:
        return [Piece(0.0, clip_len)]
    audio, sr = (_audio(wav, start, start + clip_len) if wav and wav.exists() else (np.zeros(0), 16000))

    def level(a: float, b: float) -> float:
        return _db(audio[int(a * sr):int(b * sr)])

    speech_db = np.median([level(w["s"], w["e"]) for w in words]) if (words and audio.size) else -20.0

    # 1. jump cuts over quiet pauses
    keep = [[0.0, clip_len]]
    removed = 0.0
    if st.jump_cuts and words:
        cuts = []
        for a, b in zip(words, words[1:]):
            gap = b["s"] - a["e"]
            if gap <= st.max_gap_s:
                continue
            if audio.size and level(a["e"], b["s"]) > speech_db - st.quiet_db:
                continue  # something is happening in the pause; keep it
            cuts.append((a["e"] + st.keep_gap_s / 2, b["s"] - st.keep_gap_s / 2))
        budget = max(0.0, min(clip_len * st.max_cut_share, clip_len - min_len))
        for a, b in sorted(cuts, key=lambda c: c[1] - c[0], reverse=True):  # longest pauses first
            if removed + (b - a) > budget:
                continue
            removed += b - a
            new = []
            for s, e in keep:
                if b <= s or a >= e:
                    new.append([s, e])
                    continue
                if a > s:
                    new.append([s, a])
                if b < e:
                    new.append([b, e])
            keep = new
    keep = [k for k in keep if k[1] - k[0] > 0.05]

    # 2. split long stretches at word gaps so there's a cut about every zoom_every_s
    pieces: list[Piece] = []
    bounds = sorted({round(w["s"] - 0.02, 3) for w in words})
    for s, e in keep:
        t = s
        while e - t > st.zoom_every_s * 1.5 and st.zoom_every_s > 0:
            target = t + st.zoom_every_s
            cands = [x for x in bounds if t + st.zoom_every_s * 0.6 < x < min(e - 1.0, t + st.zoom_every_s * 1.5)]
            # Prefer a word boundary; inside one long run of speech, cut on time.
            cut = min(cands, key=lambda x: abs(x - target)) if cands else target
            pieces.append(Piece(t, cut))
            t = cut
        pieces.append(Piece(t, e))

    # 3. zoom levels: alternate on every cut; loud pieces punch in harder
    levels = [level(p.s, p.e) for p in pieces] if audio.size else [0.0] * len(pieces)
    loud_line = np.percentile(levels, 80) if len(levels) >= 5 else max(levels) + 1
    z = False
    for p, lv in zip(pieces, levels):
        if st.zoom_every_s <= 0:
            p.zoom = 1.0
            continue
        if audio.size and lv >= loud_line and lv > speech_db + st.reaction_db:
            p.zoom = st.reaction_zoom
            z = True  # the next piece eases back out
            continue
        if p is pieces[0]:
            z = False
        elif p.e - p.s >= 0.8:  # very short pieces keep the zoom, so it doesn't flicker
            z = not z
        p.zoom = st.zoom_amount if z else 1.0
    return pieces


def length(pieces: list[Piece]) -> float:
    return sum(p.e - p.s for p in pieces)


def remap(words: list[dict], pieces: list[Piece]) -> list[dict]:
    """Move word timings onto the edited timeline; words inside a cut are dropped."""
    out, offset = [], 0.0
    for p in pieces:
        for w in words:
            if w["s"] >= p.s - 0.01 and w["s"] < p.e:
                out.append({**w, "s": w["s"] - p.s + offset, "e": min(w["e"], p.e) - p.s + offset})
        offset += p.e - p.s
    return out


def filter_graph(pieces: list[Piece], W: int, H: int) -> str:
    """ffmpeg filters taking [base] and [0:a], producing [cut] and [cuta]."""
    n = len(pieces)
    parts = [f"[base]split={n}" + "".join(f"[vb{i}]" for i in range(n)),
             f"[0:a]asplit={n}" + "".join(f"[ab{i}]" for i in range(n))]
    for i, p in enumerate(pieces):
        v = f"[vb{i}]trim=start={p.s:.3f}:end={p.e:.3f},setpts=PTS-STARTPTS"
        if p.zoom > 1.001:
            zw, zh = int(round(W * p.zoom / 2)) * 2, int(round(H * p.zoom / 2)) * 2
            v += f",scale={zw}:{zh},crop={W}:{H}:{(zw - W) // 2}:{(zh - H) // 2}"
        parts.append(v + f",setsar=1[v{i}]")
        parts.append(f"[ab{i}]atrim=start={p.s:.3f}:end={p.e:.3f},asetpts=PTS-STARTPTS,"
                     f"afade=t=in:d=0.01,afade=t=out:st={max(0.0, p.e - p.s - 0.01):.3f}:d=0.01[a{i}]")
    parts.append("".join(f"[v{i}][a{i}]" for i in range(n)) + f"concat=n={n}:v=1:a=1[cut][cuta]")
    return ";".join(parts)
