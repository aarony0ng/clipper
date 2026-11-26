"""Text must always be fully legible. Hard rules, enforced on every render:

1. Our own text (captions, hook, subcaption, handle, #ad) is burned in after
   any zoom and must sit fully inside the safe area. Each event is rendered
   by libass itself (so the real font and its fallback are measured, outline
   included) and shrunk with \\fs until it fits.
2. Text already in the picture (game UI, scoreboards, banners, alerts) must
   never be cut by a zoom. Text is detected in sampled frames of each piece
   (white-on-outline UI text plus Tesseract words), and each panel's zoom
   and crop are chosen so every detected box stays inside with a margin.
   A box already touching the panel edge keeps that panel at 1.0x.
3. QA re-checks the output: text that touches an edge in the output but not
   in the unzoomed frame fails the clip, and the renderer retries with less zoom.
"""
from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

W, H = 1080, 1920
SAFE_X = 90                 # side margins
SAFE_TOP, SAFE_BOTTOM = 150, H - 330   # platform UI (top bar; caption/buttons zone at the bottom)
EDGE = 4                    # a box this close to an edge counts as touching it
MARGIN = 24                 # keep detected text at least this far inside after a zoom


# ---------------------------------------------------------------- frames

def base_frames(video: Path, start: float, layout_filter: str, times: list[float]) -> dict[float, np.ndarray]:
    """Frames of the reframed (unzoomed, uncaptioned) composite at clip times."""
    import cv2
    out = {}
    for t in times:
        p = subprocess.run(
            ["ffmpeg", "-v", "error", "-ss", f"{start + t:.3f}", "-i", str(video), "-frames:v", "1",
             "-filter_complex", layout_filter, "-map", "[base]", "-f", "image2pipe", "-vcodec", "png", "-"],
            capture_output=True)
        if p.stdout:
            img = cv2.imdecode(np.frombuffer(p.stdout, np.uint8), cv2.IMREAD_COLOR)
            if img is not None:
                out[t] = img
    return out


def output_frame(clip: Path, t: float) -> np.ndarray | None:
    import cv2
    p = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{t:.3f}", "-i", str(clip), "-frames:v", "1",
                        "-f", "image2pipe", "-vcodec", "png", "-"], capture_output=True)
    return cv2.imdecode(np.frombuffer(p.stdout, np.uint8), cv2.IMREAD_COLOR) if p.stdout else None


# ---------------------------------------------------------------- text detection

def _ui_text(img: np.ndarray) -> list[list[int]]:
    """Bright, low-saturation glyphs with a dark outline: how game UI and overlays draw text."""
    import cv2
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    white = ((hsv[..., 2] > 215) & (hsv[..., 1] < 45)).astype(np.uint8)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dark = (cv2.dilate((gray < 70).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0)
    # glyph pixels must sit next to an outline/shadow (dark) somewhere close by
    edge = cv2.dilate(white, np.ones((7, 7), np.uint8)) & dark.astype(np.uint8)
    keep = white & (cv2.dilate(edge, np.ones((9, 9), np.uint8)) > 0)
    words = cv2.morphologyEx(keep, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (17, 5)))
    n, _, stats, _ = cv2.connectedComponentsWithStats(words, 8)
    boxes = []
    for x, y, w, h, area in stats[1:]:
        if not (10 <= h <= 160 and w >= 12 and w <= 900):
            continue
        fill = keep[y:y + h, x:x + w].mean()
        if 0.12 <= fill <= 0.7 and w / h >= 0.6:
            boxes.append([int(x), int(y), int(x + w), int(y + h)])
    return boxes


def _tesseract(img: np.ndarray) -> list[list[int]]:
    exe = shutil.which("tesseract")
    if not exe:
        return []
    import cv2
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "f.png"
        cv2.imwrite(str(p), img)
        r = subprocess.run([exe, str(p), "stdout", "--psm", "11", "tsv"], capture_output=True, text=True)
    boxes = []
    for line in r.stdout.splitlines()[1:]:
        f = line.split("\t")
        if len(f) < 12 or not f[11].strip():
            continue
        try:
            conf = float(f[10])
        except ValueError:
            continue
        x, y, w, h = map(int, f[6:10])
        if conf >= 60 and len(re.sub(r"\W", "", f[11])) >= 2 and 8 <= h <= 200:
            boxes.append([x, y, x + w, y + h])
    return boxes


_VISION_SRC = Path(__file__).with_name("vision_text.swift")
_VISION_BIN = Path(__file__).resolve().parents[1] / ".bin" / "vision_text"


def _vision_bin() -> Path | None:
    """Apple Vision text recognizer (macOS), compiled once from vision_text.swift."""
    if _VISION_BIN.exists() and _VISION_BIN.stat().st_mtime >= _VISION_SRC.stat().st_mtime:
        return _VISION_BIN
    if not shutil.which("swiftc"):
        return None
    _VISION_BIN.parent.mkdir(exist_ok=True)
    r = subprocess.run(["swiftc", "-O", str(_VISION_SRC), "-o", str(_VISION_BIN)], capture_output=True)
    return _VISION_BIN if r.returncode == 0 else None


def detect_many(imgs: list[np.ndarray], min_conf: float = 0.4) -> list[list[list[int]]]:
    """Text boxes per image. Apple Vision when available (reads stylised game UI well);
    otherwise Tesseract plus a white-glyph detector."""
    import cv2
    exe = _vision_bin()
    if exe is None:
        return [_merge(_ui_text(i) + _tesseract(i)) for i in imgs]
    out = [[] for _ in imgs]
    with tempfile.TemporaryDirectory() as d:
        paths = []
        for k, img in enumerate(imgs):
            p = Path(d) / f"{k}.png"
            cv2.imwrite(str(p), img)
            paths.append(str(p))
        r = subprocess.run([str(exe), *paths], capture_output=True, text=True)
    for line in r.stdout.splitlines():
        f = line.split(" ", 6)
        if len(f) < 6:
            continue
        k, x0, y0, x1, y1, conf = int(f[0]), *map(int, f[1:5]), float(f[5])
        text = f[6] if len(f) > 6 else ""
        if conf >= min_conf and len(text.strip()) >= 1:
            out[k].append([x0, y0, x1, y1])
    return [_merge(b) for b in out]


def detect_text(img: np.ndarray) -> list[list[int]]:
    return detect_many([img])[0]


def _merge(boxes: list[list[int]], pad: int = 6) -> list[list[int]]:
    boxes = [b[:] for b in boxes]
    changed = True
    while changed:
        changed = False
        out = []
        for b in boxes:
            for o in out:
                if b[0] <= o[2] + pad and o[0] <= b[2] + pad and b[1] <= o[3] + pad and o[1] <= b[3] + pad:
                    o[:] = [min(o[0], b[0]), min(o[1], b[1]), max(o[2], b[2]), max(o[3], b[3])]
                    changed = True
                    break
            else:
                out.append(b)
        boxes = out
    return boxes


def source_text(video: Path, times: list[float]) -> list[list[int]]:
    """Text boxes in the source video (source pixels) at the given times."""
    import cv2
    imgs = []
    for t in times:
        p = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{t:.3f}", "-i", str(video), "-frames:v", "1",
                            "-f", "image2pipe", "-vcodec", "png", "-"], capture_output=True)
        if p.stdout:
            img = cv2.imdecode(np.frombuffer(p.stdout, np.uint8), cv2.IMREAD_COLOR)
            if img is not None:
                imgs.append(img)
    return [b for bs in detect_many(imgs) for b in bs] if imgs else []


def clear_crop_x(boxes: list[list[int]], x: int, w: int, y: int, h: int, lo: int, hi: int,
                 pad: int = 8) -> int:
    """Nearest crop left edge in [lo, hi] whose sides don't cut through any text box."""
    rows = [b for b in boxes if b[3] > y and b[1] < y + h]

    def cuts(cx):
        return sum(1 for b in rows if (b[0] - pad < cx < b[2] + pad) or (b[0] - pad < cx + w < b[2] + pad))

    if not rows or not cuts(x) or hi < lo:
        return x
    best = min(range(lo, hi + 1, 2), key=lambda cx: (cuts(cx), abs(cx - x)))
    return best


# ---------------------------------------------------------------- zoom solver

def panel_boxes(boxes: list[list[int]], y0: int, y1: int) -> list[list[int]]:
    """Boxes inside the panel [y0, y1), in panel coordinates (clipped to the panel)."""
    out = []
    for x0, a, x1, b in boxes:
        if b <= y0 or a >= y1:
            continue
        out.append([x0, max(a, y0) - y0, x1, min(b, y1) - y0])
    return out


def solve_zoom(boxes: list[list[int]], pw: int, ph: int, want: float) -> tuple[float, int, int, int, int]:
    """Largest zoom <= want whose crop keeps every box fully inside (with MARGIN).
    Returns (zoom, crop x, crop y, crop w, crop h) in panel pixels."""
    full = (1.0, 0, 0, pw, ph)
    if want <= 1.001:
        return full
    if any(b[0] <= EDGE or b[1] <= EDGE or b[2] >= pw - EDGE or b[3] >= ph - EDGE for b in boxes):
        return full  # text already at the edge: any zoom would cut it further
    if boxes:
        ux0, uy0 = min(b[0] for b in boxes) - MARGIN, min(b[1] for b in boxes) - MARGIN
        ux1, uy1 = max(b[2] for b in boxes) + MARGIN, max(b[3] for b in boxes) + MARGIN
    z = want
    while z > 1.03:
        cw, ch = int(pw / z) // 2 * 2, int(ph / z) // 2 * 2
        if not boxes:
            return z, (pw - cw) // 2, (ph - ch) // 2, cw, ch
        if ux1 - ux0 <= cw and uy1 - uy0 <= ch:
            # centre on the panel, then shift just enough to contain the text
            x = int(np.clip((pw - cw) / 2, ux1 - cw, ux0))
            y = int(np.clip((ph - ch) / 2, uy1 - ch, uy0))
            x, y = int(np.clip(x, 0, pw - cw)), int(np.clip(y, 0, ph - ch))
            if x <= ux0 and x + cw >= ux1 and y <= uy0 and y + ch >= uy1:
                return z, x, y, cw, ch
        z -= 0.02
    return full


# ---------------------------------------------------------------- our own overlay text

_DLG = re.compile(r"^(Dialogue: \d+,)([^,]+),([^,]+),(\w+),(.*?,.*?,.*?,.*?,.*?,)(.*)$")


def overlay_boxes(ass_path: Path) -> list[dict]:
    """Render every overlay event alone with libass and measure its pixel bounding box."""
    text = ass_path.read_text()
    head, _, body = text.partition("[Events]")
    fmt, *lines = body.strip().splitlines()
    events = [l for l in lines if l.startswith("Dialogue:")]
    if not events:
        return []
    out = []
    for start in range(0, len(events), 60):  # ASS seconds field in this test is 2 digits; batch by 60
        chunk = events[start:start + 60]
        tl = []
        for i, l in enumerate(chunk):
            m = _DLG.match(l)
            # shown from i-0.5 s so the frame at t=i is past any pop-in animation
            tl.append(f"{m.group(1)}0:00:{max(0.0, i - 0.5):05.2f},0:00:{i + 0.8:05.2f},"
                      f"{m.group(4)},{m.group(5)}{m.group(6)}")
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "t.ass"
            p.write_text(head + "[Events]\n" + fmt + "\n" + "\n".join(tl) + "\n")
            r = subprocess.run(
                ["ffmpeg", "-v", "error", "-f", "lavfi", "-i", f"color=c=0x808080:s={W}x{H}:r=1:d={len(chunk)}",
                 "-vf", f"ass={p.name}", "-f", "rawvideo", "-pix_fmt", "gray", "-"],
                capture_output=True, cwd=d)
        frames = np.frombuffer(r.stdout, np.uint8)
        n = len(frames) // (W * H)
        frames = frames[: n * W * H].reshape(n, H, W)
        for i, l in enumerate(chunk):
            m = _DLG.match(l)
            box = None
            if i < n:
                ys, xs = np.nonzero(np.abs(frames[i].astype(int) - 128) > 12)
                if len(xs):
                    box = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]
            out.append({"line": l, "style": m.group(4), "box": box})
    return out


def fit_overlay(ass_path: Path, font_sizes: dict[str, int], rounds: int = 4) -> dict:
    """Shrink any overlay event that leaves the safe area; verify by re-measuring."""
    fixed = 0
    for _ in range(rounds):
        meas = overlay_boxes(ass_path)
        bad = [e for e in meas if e["box"] and not _inside_safe(e["box"], e["style"])]
        if not bad:
            return {"overlay_ok": True, "overlay_shrunk": fixed, "overlay_events": len(meas)}
        text = ass_path.read_text()
        for e in bad:
            x0, y0, x1, y1 = e["box"]
            scale = min(1.0, (W - 2 * SAFE_X) / max(1, x1 - x0)) * 0.97
            cur = re.search(r"\\fs(\d+)", e["line"])
            size = int(cur.group(1)) if cur else font_sizes.get(e["style"], 80)
            new = max(28, int(size * scale)) if scale < 1 else max(28, int(size * 0.92))
            m = _DLG.match(e["line"])
            body = re.sub(r"\\fs\d+", "", m.group(6))
            body = "{\\fs" + str(new) + "}" + body if not body.startswith("{") else \
                "{\\fs" + str(new) + body[1:]
            text = text.replace(e["line"], m.group(1) + m.group(2) + "," + m.group(3) + "," + m.group(4) + ","
                                + m.group(5) + body, 1)
            fixed += 1
        ass_path.write_text(text)
    meas = overlay_boxes(ass_path)
    bad = [e for e in meas if e["box"] and not _inside_safe(e["box"], e["style"])]
    return {"overlay_ok": not bad, "overlay_shrunk": fixed, "overlay_events": len(meas),
            "overlay_bad": [e["line"][-60:] for e in bad][:5]}


def _inside_safe(box: list[int], style: str) -> bool:
    x0, y0, x1, y1 = box
    if style in ("Disc",):  # the #ad tag sits in the top bar by design; only check the sides
        return x0 >= SAFE_X and x1 <= W - SAFE_X
    return x0 >= SAFE_X and x1 <= W - SAFE_X and y0 >= SAFE_TOP and y1 <= SAFE_BOTTOM


# ---------------------------------------------------------------- output QA

def cut_text(base: np.ndarray, out: np.ndarray, panels: list[tuple[int, int]],
             overlay: list[list[int]]) -> list[list[int]]:
    """Text boxes that touch a panel edge in the output but not in the unzoomed base frame."""
    found = dict(zip(("base", "out"), detect_many([base, out])))

    def touching(img):
        res = []
        for y0, y1 in panels:
            for b in panel_boxes(found["base" if img is base else "out"], y0, y1):
                if b[0] <= EDGE or b[2] >= W - EDGE or b[1] <= EDGE or b[3] >= (y1 - y0) - EDGE:
                    res.append([b[0], b[1] + y0, b[2], b[3] + y0])
        return res

    def ours(b):
        return any(b[0] < o[2] and o[0] < b[2] and b[1] < o[3] and o[1] < b[3] for o in overlay)

    before = touching(base)
    after = [b for b in touching(out) if not ours(b)]
    # same edge-touching text before the zoom is a layout matter, not a zoom cut
    return [b for b in after if not any(abs(b[1] - a[1]) < 80 and abs(b[0] - a[0]) < 200 for a in before)]
