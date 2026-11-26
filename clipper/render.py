"""Stage 4: render each moment as a 1080x1920 clip.

Cut -> 9:16 reframe -> word-by-word captions -> hook text for the first 2 s ->
ad disclosure for the first few seconds -> loudness normalisation (-14 LUFS).
Each moment gets `variants_per_moment` renders with different hooks/start.
A contact sheet and quick automatic checks are written next to each clip.
"""
from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import db, edit, textsafe, transcribe
from .config import Campaign, Settings, guide_path
from .style import StyleGuide, ass_color, load as load_guide
from .ingest import probe_video

W, H = 1080, 1920
HOOK_SECONDS = 2.2


# ---------------------------------------------------------------- layout

@dataclass
class Layout:
    mode: str
    filter: str          # ffmpeg filter_complex producing [base]
    caption_y: int
    hook_y: int
    handle_y: int = 0
    note: str = ""
    panels: list = None   # [(y0, y1)] regions zoomed independently; None = whole frame
    text_cut: int = 0     # UI text boxes the crop could not avoid cutting (QA fails if > 0)


def _even(x: float) -> int:
    return int(round(x / 2)) * 2


def _detect_faces(video: Path, start: float, end: float, samples: int = 10) -> tuple[list, int, int]:
    import cv2

    cap = cv2.VideoCapture(str(video))
    sw, sh = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    casc = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    found = []
    for t in np.linspace(start + 0.5, max(start + 0.6, end - 0.5), samples):
        cap.set(cv2.CAP_PROP_POS_MSEC, float(t) * 1000)
        ok, frame = cap.read()
        if not ok:
            continue
        gray = cv2.equalizeHist(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        faces = casc.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6,
                                      minSize=(max(24, sw // 40), max(24, sw // 40)))
        if len(faces):
            found.append(max(faces, key=lambda f: f[2] * f[3]).tolist())
    cap.release()
    return found, sw, sh


def _panel_filter(src: str, out: str, x: float, y: float, w: float, h: float, pw: int, ph: int) -> str:
    """Source rect -> pw x ph panel. Same aspect: crop and scale. Otherwise the whole rect
    is fitted (nothing cut) over a blurred, filled copy of itself."""
    x, y, w, h = int(x), int(y), _even(w), _even(h)
    if abs((w / h) - (pw / ph)) / (pw / ph) < 0.02:
        return f"[{src}]crop={w}:{h}:{x}:{y},scale={pw}:{ph}:flags=lanczos,setsar=1[{out}]"
    return (f"[{src}]crop={w}:{h}:{x}:{y},split=2[{out}b][{out}f];"
            f"[{out}b]scale={pw}:{ph}:force_original_aspect_ratio=increase,crop={pw}:{ph},gblur=sigma=30,"
            f"eq=brightness=-0.15[{out}bb];"
            f"[{out}f]scale={pw}:{ph}:force_original_aspect_ratio=decrease:flags=lanczos[{out}ff];"
            f"[{out}bb][{out}ff]overlay=(W-w)/2:(H-h)/2,setsar=1[{out}]")


def choose_layout(video: Path, start: float, end: float, camp: Campaign, channel_url: str | None = None) -> Layout:
    rf = camp.reframe_for(channel_url)
    info = probe_video(video)
    sw, sh = info["width"], info["height"]
    top_h = _even(H * rf.facecam_height_ratio)
    mode, note, cam = rf.mode, "", None

    if rf.facecam:
        cam = rf.facecam
        mode = "facecam_split" if mode == "auto" else mode
    if mode in ("auto", "track", "facecam_split") and cam is None:
        faces, _, _ = _detect_faces(video, start, end)
        hit = len(faces) / 10
        if faces and hit >= 0.4:
            f = np.median(np.array(faces), axis=0)
            cx, cy, fw = f[0] + f[2] / 2, f[1] + f[3] / 2, f[2]
            spread = np.std([x + w / 2 for x, _, w, _ in faces]) / sw
            small, steady = fw < 0.14 * sw, spread < 0.05
            if mode == "facecam_split" or (mode == "auto" and small and steady and sw > sh):
                cw = min(sw, fw * 4.0)
                ch = cw * top_h / W
                cam = [cx - cw / 2, cy - ch / 2, cw, ch]
                mode = "facecam_split"
                note = f"facecam detected in {hit:.0%} of samples"
            else:
                mode = "track"
                note = f"face tracked at x={cx / sw:.0%} ({hit:.0%} of samples)"
                track_cx = cx
        elif mode == "facecam_split":
            mode, note = "blur_pad", "facecam_split asked but no facecam found; fell back"
        elif mode in ("auto", "track"):
            mode, note = "blur_pad", "no steady face found"

    if mode == "facecam_split":
        bot_h = H - top_h
        # Text anywhere in the moment (sampled), so crops never slice a scoreboard, HUD or overlay.
        n = 12
        times = [start + 0.4 + k * max(0.0, end - start - 0.8) / (n - 1) for k in range(n)]
        boxes = textsafe.must_keep(textsafe.source_text(video, times), frame_w=sw)
        x, y, cw0, ch0 = cam
        cam_rect = (x, y, cw0, ch0)

        # Cam panel: trim to the panel's aspect, unless that would cut text on the cam
        # (follower counts, timers); then show the whole cam box over a blurred fill.
        aspect = W / top_h
        cw, ch = (ch0 * aspect, ch0) if cw0 / ch0 > aspect else (cw0, cw0 / aspect)
        cw, ch = min(cw, sw), min(ch, sh)
        x0 = int(np.clip(x + cw0 / 2 - cw / 2, 0, sw - cw))
        y0 = int(np.clip(y + ch0 / 2 - ch / 2, 0, sh - ch))
        on_cam = [bx for bx in boxes if textsafe.overlaps(bx, cam_rect)]
        if textsafe.crop_cuts(on_cam, x0, y0, cw, ch):
            top = _panel_filter("c", "top", x, y, cw0, ch0, W, top_h)
            note = (note + "; " if note else "") + "whole facecam shown (trimming would cut its text)"
        else:
            top = _panel_filter("c", "top", x0, y0, cw, ch, W, top_h)

        # Gameplay panel: the narrowest crop, clear of the cam, whose sides cut no text.
        cam_l, cam_r = x, x + cw0
        game = [bx for bx in boxes if not textsafe.overlaps(bx, cam_rect, 0.5)]
        gw0 = min(sw, sh * W / bot_h)
        cam_left = cam_l + cw0 / 2 < sw / 2
        # widest crop that still keeps the cam out of the gameplay panel
        hi_w = max(int(gw0), int(sw - np.ceil(cam_r)) if cam_left else int(cam_l))
        best = None
        for gw in list(range(int(gw0), hi_w + 1, 8)) + [hi_w]:
            gh = min(sh, gw * bot_h / W)
            gy = int((sh - gh) / 2)
            if cam_left:
                x_lo, x_hi = int(np.ceil(cam_r)), int(sw - gw)
            else:
                x_lo, x_hi = 0, int(cam_l - gw)
            if x_hi < x_lo:  # cam can't be avoided at this width
                x_lo, x_hi = 0, int(sw - gw)
            want = int(np.clip((sw - gw) / 2, x_lo, x_hi))
            gx = textsafe.clear_crop_x(game, want, gw, gy, int(gh), x_lo, x_hi)
            c = textsafe.crop_cuts(game, gx, gy, gw, gh)
            if best is None or c < best[0]:
                best = (c, gw, gx, gy, gh)
            if c == 0:
                break
        if best[0]:
            # Last resort: allow a wider view that may take in part of the cam's corner of the
            # source. A sliver of duplicated cam is better than a scoreboard sliced in half.
            for gw in range(int(gw0), int(sw) + 1, 8):
                gh = min(sh, gw * bot_h / W)
                gy = int((sh - gh) / 2)
                want = int(np.clip(best[2] - (gw - best[1]) / 2, 0, sw - gw))
                gx = textsafe.clear_crop_x(game, want, gw, gy, int(gh), 0, int(sw - gw))
                c = textsafe.crop_cuts(game, gx, gy, gw, gh)
                if c == 0:
                    best = (0, gw, gx, gy, gh)
                    note = (note + "; " if note else "") + "gameplay view takes in part of the cam area to keep text whole"
                    break
        cuts, gw, gx, gy, gh = best
        if gw > gw0 + 1:
            note = (note + "; " if note else "") + f"gameplay widened to {gw}px to keep UI text whole"
        if cuts:
            note = (note + "; " if note else "") + f"WARNING: {cuts} UI text box(es) still cut at a crop edge"
        layout_cuts = cuts
        # If the gameplay view takes in part of the cam, paint that part over from the
        # surrounding gameplay (delogo interpolates from the edges) so the cam isn't doubled.
        ox0, ox1 = max(gx + 1, int(cam_l), 1), min(gx + gw - 1, int(np.ceil(cam_r)), sw - 2)
        oy0, oy1 = max(gy + 1, int(y), 1), min(gy + gh - 1, int(np.ceil(y + ch0)), sh - 2)
        gsrc, pre = "g", ""
        if ox1 - ox0 > 4 and oy1 - oy0 > 4:
            pre = f"[g]delogo=x={ox0}:y={oy0}:w={ox1 - ox0}:h={oy1 - oy0}[gm];"
            gsrc = "gm"
        bot = pre + _panel_filter(gsrc, "bot", gx, gy, gw, gh, W, bot_h)
        filt = f"[0:v]split=2[c][g];{top};{bot};[top][bot]vstack=inputs=2,setsar=1[base]"
        return Layout(mode, filt, caption_y=top_h + 330, hook_y=top_h, handle_y=top_h - 40,
                      note=note or "facecam box from config", panels=[(0, top_h), (top_h, H)],
                      text_cut=layout_cuts)

    if mode in ("track", "center"):
        cw = _even(min(sw, sh * 9 / 16))
        cx = track_cx if mode == "track" and "track_cx" in locals() else sw / 2
        x0 = int(np.clip(cx - cw / 2, 0, sw - cw))
        filt = f"[0:v]crop={cw}:{sh}:{x0}:0,scale={W}:{H},setsar=1[base]"
        return Layout(mode, filt, caption_y=1290, hook_y=300, handle_y=1170, note=note)

    # blur_pad
    filt = (
        f"[0:v]split=2[bg0][fg0];"
        f"[bg0]scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H},gblur=sigma=40,"
        f"eq=brightness=-0.12[bg];"
        f"[fg0]scale={W}:-2[fg];"
        f"[bg][fg]overlay=(W-w)/2:(H-h)/2,setsar=1[base]"
    )
    fg_h = W * sh / sw
    return Layout("blur_pad", filt, caption_y=int(H / 2 + fg_h / 2 + 150),
                  hook_y=int(H / 2 - fg_h / 2 - 150), handle_y=int(H / 2 - fg_h / 2 - 40), note=note)


# ---------------------------------------------------------------- captions

def _ass_escape(s: str) -> str:
    return s.replace("\\", "").replace("{", "(").replace("}", ")").replace("\n", " ")


def _ts(t: float) -> str:
    t = max(0.0, t)
    return f"{int(t // 3600)}:{int(t % 3600 // 60):02d}:{t % 60:05.2f}"


def chunk_words(words: list[dict], max_words: int = 3, max_dur: float = 1.4) -> list[list[dict]]:
    chunks, cur = [], []
    for w in words:
        if cur and (len(cur) >= max_words or w["e"] - cur[0]["s"] > max_dur or w["s"] - cur[-1]["e"] > 0.5
                    or re.search(r"[.!?,]$", cur[-1]["w"])):
            chunks.append(cur)
            cur = []
        cur.append(w)
    if cur:
        chunks.append(cur)
    return chunks


def build_ass(words: list[dict], clip_len: float, layout: Layout, hook: str, camp: Campaign,
              settings: Settings, guide: StyleGuide | None = None, subcaption: str = "") -> str:
    g = guide or StyleGuide()
    c, hk, sc = g.caption, g.hook, g.subcaption
    font = settings.caption_font
    hook_y = int(hk.y_frac * H) if hk.y_frac else layout.hook_y
    hook_box = hk.style == "box"
    sub_box = sc.style == "box"
    head = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {W}
PlayResY: {H}
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Cap,{font},{c.font_size},{ass_color(c.text_color)},{ass_color(c.text_color)},{ass_color(c.outline_color)},&H80000000,-1,0,0,0,100,100,0,0,1,{c.outline_px},3,5,60,60,0,1
Style: Hook,{font},{hk.font_size},{ass_color(hk.text_color)},{ass_color(hk.text_color)},{ass_color(hk.box_color) if hook_box else "&H00000000"},{ass_color(hk.box_color) if hook_box else "&H80000000"},-1,0,0,0,100,100,0,0,{3 if hook_box else 1},{18 if hook_box else 8},0,5,80,80,0,1
Style: Sub,{font},{sc.font_size},{ass_color(sc.text_color)},{ass_color(sc.text_color)},{ass_color(sc.box_color) if sub_box else "&H00000000"},{ass_color(sc.box_color) if sub_box else "&H80000000"},-1,0,0,0,100,100,0,0,{3 if sub_box else 1},{14 if sub_box else 6},0,5,80,80,0,1
Style: Disc,{font},40,&H00FFFFFF,&H00FFFFFF,&H99000000,&H99000000,-1,0,0,0,100,100,0,0,3,10,0,8,60,60,0,1
Style: Handle,{font},38,&H40FFFFFF,&H40FFFFFF,&H80000000,&H80000000,-1,0,0,0,100,100,0,0,1,3,0,5,60,60,0,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    ev = []
    y = int(c.y_frac * H) if c.y_frac else layout.caption_y
    hi, base = ass_color(c.highlight_color), ass_color(c.text_color)
    chunks = chunk_words(words, max_words=c.words_per_chunk, max_dur=c.max_chunk_s)
    for ci, chunk in enumerate(chunks):
        texts = [_ass_escape(_fix_word(w["w"], camp.caption_fixes).upper() if c.uppercase
                             else _fix_word(w["w"], camp.caption_fixes)) for w in chunk]
        chunk_end = min(clip_len, chunk[-1]["e"] + 0.15)
        if ci + 1 < len(chunks):  # never overlap the next chunk, or libass stacks them
            chunk_end = min(chunk_end, chunks[ci + 1][0]["s"])
        for i, w in enumerate(chunk):
            s = w["s"]
            e = chunk[i + 1]["s"] if i + 1 < len(chunk) else chunk_end
            if e <= s:
                continue
            if c.highlight_mode == "color":
                mark = lambda t: "{\\c" + hi + "}" + t + "{\\c" + base + "}"
            elif c.highlight_mode == "box":  # thick coloured outline reads as a box behind the word
                mark = lambda t: ("{\\3c" + hi + "\\bord" + str(max(12, c.outline_px * 2)) + "}" + t
                                  + "{\\3c" + ass_color(c.outline_color) + "\\bord" + str(c.outline_px) + "}")
            else:
                mark = lambda t: t
            parts = [mark(t) if k == i else t for k, t in enumerate(texts)]
            pop = "{\\fscx88\\fscy88\\t(0,90,\\fscx100\\fscy100)}" if (i == 0 and c.pop_in) else ""
            ev.append(f"Dialogue: 1,{_ts(s)},{_ts(e)},Cap,,0,0,0,,{{\\an5\\pos(540,{y})}}{pop}{' '.join(parts)}")
    hook_end = min(hk.seconds, clip_len) if hook else 0.0
    if hook and hk.seconds > 0:
        h = _ass_escape(hook.upper())
        ev.append(f"Dialogue: 3,{_ts(0)},{_ts(hook_end)},Hook,,0,0,0,,"
                  f"{{\\an5\\pos(540,{hook_y})\\fad(0,180)\\fscx92\\fscy92\\t(0,120,\\fscx100\\fscy100)}}{h}")
    if subcaption and sc.enabled:
        sub_y = int(sc.y_frac * H) if sc.y_frac else hook_y
        # Sharing the hook's spot? Then wait for the hook to leave.
        t0 = max(sc.from_s, hook_end if abs(sub_y - hook_y) < 160 else 0.0)
        words_ = subcaption.split()[: sc.max_words]
        if t0 < clip_len and words_:
            ev.append(f"Dialogue: 2,{_ts(t0)},{_ts(clip_len)},Sub,,0,0,0,,"
                      f"{{\\an5\\pos(540,{sub_y})\\fad(120,0)}}{_ass_escape(' '.join(words_))}")
    if camp.disclosure:
        ev.append(f"Dialogue: 4,{_ts(0)},{_ts(min(camp.disclosure_seconds, clip_len))},Disc,,0,0,0,,"
                  f"{{\\an8\\pos(540,165)}}{_ass_escape(camp.disclosure)}")
    if camp.handle:
        # A guide can put captions on the facecam seam, where the handle line sits; lift it clear.
        handle_y = layout.handle_y
        if abs(handle_y - y) < c.font_size // 2 + 40:
            handle_y = y - c.font_size // 2 - 40
        ev.append(f"Dialogue: 0,{_ts(0)},{_ts(clip_len)},Handle,,0,0,0,,"
                  f"{{\\an5\\pos(540,{handle_y})}}{_ass_escape(camp.handle)}")
    return head + "\n".join(ev) + "\n"


def _fix_word(w: str, fixes: dict[str, str]) -> str:
    """Whole-word replacement that keeps surrounding punctuation."""
    if not fixes:
        return w
    m = re.match(r"^(\W*)(.*?)(\W*)$", w)
    pre, core, post = m.groups()
    # Case-insensitive: Whisper writes "cypress" as often as "Cypress".
    right = next((r for k, r in fixes.items() if k.lower() == core.lower()), None)
    if right is None:
        return w
    if core.isupper() and len(core) > 1:  # keep shouting as shouting
        right = right.upper()
    return pre + right + post


def words_in(segments: list[dict], start: float, end: float) -> list[dict]:
    out = []
    for sg in segments:
        for w in sg["words"]:
            if w["e"] > start and w["s"] < end:
                out.append({"s": max(0.0, w["s"] - start), "e": min(end, w["e"]) - start, "w": w["w"]})
    return out


# ---------------------------------------------------------------- render + checks

def render_clip(video: Path, out: Path, start: float, end: float, layout: Layout, ass_path: Path,
                pieces: list | None = None) -> None:
    """`pieces` (edit.plan) applies jump cuts and zooms before the captions are burned in."""
    dur = end - start
    if pieces and (len(pieces) > 1 or pieces[0].zoom > 1.001):
        cut_dur = edit.length(pieces)
        fc = (layout.filter + ";" + edit.filter_graph(pieces, W, H, layout.panels) + f";[cut]ass={ass_path.name}[v];"
              f"[cuta]loudnorm=I=-14:TP=-1.5:LRA=11,afade=t=in:d=0.04,"
              f"afade=t=out:st={max(0.0, cut_dur - 0.3):.2f}:d=0.3,aresample=48000[a]")
    else:
        fc = (
        layout.filter + f";[base]ass={ass_path.name}[v];"
        f"[0:a]loudnorm=I=-14:TP=-1.5:LRA=11,afade=t=in:d=0.04,"
        f"afade=t=out:st={max(0.0, dur - 0.3):.2f}:d=0.3,aresample=48000[a]"
    )
    cmd = [
        "ffmpeg", "-v", "error", "-y", "-ss", f"{start:.3f}", "-i", str(video.resolve()), "-t", f"{dur:.3f}",
        "-filter_complex", fc, "-map", "[v]", "-map", "[a]",
        "-r", "30", "-c:v", "libx264", "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart", str(out.resolve()),
    ]
    subprocess.run(cmd, check=True, cwd=ass_path.parent)


def text_safe_crops(video: Path, start: float, layout: Layout, pieces: list) -> None:
    """Pick each zoomed piece's per-panel crop so no on-screen text is cut (textsafe rule 2)."""
    panels = layout.panels or [(0, H)]
    for p in pieces:
        if p.zoom <= 1.001:
            p.crops = [(1.0, 0, 0, W, y1 - y0) for y0, y1 in panels]
            continue
        times = [p.s + 0.15, (p.s + p.e) / 2, max(p.s + 0.15, p.e - 0.15)]
        frames = textsafe.base_frames(video, start, layout.filter, times)
        boxes = [b for bs in textsafe.detect_many(list(frames.values())) for b in bs]
        p.crops = [textsafe.solve_zoom(textsafe.panel_boxes(boxes, y0, y1), W, y1 - y0, p.zoom)
                   for y0, y1 in panels]


def place_hook(ass: Path, video: Path, start: float, layout: Layout) -> int:
    """Move the hook so it never sits on top of on-screen game text (banners, scores).
    Returns how far it moved (px)."""
    meas = [e for e in textsafe.overlay_boxes(ass) if e["style"] == "Hook" and e["box"]]
    if not meas:
        return 0
    line, (x0, y0, x1, y1) = meas[0]["line"], meas[0]["box"]
    m = re.search(r"\\pos\(540,(\d+)\)", line)
    if not m:
        return 0
    cur = int(m.group(1))
    frames = textsafe.base_frames(video, start, layout.filter, [0.3, 1.2, 2.2])
    game = [b for bs in textsafe.detect_many(list(frames.values())) for b in bs]
    others = [e["box"] for e in textsafe.overlay_boxes(ass)
              if e["box"] and e["style"] in ("Cap", "Handle", "Sub") and _active(e["line"], 1.0)]
    half_up, half_dn = cur - y0, y1 - cur
    lo = (layout.panels[-1][0] if layout.panels else 0) + half_up + 20
    lo = max(lo, textsafe.SAFE_TOP + half_up)
    hi = textsafe.SAFE_BOTTOM - half_dn

    def clash(y):
        a, b = y - half_up - 20, y + half_dn + 20
        return sum(1 for g in game + others if g[1] < b and g[3] > a and g[0] < x1 + 20 and g[2] > x0 - 20)

    if not clash(cur):
        return 0
    cands = sorted(range(int(lo), int(hi) + 1, 10), key=lambda y: (clash(y), abs(y - cur)))
    best = cands[0] if cands else cur
    if clash(best) >= clash(cur):
        return 0
    text = ass.read_text()
    for e in [e for e in textsafe.overlay_boxes(ass) if e["style"] == "Hook"]:
        text = text.replace(e["line"], e["line"].replace(f"\\pos(540,{cur})", f"\\pos(540,{best})"), 1)
    ass.write_text(text)
    return best - cur


def zoom_qa(video: Path, out: Path, start: float, layout: Layout, pieces: list, ass: Path) -> list[int]:
    """Indexes of zoomed pieces where the output cuts on-screen text the unzoomed frame shows whole."""
    panels = layout.panels or [(0, H)]
    overlay = textsafe.overlay_boxes(ass)
    bad, t_out = [], 0.0
    for i, p in enumerate(pieces):
        mid_out = t_out + (p.e - p.s) / 2
        t_out += p.e - p.s
        if not any(c[0] > 1.001 for c in (p.crops or [])):
            continue
        base = textsafe.base_frames(video, start, layout.filter, [(p.s + p.e) / 2]).get((p.s + p.e) / 2)
        img = textsafe.output_frame(out, mid_out)
        if base is None or img is None:
            continue
        ours = [e["box"] for e in overlay if e["box"] and _active(e["line"], mid_out)]
        if textsafe.cut_text(base, img, panels, ours):
            bad.append(i)
    return bad


def _active(line: str, t: float) -> bool:
    f = line.split(",")
    def sec(x):
        h, m, s_ = x.split(":")
        return int(h) * 3600 + int(m) * 60 + float(s_)
    return sec(f[1]) <= t <= sec(f[2])


def contact_sheet(clip: Path, out: Path, dur: float, n: int = 6) -> None:
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-i", str(clip), "-vf",
         f"fps={n / dur:.4f},scale=270:-2,tile={n}x1:padding=6:color=white", "-frames:v", "1", str(out)],
        check=True,
    )


def quick_checks(clip: Path, expected_dur: float) -> dict:
    p = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration:stream=codec_type,width,height",
         "-of", "json", str(clip)], capture_output=True, text=True, check=True)
    info = json.loads(p.stdout)
    v = next(s for s in info["streams"] if s["codec_type"] == "video")
    has_audio = any(s["codec_type"] == "audio" for s in info["streams"])
    dur = float(info["format"]["duration"])
    d = subprocess.run(
        ["ffmpeg", "-v", "info", "-i", str(clip), "-vf",
         "blackdetect=d=0.5:pix_th=0.08,freezedetect=n=0.003:d=2",
         "-af", "ebur128", "-f", "null", "-"], capture_output=True, text=True)
    black = len(re.findall(r"black_start", d.stderr))
    frozen = len(re.findall(r"freeze_start", d.stderr))
    lufs = re.findall(r"I:\s+(-?[\d.]+) LUFS", d.stderr)
    checks = {
        "size_ok": v["width"] == W and v["height"] == H,
        "duration_ok": abs(dur - expected_dur) < 0.6,
        "has_audio": has_audio,
        "black_segments": black,
        "frozen_segments": frozen,
        "integrated_lufs": float(lufs[-1]) if lufs else None,
        "duration": round(dur, 2),
    }
    checks["pass"] = (checks["size_ok"] and checks["duration_ok"] and has_audio
                      and black == 0 and frozen == 0)
    return checks


def render_source(settings: Settings, camp: Campaign, source_id: str, force: bool = False,
                  only: list[int] | None = None, use_guide: bool = True, tag: str = "") -> list[Path]:
    """Render every picked moment. With use_guide, styling follows the tastemaker's guide
    (if one exists). `tag` writes to clips/<source>-<tag>/ so styles can be compared."""
    guide = load_guide(guide_path(settings, camp)) if use_guide else StyleGuide()
    con = db.connect(settings.work_dir)
    src = db.get_source(con, source_id)
    sdir = Path(src["dir"])
    video = sdir / "source.mp4"
    segments = transcribe.load(sdir)
    moments = con.execute("SELECT * FROM moment WHERE source_id=? ORDER BY rank", (source_id,)).fetchall()
    out_dir = settings.work_dir / "clips" / (f"{source_id}-{tag}" if tag else source_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    made = []
    for m in moments:
        if only and m["rank"] not in only:
            continue
        hooks = json.loads(m["hooks"] or "[]") or [""]
        pieces = None
        layout = choose_layout(video, m["start"], m["end"], camp, src["channel_url"])
        for v in range(1, settings.variants_per_moment + 1):
            hook = hooks[(v - 1) % len(hooks)]
            # Variants share the same in-point (campaigns like SypherPK's require opening
            # on the first spoken line) and differ in hook text.
            start = m["start"]
            end = m["end"]
            cid = f"{m['id']}-v{v}"
            out = out_dir / f"{cid}.mp4"
            if out.exists() and not force:
                made.append(out)
                continue
            ass = out_dir / f"{cid}.ass"
            words = words_in(segments, start, end)
            if pieces is None:  # shared by the moment's variants
                pieces = edit.plan(words, end - start, sdir / "audio.wav", start, guide.edit, min_len=camp.min_s)
                text_safe_crops(video, start, layout, pieces)
            clip_len = edit.length(pieces)
            ass.write_text(build_ass(edit.remap(words, pieces), clip_len, layout, hook, camp, settings,
                                     guide=guide, subcaption=m["subcaption"] or ""))
            fit = textsafe.fit_overlay(ass, {"Cap": guide.caption.font_size, "Hook": guide.hook.font_size,
                                             "Sub": guide.subcaption.font_size, "Disc": 40, "Handle": 38})
            fit["hook_moved"] = place_hook(ass, video, start, layout)
            try:
                for attempt in range(3):
                    render_clip(video, out, start, end, layout, ass, pieces)
                    cut = zoom_qa(video, out, start, layout, pieces, ass)
                    if not cut:
                        break
                    print(f"[render] {cid}: zoom cut on-screen text in {len(cut)} piece(s); retrying with less zoom")
                    for i in cut:
                        p = pieces[i]
                        p.zoom = 1.0 if attempt >= 1 else 1 + (p.zoom - 1) / 2
                        text_safe_crops(video, start, layout, [p])
                (out_dir / f"{cid}.edit.json").write_text(json.dumps(
                    [{"s": p.s, "e": p.e, "zoom": p.zoom, "crops": p.crops} for p in pieces]))
                contact_sheet(out, out_dir / f"{cid}.jpg", clip_len)
                qa = quick_checks(out, clip_len)
                qa.update(fit)
                qa["zoom_text_cuts"] = len(cut)
                qa["layout_text_cuts"] = layout.text_cut
                qa["pass"] = qa["pass"] and fit["overlay_ok"] and not cut and not layout.text_cut
                qa["layout_note"] = layout.note
                qa["cuts"] = len(pieces) - 1
                qa["cut_s"] = round(end - start - clip_len, 2)
                qa["zooms"] = sum(any(c[0] > 1.001 for c in (p.crops or [])) for p in pieces)
                status = "rendered" if qa["pass"] else "qa_failed"
                db.upsert(con, "clip", {"id": cid, "moment_id": m["id"], "variant": v, "path": str(out),
                                        "layout": layout.mode, "qa": qa, "status": status, "error": None})
                print(f"[render] {cid}: {layout.mode} {qa['duration']}s ({qa['cuts']} cuts, "
                      f"{qa['cut_s']}s of pauses removed, {qa['zooms']} zoomed) lufs={qa['integrated_lufs']} "
                      f"{'OK' if qa['pass'] else 'CHECK'} hook={hook!r}")
                made.append(out)
            except subprocess.CalledProcessError as e:
                db.upsert(con, "clip", {"id": cid, "moment_id": m["id"], "variant": v, "path": None,
                                        "layout": layout.mode, "qa": None, "status": "failed", "error": str(e)})
                print(f"[render] {cid}: FAILED {e}")
    db.set_status(con, "source", source_id, "rendered")
    write_review(settings, source_id, out_dir)
    return made


def write_review(settings: Settings, source_id: str, out_dir: Path | None = None) -> Path:
    """A plain HTML page to watch every clip side by side with its rationale."""
    con = db.connect(settings.work_dir)
    src = db.get_source(con, source_id)
    rows = con.execute(
        "SELECT c.*, m.title, m.rationale, m.start, m.end, m.ranked_by, m.rank FROM clip c "
        "JOIN moment m ON m.id=c.moment_id WHERE m.source_id=? ORDER BY m.rank, c.variant",
        (source_id,)).fetchall()
    out_dir = out_dir or settings.work_dir / "clips" / source_id
    cards = []
    for r in rows:
        if not r["path"]:
            continue
        qa = json.loads(r["qa"] or "{}")
        name = Path(r["path"]).name
        cards.append(f"""<div class="c"><video src="{name}" controls preload="metadata"></video>
<h3>#{r['rank']} v{r['variant']} · {r['end'] - r['start']:.0f}s · {r['layout']}</h3>
<p><b>{_html(r['title'])}</b></p><p>{_html(r['rationale'])}</p>
<p class="m">ranked by {r['ranked_by']} · {r['start']:.1f}s–{r['end']:.1f}s in source ·
{'checks passed' if qa.get('pass') else 'check: ' + _html(json.dumps(qa))}</p></div>""")
    html = f"""<!doctype html><meta charset="utf-8"><title>Clips: {_html(src['title'] or source_id)}</title>
<style>body{{font:14px system-ui;margin:24px;background:#111;color:#eee}}
.g{{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:20px}}
video{{width:100%;aspect-ratio:9/16;background:#000;border-radius:8px}}h3{{margin:8px 0 4px}}
p{{margin:4px 0}}.m{{color:#999;font-size:12px}}</style>
<h1>{_html(src['title'] or source_id)}</h1><p class="m">{_html(src['url'])}</p><div class="g">{''.join(cards)}</div>"""
    p = out_dir / "review.html"
    p.write_text(html)
    return p


def _html(s) -> str:
    return (str(s or "")).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
