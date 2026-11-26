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

from . import db, edit, transcribe
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


def choose_layout(video: Path, start: float, end: float, camp: Campaign) -> Layout:
    rf = camp.reframe
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
        x, y, cw, ch = cam
        # Fit the box to the top panel's aspect around its centre, inside the frame.
        aspect = W / top_h
        cxm, cym = x + cw / 2, y + ch / 2
        # Trim (not grow) to the panel aspect so no gameplay bleeds into the cam panel.
        if cw / ch > aspect:
            cw = ch * aspect
        else:
            ch = cw / aspect
        cw, ch = min(cw, sw), min(ch, sh)
        x0 = int(np.clip(cxm - cw / 2, 0, sw - cw))
        y0 = int(np.clip(cym - ch / 2, 0, sh - ch))
        bot_h = H - top_h
        gw = min(sw, sh * W / bot_h)
        gh = min(sh, gw * bot_h / W)
        gx, gy = int((sw - gw) / 2), int((sh - gh) / 2)
        # Keep the gameplay crop clear of the facecam so no strip of cam shows below.
        cam_l, cam_r = cam[0], cam[0] + cam[2]
        if gx < cam_r and gx + gw > cam_l:
            if cam_l + cam[2] / 2 < sw / 2 and cam_r + gw <= sw:
                gx = int(np.ceil(cam_r))
            elif cam_l - gw >= 0:
                gx = int(cam_l - gw)
        filt = (
            f"[0:v]split=2[c][g];"
            f"[c]crop={_even(cw)}:{_even(ch)}:{x0}:{y0},scale={W}:{top_h}:force_original_aspect_ratio=increase,"
            f"crop={W}:{top_h}[top];"
            f"[g]crop={_even(gw)}:{_even(gh)}:{gx}:{gy},scale={W}:{bot_h}:force_original_aspect_ratio=increase,"
            f"crop={W}:{bot_h}[bot];"
            f"[top][bot]vstack=inputs=2,setsar=1[base]"
        )
        return Layout(mode, filt, caption_y=top_h + 330, hook_y=top_h, handle_y=top_h - 40, note=note or "facecam box from config")

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
    return pre + fixes.get(core, core) + post


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
        fc = (layout.filter + ";" + edit.filter_graph(pieces, W, H) + f";[cut]ass={ass_path.name}[v];"
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
        layout = choose_layout(video, m["start"], m["end"], camp)
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
            pieces = edit.plan(words, end - start, sdir / "audio.wav", start, guide.edit, min_len=camp.min_s)
            clip_len = edit.length(pieces)
            ass.write_text(build_ass(edit.remap(words, pieces), clip_len, layout, hook, camp, settings,
                                     guide=guide, subcaption=m["subcaption"] or ""))
            try:
                render_clip(video, out, start, end, layout, ass, pieces)
                contact_sheet(out, out_dir / f"{cid}.jpg", clip_len)
                qa = quick_checks(out, clip_len)
                qa["layout_note"] = layout.note
                qa["cuts"] = len(pieces) - 1
                qa["cut_s"] = round(end - start - clip_len, 2)
                qa["zooms"] = sum(p.zoom > 1.001 for p in pieces)
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
