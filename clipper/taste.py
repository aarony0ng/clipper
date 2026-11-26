"""The tastemaker: learn what top clips of similar content have in common.

collect  -> find the most-viewed short clips for a campaign's sources (yt-dlp)
measure  -> per clip: length, cut rate, loudness, speech pace, hook line, frame grid
analyze  -> Claude looks at each clip's frames + transcript, then distills a
            style guide (style.StyleGuide) weighted by views
report   -> style_guide.md, a readable summary of what was learned

The clipper reads the guide when ranking moments (rules go into Claude's
prompt) and when rendering (caption, hook and subcaption styling).

Downloaded clips are for analysis only; they stay in work/taste and are never
re-posted.
"""
from __future__ import annotations

import json
import subprocess
from datetime import date
from pathlib import Path

import numpy as np
from pydantic import BaseModel, Field

from . import llm
from .config import Campaign, Settings, guide_path
from .style import StyleGuide


def _dir(settings: Settings, camp: Campaign) -> Path:
    d = settings.work_dir / "taste" / camp.id
    (d / "clips").mkdir(parents=True, exist_ok=True)
    return d


# ---------------------------------------------------------------- collect

def collect(settings: Settings, camp: Campaign, sources: list[str] | None = None, n: int = 24,
            max_dur: float = 90.0, min_views: int = 10_000) -> list[dict]:
    import yt_dlp

    sources = sources or camp.taste_sources
    if not sources:
        raise ValueError(f"No taste_sources for '{camp.id}'. Add yt-dlp URLs or searches to campaigns.yaml "
                         "or pass --source.")
    d = _dir(settings, camp)
    found: dict[str, dict] = {}
    with yt_dlp.YoutubeDL({"quiet": True, "no_warnings": True, "extract_flat": "in_playlist",
                           "ignoreerrors": True}) as ydl:
        for src in sources:
            info = ydl.extract_info(src, download=False) or {}
            entries = info.get("entries") or [info]
            got = 0
            for e in entries:
                if not e or not e.get("id"):
                    continue
                dur, views = e.get("duration"), e.get("view_count")
                if dur and dur > max_dur:
                    continue
                url = e.get("url") or e.get("webpage_url") or e.get("original_url")
                if not url:
                    continue
                if url.startswith(("http://", "https://")) is False:
                    url = f"https://www.youtube.com/shorts/{e['id']}"
                key = f"{(e.get('ie_key') or info.get('extractor_key') or 'x').lower()}-{e['id']}"
                prev = found.get(key)
                if not prev or (views or 0) > (prev.get("view_count") or 0):
                    found[key] = {"key": key, "url": url, "title": e.get("title"), "view_count": views,
                                  "duration": dur, "channel": e.get("channel") or e.get("uploader"),
                                  "source": src}
                    got += 1
            print(f"[taste] {src}: {got} candidates")

    ranked = sorted(found.values(), key=lambda c: c.get("view_count") or 0, reverse=True)
    picked = []
    for c in ranked:
        if len(picked) >= n:
            break
        if (c.get("view_count") or 0) < min_views:
            continue
        cdir = d / "clips" / c["key"]
        cdir.mkdir(exist_ok=True)
        if not (cdir / "clip.mp4").exists():
            opts = {"quiet": True, "no_warnings": True, "noprogress": True,
                    "format": "bv*[height<=1280][ext=mp4]+ba[ext=m4a]/bv*[height<=1280]+ba/b",
                    "merge_output_format": "mp4", "outtmpl": {"default": str(cdir / "clip.%(ext)s")}}
            try:
                with yt_dlp.YoutubeDL(opts) as ydl:
                    full = ydl.sanitize_info(ydl.extract_info(c["url"], download=True))
            except Exception as ex:  # private, removed, region-locked...
                print(f"[taste] skip {c['url']}: {ex}")
                continue
            if full.get("duration") and full["duration"] > max_dur:
                continue
            c.update({
                "title": full.get("title"), "view_count": full.get("view_count"),
                "like_count": full.get("like_count"), "comment_count": full.get("comment_count"),
                "duration": full.get("duration"), "channel": full.get("channel") or full.get("uploader"),
                "channel_followers": full.get("channel_follower_count"),
                "upload_date": full.get("upload_date"),
                "description": (full.get("description") or "")[:400], "tags": (full.get("tags") or [])[:20],
            })
            (cdir / "meta.json").write_text(json.dumps(c, indent=2, ensure_ascii=False))
        picked.append(json.loads((cdir / "meta.json").read_text()))
        print(f"[taste] #{len(picked)} {picked[-1].get('view_count') or 0:>12,} views  {picked[-1]['title']!r}")
    (d / "collected.json").write_text(json.dumps(picked, indent=2, ensure_ascii=False))
    return picked


# ---------------------------------------------------------------- measure

def _frame(video: Path, t: float, out: Path, width: int) -> bool:
    r = subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{t:.2f}", "-i", str(video), "-frames:v", "1",
                        "-vf", f"scale={width}:-2", str(out)], capture_output=True)
    return r.returncode == 0 and out.exists()


def _grid(video: Path, times: list[float], out: Path, cols: int = 6, cell=(270, 480)) -> None:
    import cv2

    cells = []
    tmp = out.parent / "_f.jpg"
    for t in times:
        img = None
        if _frame(video, t, tmp, cell[0] * 2):
            img = cv2.imread(str(tmp))
        if img is None:
            img = np.zeros((cell[1], cell[0], 3), np.uint8)
        # Fit into a 9:16 cell without cropping, so captions at the edges stay visible.
        h, w = img.shape[:2]
        s = min(cell[0] / w, cell[1] / h)
        img = cv2.resize(img, (max(1, int(w * s)), max(1, int(h * s))))
        canvas = np.zeros((cell[1], cell[0], 3), np.uint8)
        y0, x0 = (cell[1] - img.shape[0]) // 2, (cell[0] - img.shape[1]) // 2
        canvas[y0:y0 + img.shape[0], x0:x0 + img.shape[1]] = img
        cv2.rectangle(canvas, (0, 0), (64, 30), (0, 0, 0), -1)
        cv2.putText(canvas, f"{t:.1f}s", (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cells.append(canvas)
    tmp.unlink(missing_ok=True)
    while len(cells) % cols:
        cells.append(np.zeros((cell[1], cell[0], 3), np.uint8))
    rows = [np.hstack(cells[i:i + cols]) for i in range(0, len(cells), cols)]
    cv2.imwrite(str(out), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 85])


def measure(settings: Settings, camp: Campaign, force: bool = False) -> list[dict]:
    from . import ingest, signals, transcribe

    d = _dir(settings, camp)
    out = []
    for cdir in sorted(p for p in (d / "clips").iterdir() if (p / "clip.mp4").exists()):
        fpath = cdir / "features.json"
        if fpath.exists() and not force:
            out.append(json.loads(fpath.read_text()))
            continue
        video = cdir / "clip.mp4"
        meta = json.loads((cdir / "meta.json").read_text())
        dur = ingest.probe_duration(video)
        vinfo = ingest.probe_video(video)
        ingest.extract_audio(video, cdir / "audio.wav")
        loud = signals.loudness(cdir / "audio.wav")
        n = max(1, len(loud))
        cuts = signals.scene_cuts(video, n, threshold=0.3)
        cut_times = [i for i, c in enumerate(cuts) for _ in range(int(c))]
        segs = transcribe._whisper(cdir / "audio.wav", settings)
        words = [w for s in segs for w in s["words"]]
        hook_words = [w["w"] for w in words if w["s"] < 3.0]
        times = [0.2, 1.0, 2.0, 3.0] + list(np.linspace(4.0, max(4.5, dur - 0.4), 8))
        _grid(video, [t for t in times if t < dur], cdir / "grid.jpg")
        _frame(video, 0.4, cdir / "frame0.jpg", 720)
        f = {
            "key": meta["key"], "title": meta.get("title"), "url": meta.get("url"),
            "views": meta.get("view_count"), "likes": meta.get("like_count"),
            "comments": meta.get("comment_count"), "channel": meta.get("channel"),
            "followers": meta.get("channel_followers"), "upload_date": meta.get("upload_date"),
            "duration_s": round(dur, 2), "width": vinfo["width"], "height": vinfo["height"],
            "vertical": vinfo["height"] > vinfo["width"],
            "cuts": len(cut_times), "cuts_per_10s": round(len(cut_times) / dur * 10, 2),
            "first_cut_s": cut_times[0] if cut_times else None,
            "loudness_mean_db": round(float(np.mean(loud)), 1) if len(loud) else None,
            "silence_share": round(float(np.mean(loud < -40)), 2) if len(loud) else None,
            "first_word_s": round(words[0]["s"], 2) if words else None,
            "words_per_s": round(len(words) / dur, 2),
            "hook_line": " ".join(hook_words),
            "transcript": " ".join(w["w"] for w in words)[:1500],
            "like_rate": round(meta["like_count"] / meta["view_count"], 4)
            if meta.get("like_count") and meta.get("view_count") else None,
        }
        fpath.write_text(json.dumps(f, indent=2, ensure_ascii=False))
        print(f"[taste] measured {cdir.name}: {dur:.0f}s, {f['cuts_per_10s']} cuts/10s, "
              f"{f['words_per_s']} words/s, first word {f['first_word_s']}s")
        out.append(f)
    (d / "features.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    return out


# ---------------------------------------------------------------- analyze

class CaptionObs(BaseModel):
    present: bool
    y_frac: float | None = Field(None, description="Vertical centre of spoken captions, 0=top 1=bottom")
    words_per_chunk: int | None = None
    uppercase: bool | None = None
    font: str = Field("", description="e.g. 'heavy sans, rounded', 'Impact-like'")
    text_color: str = Field("", description="#RRGGBB")
    highlight: str = Field("", description="How the current word is marked, e.g. 'yellow text', 'green box', 'none'")
    highlight_color: str = Field("", description="#RRGGBB or empty")
    outline_or_box: str = ""
    emoji: bool = False


class SubcaptionObs(BaseModel):
    present: bool
    text: str = ""
    y_frac: float | None = None
    style: str = Field("", description="e.g. 'white box, black text', 'outlined white text'")
    persistent: bool = Field(False, description="Stays up for most of the clip")


class ClipAnalysis(BaseModel):
    key: str
    content_type: str = Field(description="e.g. 'reaction', 'clutch play', 'banter', 'fail', 'story'")
    first_2s: str = Field(description="What happens on screen and in audio in the first 2 seconds")
    hook_type: str = Field(description="e.g. 'mid-action cold open', 'bold quote', 'question', 'tease text'")
    hook_text_on_screen: str = ""
    caption: CaptionObs
    subcaption: SubcaptionObs
    layout: str = Field(description="e.g. 'facecam top / gameplay bottom', 'full-frame gameplay', 'zoomed face'")
    edits: list[str] = Field(default_factory=list, description="zooms, jump cuts, freeze frames, sfx cues, etc.")
    payoff_s: float | None = Field(None, description="When the payoff lands")
    why_it_works: str


def _analysis_system() -> str:
    return (
        "You are the tastemaker for a short-form clipping team. You study top-performing "
        "vertical clips of streamers and describe precisely what they do, so an automated "
        "editor can copy the patterns that drive views. Report what you see, not what you "
        "assume: if a detail is not visible in the frames, leave it empty or null. Colours as "
        "#RRGGBB. Positions as a fraction of frame height (0 = top, 1 = bottom)."
    )


def _guide_system() -> str:
    return (
        "You are the tastemaker for a short-form clipping team. From per-clip analyses of the "
        "top-performing clips in a niche, write the style guide the team's editor will follow. "
        "Weight each clip by its views, and keep only patterns that are consistent across the "
        "high-view clips; ignore one-offs. Translate patterns into concrete parameters the "
        "renderer supports (caption words per chunk, case, size, position, colours, highlight; "
        "hook duration and style; an optional persistent subcaption/title line; target length). "
        "Write rules for the moment picker and hook writer as short imperative sentences. "
        "Every rule must be supported by an 'evidence' line naming what you saw and how often. "
        "Frame sizes are 1080x1920; keep captions inside y 0.15-0.80 so platform UI doesn't cover them."
    )


def analyze(settings: Settings, camp: Campaign, force: bool = False) -> StyleGuide | None:
    d = _dir(settings, camp)
    feats = json.loads((d / "features.json").read_text())
    adir = d / "analysis"
    adir.mkdir(exist_ok=True)

    if not llm.available():
        _write_manual_brief(d, feats, camp)
        print(f"[taste] No Anthropic credentials. Wrote {d / 'ANALYZE.md'}: have Claude follow it, "
              f"then run `clipper taste report {camp.id}`.")
        return None

    analyses = []
    for f in feats:
        ap = adir / f"{f['key']}.json"
        if ap.exists() and not force:
            analyses.append(json.loads(ap.read_text()))
            continue
        cdir = d / "clips" / f["key"]
        text = _clip_brief(f)
        a = llm.parse(settings.claude_model, _analysis_system(),
                      [llm.image_block(cdir / "grid.jpg"), llm.image_block(cdir / "frame0.jpg"),
                       {"type": "text", "text": text}], ClipAnalysis, effort="medium", max_tokens=8000)
        a.key = f["key"]
        ap.write_text(a.model_dump_json(indent=2))
        analyses.append(a.model_dump())
        print(f"[taste] analysed {f['key']}: {a.hook_type} / {a.layout}")

    guide = llm.parse(settings.claude_model, _guide_system(), _guide_brief(camp, feats, analyses),
                      StyleGuide, effort="high")
    guide.based_on = _based_on(feats)
    gp = guide_path(settings, camp)
    gp.parent.mkdir(parents=True, exist_ok=True)
    gp.write_text(guide.model_dump_json(indent=2))
    report(settings, camp)
    return guide


def _clip_brief(f: dict) -> str:
    return (
        f"Clip {f['key']} — {f.get('views') or 0:,} views, {f.get('likes') or 0:,} likes, "
        f"channel {f.get('channel')!r}, title {f.get('title')!r}.\n"
        f"Measured: {f['duration_s']}s long, {'vertical' if f['vertical'] else 'horizontal'} "
        f"{f['width']}x{f['height']}, {f['cuts_per_10s']} scene cuts per 10s (first at {f['first_cut_s']}s), "
        f"first word at {f['first_word_s']}s, {f['words_per_s']} words/s, silence share {f['silence_share']}.\n"
        f"Spoken in first 3s: {f['hook_line']!r}\n"
        f"Transcript: {f['transcript']}\n\n"
        "Image 1 is a grid of frames with timestamps (0.2s, 1s, 2s, 3s, then spread to the end). "
        "Image 2 is the opening frame at higher resolution. Describe the clip."
    )


def _based_on(feats: list[dict]) -> str:
    views = sum(f.get("views") or 0 for f in feats)
    return f"{len(feats)} clips, {views / 1e6:.1f}M total views, analysed {date.today().isoformat()}"


def _guide_brief(camp: Campaign, feats: list[dict], analyses: list[dict]) -> str:
    by_key = {f["key"]: f for f in feats}
    rows = []
    for a in sorted(analyses, key=lambda a: by_key.get(a["key"], {}).get("views") or 0, reverse=True):
        f = by_key.get(a["key"], {})
        slim = {k: f.get(k) for k in ("views", "likes", "like_rate", "duration_s", "cuts_per_10s",
                                      "first_word_s", "words_per_s", "title")}
        rows.append(json.dumps({"measured": slim, "observed": a}, ensure_ascii=False))
    return (
        f"<campaign>\nCreator: {camp.creator}\nRules: clip length {camp.min_s:.0f}-{camp.max_s:.0f}s. "
        f"Prohibited: {', '.join(camp.prohibited) or 'none'}.\nBrief: {camp.brief.strip()}\n</campaign>\n\n"
        "Our renderer: 1080x1920; word-by-word spoken captions (chunks of N words, current word "
        "highlighted by colour or box); a hook text block for the first seconds; an optional "
        "persistent subcaption/title line; '#ad' disclosure for 3s; a small creator-name line. "
        "It cannot add sound effects, zooms or emoji images yet; if top clips rely on those, "
        "say so in evidence and 'avoid' nothing on that account.\n\n"
        "Clips, most-viewed first (one JSON object per line):\n" + "\n".join(rows)
        + "\n\nWrite the style guide. Rules must stay inside the campaign's rules."
    )


def _write_manual_brief(d: Path, feats: list[dict], camp: Campaign) -> None:
    lines = [
        "# Tastemaker analysis (manual mode)",
        "",
        "No API key was found, so the analysis is done by hand (for example by Claude Code on this machine).",
        "",
        "## Step 1: analyse each clip",
        "",
        _analysis_system(),
        "",
        "For each clip below, look at `clips/<key>/grid.jpg` and `clips/<key>/frame0.jpg`, read the "
        "measurements, and write `analysis/<key>.json` matching this JSON schema:",
        "",
        "```json", json.dumps(ClipAnalysis.model_json_schema(), indent=1), "```",
        "",
    ]
    for f in sorted(feats, key=lambda f: f.get("views") or 0, reverse=True):
        lines += [f"### {f['key']}", "", _clip_brief(f), ""]
    lines += [
        "## Step 2: write the style guide",
        "",
        _guide_system(),
        "",
        "Then read every `analysis/*.json` with its measurements and write `style_guide.json` in this "
        "folder, matching this JSON schema (leave `based_on` empty; it is filled in):",
        "",
        "```json", json.dumps(StyleGuide.model_json_schema(), indent=1), "```",
        "",
        "Context for step 2:",
        "",
        _guide_brief(camp, feats, []).split("Clips, most-viewed first")[0],
        "",
        f"Finally run `python -m clipper taste report {camp.id}` to validate it and write style_guide.md.",
    ]
    (d / "ANALYZE.md").write_text("\n".join(lines))


# ---------------------------------------------------------------- report

def report(settings: Settings, camp: Campaign) -> Path:
    d = _dir(settings, camp)
    gp = guide_path(settings, camp)
    guide = StyleGuide.model_validate(json.loads(gp.read_text()))
    feats = json.loads((d / "features.json").read_text()) if (d / "features.json").exists() else []
    if not guide.based_on and feats:
        guide.based_on = _based_on(feats)
        gp.write_text(guide.model_dump_json(indent=2))
    c, h, s, p = guide.caption, guide.hook, guide.subcaption, guide.pacing
    md = [
        f"# Style guide: {camp.creator}", "", f"Based on {guide.based_on}.", "",
        "## What the clipper will do", "",
        f"- Length: about {p.target_s:.0f}s ({p.min_s:.0f}-{p.max_s:.0f}s); speech or action by {p.first_words_by_s:.1f}s.",
        f"- Captions: {c.words_per_chunk} words at a time, {'UPPERCASE' if c.uppercase else 'sentence case'}, "
        f"{c.font_size}px, {c.text_color} with {c.highlight_mode} highlight {c.highlight_color}, "
        f"{'at ' + format(c.y_frac, '.0%') + ' height' if c.y_frac else 'layout default position'}.",
        f"- Hook: {h.seconds:.1f}s, {h.style} style, {h.font_size}px.",
        f"- Subcaption: {'on, ' + s.style + ' style, ' + str(s.font_size) + 'px, from ' + format(s.from_s, '.1f') + 's' if s.enabled else 'off'}.",
        f"- Layout: {guide.layout_preference}.", "",
    ]
    for title, items in (("Moments to pick", guide.moment_rules), ("Hooks", guide.hook_rules),
                         ("Subcaptions", guide.subcaption_rules), ("Avoid", guide.avoid),
                         ("Evidence", guide.evidence)):
        if items:
            md += [f"## {title}", ""] + [f"- {r}" for r in items] + [""]
    if feats:
        md += ["## Clips analysed", "", "| Views | Length | Cuts/10s | 1st word | Words/s | Title |",
               "|---:|---:|---:|---:|---:|---|"]
        for f in sorted(feats, key=lambda f: f.get("views") or 0, reverse=True):
            md.append(f"| {f.get('views') or 0:,} | {f['duration_s']:.0f}s | {f['cuts_per_10s']} | "
                      f"{f['first_word_s']} | {f['words_per_s']} | [{(f.get('title') or '')[:60]}]({f.get('url')}) |")
    out = gp.with_suffix(".md")
    out.write_text("\n".join(md) + "\n")
    print(f"[taste] {out}")
    return out
