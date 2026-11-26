"""Stage 2: pull a VOD (plus metadata, chat replay and auto-captions) with yt-dlp."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

from . import db
from .config import Campaign, Settings, require_licensed


class NotOnAllowlist(RuntimeError):
    pass


def _ydl(opts: dict):
    import yt_dlp

    return yt_dlp.YoutubeDL({"quiet": True, "no_warnings": True, **opts})


def probe(url: str) -> dict:
    with _ydl({"skip_download": True}) as ydl:
        return ydl.sanitize_info(ydl.extract_info(url, download=False))


def ingest(
    settings: Settings,
    camp: Campaign,
    url: str,
    section: tuple[float, float] | None = None,
    force: bool = False,
) -> str:
    """Download one VOD for a licensed campaign. Returns the source id.

    `section` limits the download to [start, end] seconds of the VOD, which is
    handy for testing on a long stream. Times in the transcript, chat and clips
    are then relative to the section start; meta.json records the offset.
    """
    require_licensed(camp)
    info = probe(url)
    channel_urls = (info.get("channel_url"), info.get("uploader_url"))
    if info.get("extractor_key", "").lower().startswith("twitch") and info.get("uploader_id"):
        channel_urls = channel_urls + (f"https://www.twitch.tv/{info['uploader_id']}",)
    if not camp.channel_matches(*channel_urls):
        raise NotOnAllowlist(
            f"{url} belongs to {channel_urls[0] or channel_urls[1]!r}, which is not one of "
            f"campaign '{camp.id}' channels {camp.channels}. Refusing to download."
        )

    source_id = f"{info['extractor_key'].lower()}-{info['id']}"
    if section:
        source_id += f"-{int(section[0])}-{int(section[1])}"
    sdir = settings.work_dir / "sources" / source_id
    sdir.mkdir(parents=True, exist_ok=True)
    video = sdir / "source.mp4"

    con = db.connect(settings.work_dir)
    if video.exists() and not force:
        print(f"[ingest] {source_id}: already downloaded")
    else:
        opts = {
            "format": "bv*[height<=1080][ext=mp4]+ba[ext=m4a]/bv*[height<=1080]+ba/b[height<=1080]/b",
            "merge_output_format": "mp4",
            "outtmpl": {"default": str(sdir / "source.%(ext)s")},
            "writeinfojson": True,
            # Chat replay (YouTube) and auto-captions (fallback transcript).
            "writesubtitles": True,
            "writeautomaticsub": True,
            "subtitleslangs": ["live_chat", "en", "en-orig"],
            "subtitlesformat": "json3/best",
            "ignoreerrors": False,
            "noprogress": True,
            "quiet": False,
        }
        if section:
            from yt_dlp.utils import download_range_func

            opts["download_ranges"] = download_range_func(None, [section])
            opts["force_keyframes_at_cuts"] = True
        try:
            with _ydl(opts) as ydl:
                ydl.download([url])
        except Exception:
            # Subtitle/chat failures should not sink the video download; retry without them.
            opts.update(writesubtitles=False, writeautomaticsub=False)
            with _ydl(opts) as ydl:
                ydl.download([url])

    meta = {
        "url": url,
        "title": info.get("title"),
        "channel_url": channel_urls[0] or channel_urls[1],
        "duration": info.get("duration"),
        "offset": section[0] if section else 0.0,
        "section": list(section) if section else None,
        "was_live": info.get("was_live"),
    }
    (sdir / "meta.json").write_text(json.dumps(meta, indent=2))
    extract_audio(video, sdir / "audio.wav", force=force)

    db.upsert(con, "source", {
        "id": source_id, "campaign_id": camp.id, "url": url, "title": meta["title"],
        "channel_url": meta["channel_url"], "duration": probe_duration(video),
        "dir": str(sdir), "status": "ingested", "error": None,
    })
    print(f"[ingest] {source_id}: {meta['title']!r} -> {video}")
    return source_id


def ingest_local(settings: Settings, camp: Campaign, path: Path, source_id: str) -> str:
    """Register a file you already have (e.g. the creator sent you the VOD)."""
    require_licensed(camp)
    sdir = settings.work_dir / "sources" / source_id
    sdir.mkdir(parents=True, exist_ok=True)
    video = sdir / "source.mp4"
    if not video.exists():
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(path), "-c", "copy", str(video)], check=True)
    (sdir / "meta.json").write_text(json.dumps({"url": str(path), "title": path.stem, "offset": 0.0}, indent=2))
    extract_audio(video, sdir / "audio.wav")
    con = db.connect(settings.work_dir)
    db.upsert(con, "source", {
        "id": source_id, "campaign_id": camp.id, "url": str(path), "title": path.stem,
        "channel_url": None, "duration": probe_duration(video), "dir": str(sdir),
        "status": "ingested", "error": None,
    })
    return source_id


def extract_audio(video: Path, wav: Path, force: bool = False) -> None:
    if wav.exists() and not force:
        return
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-i", str(video), "-vn", "-ac", "1", "-ar", "16000",
         "-c:a", "pcm_s16le", str(wav)],
        check=True,
    )


def probe_duration(path: Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", str(path)],
        check=True, capture_output=True, text=True,
    ).stdout
    return float(json.loads(out)["format"]["duration"])


def probe_video(path: Path) -> dict:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=width,height,r_frame_rate", "-of", "json", str(path)],
        check=True, capture_output=True, text=True,
    ).stdout
    s = json.loads(out)["streams"][0]
    num, den = s["r_frame_rate"].split("/")
    return {"width": int(s["width"]), "height": int(s["height"]), "fps": float(num) / float(den or 1)}
