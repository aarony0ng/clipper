"""Campaign allowlist and pipeline settings.

Every source the pipeline touches must belong to a campaign in campaigns.yaml
with `licensed: true`. That flag is set by a person after reading the
campaign's terms on its page; the pipeline never sets it.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class Reframe:
    # auto: detect a facecam overlay or a talking head, else blur-pad
    # facecam_split: facecam on top, gameplay below (needs facecam box or detection)
    # track: 9:16 crop centred on the speaker's face
    # center: plain centre crop
    # blur_pad: full frame fitted to width over a blurred fill
    mode: str = "auto"
    # facecam box in source pixels [x, y, w, h]; skips detection when given
    facecam: list[int] | None = None
    # share of the 1920px height given to the facecam in facecam_split
    facecam_height_ratio: float = 0.38


@dataclass
class Campaign:
    id: str
    creator: str
    licensed: bool
    channels: list[str]
    campaign_url: str = ""
    rules_url: str = ""
    payout: dict[str, Any] = field(default_factory=dict)
    min_s: float = 15.0
    max_s: float = 60.0
    handle: str = ""
    hashtags: list[str] = field(default_factory=list)
    # Text burned into the first seconds of every clip, e.g. "#ad". The
    # Content Rewards guide treats a missing disclosure as grounds to reject.
    disclosure: str = ""
    disclosure_seconds: float = 3.0
    prohibited: list[str] = field(default_factory=list)
    brief: str = ""
    # Whole-word caption corrections for names Whisper mishears, e.g. {"Cypher": "Sypher"}
    caption_fixes: dict[str, str] = field(default_factory=dict)
    # Tastemaker: where to find top clips of similar content (yt-dlp URLs or
    # searches like "ytsearch60:sypherpk shorts"), and an optional guide path.
    taste_sources: list[str] = field(default_factory=list)
    style_guide: str = ""
    reframe: Reframe = field(default_factory=Reframe)
    # Revenue loop (clipper ops): minimum tastemaker score to post (never below 7.0), and
    # Reach-run campaigns stay blocked until Aaron confirms our edits aren't "AI content".
    gate: float = 7.5
    reach_run: bool = False
    reach_ai_content_ok: bool = False
    # Per-campaign extras (see campaigns.yaml for examples)
    logo_overlay: dict[str, Any] | None = None   # {path, corner: top-left|top-right|bottom-left|bottom-right, width}
    cta_text: str = ""                            # call to action burned in over the last seconds
    cta_seconds: float = 4.0
    title_on_screen: bool = False                 # the clip's concept stays on screen as a title
    mentions: list[str] = field(default_factory=list)   # @handles for the post caption
    stories_per_clip: bool = False                # campaign wants a story per posted clip
    min_account_age_h: float = 0.0                # an account must be this old before its first post
    min_followers: int = 0                        # ... and have this many followers, unless dedicated
    application_required: bool = False
    footage: str = ""                             # where the allowed source footage comes from
    notes: list[str] = field(default_factory=list)
    # Per-channel overrides for group streams where each POV has its own facecam spot,
    # keyed by channel login (e.g. {"replays": Reframe(...)}).
    reframe_by_channel: dict[str, Reframe] = field(default_factory=dict)

    def reframe_for(self, channel_url: str | None) -> Reframe:
        login = _norm_channel(channel_url or "").rsplit("/", 1)[-1]
        return self.reframe_by_channel.get(login, self.reframe)

    def channel_matches(self, *urls: str | None) -> bool:
        """True if any of the given channel/uploader URLs is on this campaign's list."""
        wanted = {_norm_channel(c) for c in self.channels}
        return any(u and _norm_channel(u) in wanted for u in urls)


def _norm_channel(url: str) -> str:
    u = url.strip().lower()
    u = re.sub(r"^https?://(www\.|m\.)?", "", u)
    u = re.sub(r"/(videos|streams|featured|live)/?$", "", u)
    return u.rstrip("/")


@dataclass
class Settings:
    work_dir: Path
    whisper_model: str = "small"
    whisper_device: str = "auto"
    # "auto": mlx-whisper on Apple Silicon when installed, else faster-whisper.
    transcribe_engine: str = "auto"
    mlx_model: str = "mlx-community/whisper-large-v3-turbo"
    language: str = "en"
    claude_model: str = "claude-opus-5-5"
    clips_per_source: int = 5
    variants_per_moment: int = 2
    caption_font: str = "Inter ExtraBold"
    caption_font_fallback: str = "DejaVu Sans"


def load(path: str | Path) -> tuple[Settings, dict[str, Campaign]]:
    path = Path(path)
    raw = yaml.safe_load(path.read_text()) or {}
    s = raw.get("settings", {}) or {}
    work = Path(s.pop("work_dir", "work"))
    if not work.is_absolute():
        work = (path.parent / work).resolve()
    settings = Settings(work_dir=work, **s)

    campaigns: dict[str, Campaign] = {}
    for c in raw.get("campaigns", []) or []:
        c = dict(c)
        clip = c.pop("clip", {}) or {}
        req = c.pop("required", {}) or {}
        rf = Reframe(**(c.pop("reframe", {}) or {}))
        rf_by = {k.lower(): Reframe(**v) for k, v in (c.pop("reframe_by_channel", {}) or {}).items()}
        camp = Campaign(
            id=c.pop("id"),
            creator=c.pop("creator"),
            licensed=bool(c.pop("licensed", False)),
            channels=list(c.pop("channels", []) or []),
            min_s=float(clip.get("min_s", 15)),
            max_s=float(clip.get("max_s", 60)),
            handle=req.get("handle", ""),
            hashtags=list(req.get("hashtags", []) or []),
            disclosure=req.get("disclosure", ""),
            disclosure_seconds=float(req.get("disclosure_seconds", 3.0)),
            reframe=rf,
            reframe_by_channel=rf_by,
            **c,
        )
        campaigns[camp.id] = camp
    return settings, campaigns


def guide_path(settings: "Settings", camp: Campaign) -> Path:
    if camp.style_guide:
        p = Path(camp.style_guide)
        return p if p.is_absolute() else settings.work_dir.parent / p
    return settings.work_dir / "taste" / camp.id / "style_guide.json"


class NotLicensed(RuntimeError):
    pass


def require_licensed(camp: Campaign) -> None:
    if not camp.licensed:
        raise NotLicensed(
            f"Campaign '{camp.id}' is not marked licensed: true. Read the campaign "
            f"terms at {camp.campaign_url or '(no URL set)'} and set the flag yourself "
            "before clipping this creator."
        )
