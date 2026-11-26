"""The style guide the tastemaker writes and the clipper follows.

Every field has a default that matches the clipper's original house style, so
a missing or partial guide never breaks a render.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


class CaptionStyle(BaseModel):
    """Word-by-word spoken captions."""
    words_per_chunk: int = Field(3, ge=1, le=8, description="Words on screen at once")
    max_chunk_s: float = Field(1.4, ge=0.4, le=4.0, description="Longest a chunk stays up")
    uppercase: bool = True
    font_size: int = Field(86, ge=40, le=140, description="Pixels on a 1080x1920 frame")
    y_frac: float | None = Field(None, ge=0.1, le=0.85,
                                 description="Vertical centre as a fraction of frame height; null = layout default")
    text_color: str = Field("#FFFFFF", description="#RRGGBB")
    highlight_mode: Literal["color", "box", "none"] = "color"
    highlight_color: str = Field("#FFE500", description="#RRGGBB for the word being spoken")
    outline_px: int = Field(7, ge=0, le=16)
    outline_color: str = Field("#000000", description="#RRGGBB of the text outline")
    pop_in: bool = Field(True, description="Small scale-up when a chunk appears")


class HookStyle(BaseModel):
    """Big text over the first seconds that stops the scroll."""
    seconds: float = Field(2.2, ge=0.0, le=6.0)
    font_size: int = Field(74, ge=40, le=140)
    style: Literal["box", "outline"] = "box"
    text_color: str = "#111111"
    box_color: str = "#FFFFFF"
    y_frac: float | None = Field(None, ge=0.05, le=0.85)


class SubcaptionStyle(BaseModel):
    """Persistent title line (e.g. 'Sypher got griefed by a rock 😭') separate from spoken captions."""
    enabled: bool = False
    font_size: int = Field(56, ge=30, le=110)
    style: Literal["box", "outline"] = "box"
    text_color: str = "#111111"
    box_color: str = "#FFFFFF"
    y_frac: float | None = Field(None, ge=0.03, le=0.9,
                                 description="null = top of the frame, under the platform UI")
    max_words: int = Field(9, ge=2, le=16)
    from_s: float = Field(0.0, ge=0.0, description="When it appears (after the hook, usually)")


class Pacing(BaseModel):
    min_s: float = 15.0
    max_s: float = 60.0
    target_s: float = 30.0
    first_words_by_s: float = Field(0.5, description="Speech or action should start by this time")


class StyleGuide(BaseModel):
    caption: CaptionStyle = CaptionStyle()
    hook: HookStyle = HookStyle()
    subcaption: SubcaptionStyle = SubcaptionStyle()
    pacing: Pacing = Pacing()
    layout_preference: str = Field("auto", description="auto | facecam_split | track | center | blur_pad")
    moment_rules: list[str] = Field(default_factory=list, description="What to pick, for the moment picker")
    hook_rules: list[str] = Field(default_factory=list, description="How to write on-screen hooks")
    subcaption_rules: list[str] = Field(default_factory=list, description="How to write the persistent title line")
    avoid: list[str] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list,
                                description="Short observations tying each rule to the analysed clips")
    based_on: str = Field("", description="e.g. '24 clips, 61M total views, collected 2026-11-26'")

    def picker_brief(self) -> str:
        """Text appended to the moment picker's prompt."""
        if not (self.moment_rules or self.hook_rules or self.subcaption_rules or self.avoid):
            return ""
        out = [f"<style_guide based_on=\"{self.based_on}\">",
               "These rules come from analysing the best-performing clips of similar content. Follow them.",
               f"Target length about {self.pacing.target_s:.0f}s; speech or action by "
               f"{self.pacing.first_words_by_s:.1f}s."]
        for title, items in (("Pick moments that", self.moment_rules), ("Hooks", self.hook_rules),
                             ("Subcaption (persistent title line)", self.subcaption_rules),
                             ("Avoid", self.avoid)):
            if items:
                out.append(f"{title}:")
                out += [f"- {r}" for r in items]
        out.append("</style_guide>")
        return "\n".join(out)


def load(path: Path | str | None) -> StyleGuide:
    if path and Path(path).exists():
        return StyleGuide.model_validate(json.loads(Path(path).read_text()))
    return StyleGuide()


def ass_color(hex_rgb: str, alpha: int = 0) -> str:
    """#RRGGBB -> ASS &HAABBGGRR."""
    h = hex_rgb.lstrip("#")
    if len(h) != 6:
        h = "FFFFFF"
    r, g, b = h[0:2], h[2:4], h[4:6]
    return f"&H{alpha:02X}{b}{g}{r}".upper()
