"""Stage 3c: find candidate moments, score them on signals, let Claude pick.

1. Candidates: every window that starts and ends on a transcript segment
   boundary and fits the campaign's length limits.
2. Heuristic score: loudness peaks, chat spikes, scene cuts, hype words,
   speech rate, hook strength in the first 3 seconds, minus dead air.
3. Non-max suppression keeps the best non-overlapping windows.
4. Claude re-ranks the shortlist against the campaign brief and writes hooks.
   Without ANTHROPIC_API_KEY the heuristic ranking is used as-is, and the
   prompt is saved so you can run it by hand.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from . import concept as big_idea, db, signals as sig, transcribe
from .config import Campaign, Settings, guide_path
from .style import StyleGuide, load as load_guide

HYPE = re.compile(
    r"\b(no way|what+|bro+|yo+|oh my god|omg|let'?s go+|holy|insane|crazy|wtf|"
    r"bruh|dude|chat|look at|watch this|i can'?t|are you serious|why|run|get him|"
    r"oh no|no no|hell|damn|nah+)\b", re.I)
LAUGH = re.compile(r"\[(laughter|laughs|laughing)\]|\b(ha){2,}\b|\blmao\b|\blol\b", re.I)


@dataclass
class Candidate:
    start: float
    end: float
    score: float
    features: dict = field(default_factory=dict)
    text: str = ""

    @property
    def dur(self) -> float:
        return self.end - self.start


def _robust_z(x: np.ndarray) -> np.ndarray:
    med = np.median(x)
    # Floor of 1 unit (1 dB, or 1 chat msg/s) so a flat signal can't explode the z-scores.
    mad = max(float(np.median(np.abs(x - med)) * 1.4826), 1.0)
    return (x - med) / mad


def _smooth(x: np.ndarray, k: int) -> np.ndarray:
    if k <= 1:
        return x
    return np.convolve(x, np.ones(k) / k, mode="same")


def build_candidates(segments: list[dict], s: dict, camp: Campaign) -> list[Candidate]:
    loud = np.array(s["loudness_db"])
    n = len(loud)
    loud_z = _robust_z(loud)
    chat_z = _robust_z(_smooth(np.array(s["chat"]), 5)) if s.get("chat") else np.zeros(n)
    scene = np.array(s["scene"]) if s.get("scene") else np.zeros(n)
    silent = loud < max(-45.0, float(np.percentile(loud, 10)))

    def window(a: float, b: float) -> slice:
        return slice(max(0, int(a)), min(n, int(np.ceil(b))))

    cands = []
    for i, first in enumerate(segments):
        text_parts = []
        for j in range(i, len(segments)):
            last = segments[j]
            text_parts.append(last["text"])
            a, b = first["s"], last["e"]
            if b - a > camp.max_s:
                break
            if b - a < camp.min_s:
                continue
            w, hook = window(a, b), window(a, a + 3)
            if w.stop - w.start < 2:
                continue
            text = " ".join(text_parts)
            words = sum(len(segments[k]["words"]) for k in range(i, j + 1))
            lz = np.sort(loud_z[w])[::-1]
            f = {
                "loud_peak": float(lz[0]),
                "loud_top3": float(lz[:3].mean()),
                "chat_peak": float(chat_z[w].max()),
                "scene_rate": float(scene[w].sum() / (b - a)),
                "hype": len(HYPE.findall(text)) / (b - a) * 10,
                "laugh": len(LAUGH.findall(text)),
                "exclaim": text.count("!") / (b - a) * 10,
                "wps": words / (b - a),
                "hook_loud": float(loud_z[hook].max()) if hook.stop > hook.start else 0.0,
                "hook_hype": len(HYPE.findall(first["text"])),
                "silence": float(silent[w].mean()),
                "ends_clean": float(bool(re.search(r"[.!?]$", last["text"].strip()))),
            }
            score = (
                0.9 * np.clip(f["loud_peak"], 0, 6) + 0.6 * np.clip(f["loud_top3"], 0, 5)
                + 1.0 * np.clip(f["chat_peak"], 0, 6) + 0.6 * min(f["scene_rate"] * 10, 3)
                + 0.5 * min(f["hype"], 4) + 0.8 * min(f["laugh"], 3) + 0.3 * min(f["exclaim"], 3)
                + 0.4 * np.clip(f["wps"] - 2.0, 0, 2)
                + 0.8 * np.clip(f["hook_loud"], 0, 4) + 0.5 * min(f["hook_hype"], 2)
                - 4.0 * f["silence"] + 0.4 * f["ends_clean"]
                - 0.02 * abs((b - a) - 35)  # mild preference for ~35s
            )
            cands.append(Candidate(a, b, float(score), f, text))
    return cands


def suppress(cands: list[Candidate], k: int, max_overlap: float = 0.25) -> list[Candidate]:
    picked: list[Candidate] = []
    for c in sorted(cands, key=lambda c: c.score, reverse=True):
        if all(_overlap(c, p) <= max_overlap for p in picked):
            picked.append(c)
            if len(picked) >= k:
                break
    return picked


def _overlap(a: Candidate, b: Candidate) -> float:
    inter = max(0.0, min(a.end, b.end) - max(a.start, b.start))
    return inter / min(a.dur, b.dur)


def _fmt(t: float) -> str:
    return f"{int(t // 3600)}:{int(t % 3600 // 60):02d}:{t % 60:05.2f}"


def build_prompt(camp: Campaign, shortlist: list[Candidate], segments: list[dict], n: int,
                 guide: StyleGuide | None = None) -> tuple[str, str]:
    system = (
        "You are the moment picker in a short-form clipping pipeline. You read stream "
        "transcripts and choose the moments most likely to perform as 9:16 vertical clips "
        "on TikTok, YouTube Shorts and Instagram Reels, for a clipping campaign the creator "
        "has licensed. You also write the on-screen hook text shown in the first 2 seconds.\n\n"
        "What makes a good pick: it is self-contained (a viewer with no context gets it), "
        "the first 2 seconds hook (a reaction, a bold line, a question, visible chaos), it has "
        "a payoff before it ends, and it fits the campaign brief. Prefer genuine reactions, "
        "jokes, arguments, reveals, wins and fails over setup or menu time. Reject anything "
        "the campaign prohibits and anything that is not advertiser-safe.\n\n"
        "Hook text is 2 to 6 words, written like a caption a person would actually post, "
        "and never misrepresents what happens. Give one hook per variant and make the two "
        "hooks take different angles (e.g. a quote from the clip versus a tease of the payoff).\n\n"
        "Every pick needs a `concept`: the clip's one big idea in 5 words or fewer, naming a "
        "creator plus a superlative, a number, or clear stakes or payoff (e.g. \"Sypher's first "
        "ever hole-in-one\"). If a moment can't be summed up like that, don't pick it. The first "
        "hook is the concept; the second takes another angle."
    )
    rules = [
        f"Creator: {camp.creator}",
        f"Clip length: {camp.min_s:.0f}-{camp.max_s:.0f} seconds",
        f"Prohibited: {', '.join(camp.prohibited) or 'nothing listed'}",
        f"Brief: {camp.brief.strip() or '(none)'}",
    ]
    blocks = []
    for idx, c in enumerate(shortlist):
        lines = [f"[{_fmt(sg['s'])}] {sg['text']}" for sg in segments
                 if sg["e"] > c.start - 8 and sg["s"] < c.end + 8]
        f = c.features
        blocks.append(
            f'<candidate id="{idx}" start="{c.start:.2f}" end="{c.end:.2f}" '
            f'signal_score="{c.score:.2f}" loudness_peak_z="{f["loud_peak"]:.1f}" '
            f'chat_peak_z="{f["chat_peak"]:.1f}" laughs="{f["laugh"]}">\n'
            + "\n".join(lines) + "\n</candidate>"
        )
    user = (
        "<campaign>\n" + "\n".join(rules) + "\n</campaign>\n\n"
        "Each candidate below is a window from the stream. The transcript includes about "
        "8 seconds of context on each side; times are seconds from the start of the video. "
        "Signal fields are automatic: loudness and chat-velocity peaks as z-scores.\n\n"
        + "\n\n".join(blocks)
        + f"\n\nPick up to {n} candidates, best first. For each, give tightened start and end "
        "times (you may move each edge up to 8 seconds, landing on a line boundary, so the clip "
        "opens on the hook and ends right after the payoff), keeping it within the length "
        "limits. Skip candidates that are weak; fewer strong picks beat padding the list."
    )
    if guide is not None:
        brief = guide.picker_brief()
        if brief:
            user = brief + "\n\n" + user
        if guide.subcaption.enabled:
            user += (" Also write a `subcaption` for each pick: the persistent title line shown "
                     f"under the hook for the rest of the clip, at most {guide.subcaption.max_words} words, "
                     "following the subcaption rules.")
    return system, user


def rank_with_claude(settings: Settings, system: str, user: str, n: int) -> list[dict]:
    import anthropic
    from pydantic import BaseModel, Field

    class Pick(BaseModel):
        candidate_id: int
        start: float
        end: float
        title: str = Field(description="Short internal title for the moment")
        concept: str = Field(description="The clip's big idea: <=5 words, a creator name plus a "
                                         "superlative, number, or clear stakes/payoff")
        hooks: list[str] = Field(description="On-screen hook text, one per variant")
        subcaption: str = Field("", description="Persistent title line, if the style guide asks for one")
        rationale: str = Field(description="One or two sentences on why it will perform")
        score: float = Field(description="0-10 predicted strength")

    class Picks(BaseModel):
        picks: list[Pick]

    client = anthropic.Anthropic()
    with client.beta.messages.stream(
        model=settings.claude_model,
        max_tokens=32000,
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
        thinking={"type": "adaptive"},
        output_config={"effort": "high"},
        output_format=Picks,
        system=system,
        messages=[{"role": "user", "content": user}],
    ) as stream:
        msg = stream.get_final_message()
    if msg.stop_reason == "refusal":
        raise RuntimeError(f"Claude declined to rank: {msg.stop_details}")
    if msg.stop_reason == "max_tokens":
        raise RuntimeError("Ranking response hit max_tokens before finishing")
    return [p.model_dump() for p in msg.parsed_output.picks[:n]]


def _snap(t: float, segments: list[dict], edge: str) -> float:
    """Move a time to the nearest word boundary and add a small handle."""
    words = [w for sg in segments for w in sg["words"]]
    if not words:
        return t
    if edge == "start":
        w = min(words, key=lambda w: abs(w["s"] - t))
        return max(0.0, w["s"] - 0.12)
    w = min(words, key=lambda w: abs(w["e"] - t))
    return w["e"] + 0.35


def _default_hooks(c: Candidate, segments: list[dict]) -> list[str]:
    inside = [sg for sg in segments if sg["s"] >= c.start - 0.01 and sg["e"] <= c.end + 0.01]
    best = max(inside, key=lambda sg: len(HYPE.findall(sg["text"])) + sg["text"].count("!"), default=None)
    quote = ""
    if best:
        words = re.sub(r"[^\w' ?!]", "", best["text"]).split()
        quote = " ".join(words[:6]).upper()
    return [quote or "WAIT FOR IT", "WATCH TILL THE END"]


def rank(settings: Settings, camp: Campaign, source_id: str, n: int | None = None,
         use_claude: bool | None = None, manual: Path | None = None) -> list[dict]:
    con = db.connect(settings.work_dir)
    src = db.get_source(con, source_id)
    sdir = Path(src["dir"])
    segments = transcribe.load(sdir)
    n = n or settings.clips_per_source

    if manual:
        picks = json.loads(Path(manual).read_text())
        ranked_by = "manual"
    else:
        s = sig.load(sdir)
        cands = build_candidates(segments, s, camp)
        if not cands:
            raise RuntimeError("No candidate windows fit the campaign length limits.")
        shortlist = suppress(cands, k=max(3 * n, 12))
        (sdir / "candidates.json").write_text(json.dumps([asdict(c) for c in shortlist], indent=1))
        system, user = build_prompt(camp, shortlist, segments, n,
                                    guide=load_guide(guide_path(settings, camp)))
        (sdir / "rank_prompt.md").write_text(f"# System\n\n{system}\n\n# User\n\n{user}\n")

        if use_claude is None:
            use_claude = bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"))
        if use_claude:
            raw = rank_with_claude(settings, system, user, n)
            picks = []
            for p in raw:
                c = shortlist[p["candidate_id"]]
                # Keep Claude's edits within the promised 8 s of the candidate.
                p["start"] = float(np.clip(p["start"], c.start - 8, c.start + 8))
                p["end"] = float(np.clip(p["end"], c.end - 8, c.end + 8))
                picks.append(p)
            ranked_by = "claude"
        else:
            print("[moments] No Anthropic credentials found; using heuristic ranking. "
                  f"Prompt saved to {sdir / 'rank_prompt.md'}")
            picks = [{"start": c.start, "end": c.end, "score": c.score,
                      "title": c.text[:60], "hooks": _default_hooks(c, segments),
                      "rationale": "heuristic: " + ", ".join(
                          f"{k}={v:.1f}" for k, v in c.features.items() if k in
                          ("loud_peak", "chat_peak", "hype", "laugh", "hook_loud"))}
                     for c in shortlist[:n]]
            ranked_by = "heuristic"

    # Big-idea gate: no concept, no clip.
    kept = []
    for p in picks:
        ok, why = big_idea.check(p.get("concept", ""), camp)
        if ok:
            p["hooks"] = big_idea.hooks_from(p["concept"], p.get("hooks") or [])
            kept.append(p)
        else:
            print(f"[moments] rejected {p.get('title', '')[:50]!r}: {why}")
    picks = kept

    con.execute("DELETE FROM moment WHERE source_id=?", (source_id,))
    out = []
    for rank_i, p in enumerate(picks, 1):
        start = _snap(p["start"], segments, "start")
        end = _snap(p["end"], segments, "end")
        if end - start > camp.max_s:
            end = start + camp.max_s
        row = {
            "id": f"{source_id}-m{rank_i}", "source_id": source_id, "rank": rank_i,
            "start": round(start, 3), "end": round(end, 3), "score": p.get("score"),
            "title": p.get("title", ""), "hooks": p.get("hooks") or ["WATCH THIS"],
            "rationale": p.get("rationale", ""), "ranked_by": ranked_by, "status": "picked",
            "subcaption": p.get("subcaption", ""), "concept": p.get("concept", ""),
        }
        db.upsert(con, "moment", row)
        out.append(row)
    (sdir / "moments.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    db.set_status(con, "source", source_id, "ranked")
    for m in out:
        print(f"[moments] #{m['rank']} {_fmt(m['start'])}-{_fmt(m['end'])} "
              f"({m['end'] - m['start']:.0f}s) {m['title'][:50]!r} hooks={m['hooks']}")
    return out
