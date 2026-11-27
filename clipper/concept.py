"""Big-idea gate: every clip must be sayable as one short concept.

The top Shorts the tastemaker studied all sell one plain idea in the first second,
built on a famous name plus something that makes it matter ("Ninja accidentally
went live", "Peterbot is now #2"). A moment that can't be summed up like that in
five words or fewer is not worth rendering.

A concept passes when it:
- has 1 to 5 words,
- names a creator from the campaign,
- has a superlative, a number, or clear stakes/payoff (a win, a loss, a tie, a record...).
"""
from __future__ import annotations

import re

from .config import Campaign

SUPERLATIVE = re.compile(
    r"\b(first|last|only|ever|best|worst|biggest|smallest|craziest|rarest|fastest|slowest|"
    r"longest|shortest|luckiest|unluckiest|cleanest|hardest|easiest|funniest|weirdest|"
    r"wildest|most|least|perfect|impossible|insane|unbeatable|legendary|\w{3,}iest)\b", re.I)
NUMBER = re.compile(
    r"\d|\b(one|two|three|four|five|six|seven|eight|nine|ten|hundred|thousand|million|billion|"
    r"single|double|triple)\b", re.I)
STAKES = re.compile(
    r"\b(wins?|won|winning|winner|beats?|beat|lost|loses?|losing|tied?|ties|record|"
    r"hole-in-one|clutch|steals?|stole|escapes?|outruns?|outrun|vs|versus|knocked|eliminated|"
    r"comeback|rage|fails?|failed|saves?|saved|destroys?|destroyed|robbed|griefed|trolled|"
    r"caught|catches|chased|chasing|leads|leading)\b", re.I)


def creator_names(camp: Campaign) -> set[str]:
    """Names that count as 'a creator': from the creator line, the on-screen handle and channels."""
    names = set()
    for part in re.split(r"[^A-Za-z0-9_]+", f"{camp.creator} {camp.handle}"):
        if len(part) >= 4 and part.lower() not in {"with", "and", "the"}:
            names.add(part.lower())
    for ch in camp.channels:
        names.add(ch.rstrip("/").rsplit("/", 1)[-1].lower())
    # common short forms: "Maddy" for Maddynf, "Sypher" for SypherPK
    for n in list(names):
        for suffix in ("pk", "nf", "tv", "ttv", "live"):
            if n.endswith(suffix) and len(n) - len(suffix) >= 4:
                names.add(n[: -len(suffix)])
    if "maddyn" in names or "maddynf" in names:
        names.add("maddy")
    return names


def check(concept: str, camp: Campaign) -> tuple[bool, str]:
    """(passes, reason)."""
    c = (concept or "").strip()
    if not c:
        return False, "no concept: the moment can't be summed up as one big idea"
    words = c.replace("—", " ").split()
    if len(words) > 5:
        return False, f"concept is {len(words)} words; it must be 5 or fewer"
    low = re.sub(r"[^a-z0-9' -]", " ", c.lower())
    tokens = {t.split("'")[0] for t in low.split()}
    names = creator_names(camp)
    if not tokens & names:
        return False, f"concept names no creator (expected one of: {', '.join(sorted(names))})"
    if not (SUPERLATIVE.search(c) or NUMBER.search(c) or STAKES.search(c)):
        return False, "concept has no superlative, number or clear stakes/payoff"
    return True, "ok"


def hooks_from(concept: str, hooks: list[str]) -> list[str]:
    """Variant 1's hook is the concept itself; other hooks follow as alternative angles."""
    first = concept.strip().upper()
    key = lambda h: re.sub(r"[^A-Z0-9 ]", "", h.upper()).strip()
    rest = [h for h in hooks or [] if key(h) != key(first)]
    return [first] + rest
