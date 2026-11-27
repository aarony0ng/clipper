"""Operating loop for the revenue strategist: queue, measure, calibrate, status.

Code only. Nothing here posts, submits, signs in or opens a posting site. It
builds the day's post sheet for Aaron, reads public view counts of posts he has
listed in the ledger, and moves the quality bar from results.

Files (all under work/revenue/ unless noted):
  accounts.yaml   our posting accounts, one per platform handle (template on first run)
  stops.yaml      paused campaigns and accounts, with reasons; the queue respects it
  ledger.csv      one row per post Aaron made (he adds clip_id, account, url, posted_at)
  scores.csv      tastemaker bar scores per clip (imported onto the clip rows)
  gates.json      calibrated quality bars per account and campaign
  allocation.json weekly split of clip slots across campaigns
  health.json     each account's health state
  queue-YYYY-MM-DD.md   the day's post sheet
campaigns.yaml (repo root): per-campaign `gate`, `reach_run`, `reach_ai_content_ok`
(see config.Campaign).

The plan this implements: docs/REVENUE_PLAN.md ("Autonomous operating plan").
"""
from __future__ import annotations

import csv
import json
import math
import re
import statistics
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path

import yaml

from . import db
from .config import Campaign, Settings

DEFAULT_GATE = 7.5
GATE_FLOOR = 7.0
GATE_STEP = 0.25
SPACING_H = 3
DAY_START_H = 10          # first slot of the day, local time
EXPIRE_DAYS = 1           # a clip waits one day in the queue, then expires
SUBMIT_WITHIN_MIN = 30

# state: (posts per day, minimum tastemaker score; None = the account's gate)
STATES = {
    "warmup": (1, 8.0),
    "healthy": (3, None),
    "watch": (1, 8.5),
    "recover": (1, 9.0),   # 0 for the first 48 h, see caps()
}
ORDER = ["recover", "watch", "healthy"]   # low to high; warmup sits outside
RECOVER_PAUSE_H = 48

LEDGER_FIELDS = ["clip_id", "campaign", "account", "platform", "url", "posted_at", "score",
                 "views_24h", "views_72h", "views_7d", "payout_usd"]


# ---------------------------------------------------------------- files

def rev_dir(settings: Settings) -> Path:
    d = settings.work_dir / "revenue"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _yaml(path: Path, default):
    if not path.exists():
        path.write_text(yaml.safe_dump(default, sort_keys=False))
    return yaml.safe_load(path.read_text()) or default


def load_accounts(settings: Settings) -> list[dict]:
    template = {"accounts": [
        {"id": "tiktok-main", "platform": "tiktok", "handle": "", "gate": DEFAULT_GATE,
         "campaigns": ["sypherpk-nov15"], "active": False},
        {"id": "reels-main", "platform": "instagram_reels", "handle": "", "gate": DEFAULT_GATE,
         "campaigns": ["sypherpk-nov15"], "active": False},
        {"id": "shorts-main", "platform": "youtube_shorts", "handle": "", "gate": DEFAULT_GATE,
         "campaigns": ["sypherpk-nov15"], "active": False},
        {"id": "x-main", "platform": "x", "handle": "", "gate": DEFAULT_GATE,
         "campaigns": ["sypherpk-nov15"], "active": False},
    ]}
    path = rev_dir(settings) / "accounts.yaml"
    if not path.exists():
        path.write_text("# Our posting accounts, one per platform handle. Set handle and active: true\n"
                        "# for the ones Aaron really posts from. gate = minimum tastemaker score (floor 7.0).\n"
                        + yaml.safe_dump(template, sort_keys=False))
    return (yaml.safe_load(path.read_text()) or {}).get("accounts", [])


def load_stops(settings: Settings) -> dict:
    path = rev_dir(settings) / "stops.yaml"
    if not path.exists():
        path.write_text("# Pauses the queue respects. Pause a campaign on any rejection or strike warning,\n"
                        "# an account on any reach-limit or unoriginal-content notice. Remove a line to resume.\n"
                        "# e.g. campaigns: {sypherpk-nov15: \"rejection 2026-11-30\"}\n"
                        + yaml.safe_dump({"campaigns": {}, "accounts": {}}, sort_keys=False))
    d = yaml.safe_load(path.read_text()) or {}
    return {"campaigns": d.get("campaigns") or {}, "accounts": d.get("accounts") or {}}


def read_ledger(settings: Settings) -> list[dict]:
    path = rev_dir(settings) / "ledger.csv"
    if not path.exists() or path.stat().st_size == 0:
        path.write_text(",".join(LEDGER_FIELDS) + "\n")
    with path.open() as f:
        rows = list(csv.DictReader(f))
    if rows and set(LEDGER_FIELDS) - set(rows[0].keys()):
        rows = [{k: r.get(k, "") for k in LEDGER_FIELDS} for r in rows]
    elif not rows and path.read_text().split("\n", 1)[0] != ",".join(LEDGER_FIELDS):
        path.write_text(",".join(LEDGER_FIELDS) + "\n")
    return rows


def write_ledger(settings: Settings, rows: list[dict]) -> None:
    with (rev_dir(settings) / "ledger.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=LEDGER_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in LEDGER_FIELDS})


def _json(path: Path, default):
    return json.loads(path.read_text()) if path.exists() else default


def _num(x) -> float | None:
    try:
        return float(x) if x not in (None, "") else None
    except ValueError:
        return None


def _ts(x: str) -> datetime | None:
    try:
        return datetime.fromisoformat(x) if x else None
    except ValueError:
        return None


# ---------------------------------------------------------------- health state machine

@dataclass
class Health:
    state: str = "warmup"
    since: str = ""            # ISO time the state was entered
    baseline: float | None = None
    last10_median: float | None = None
    posts: int = 0


def _views(p: dict) -> float | None:
    """Best available view count for a post: 7 d, else 72 h, else 24 h."""
    for k in ("views_7d", "views_72h", "views_24h"):
        v = _num(p.get(k))
        if v is not None:
            return v
    return None


def next_health(posts: list[dict], prev: Health, now: datetime) -> Health:
    """posts: this account's ledger rows. Pure function (tested)."""
    posts = sorted([p for p in posts if _ts(p.get("posted_at"))], key=lambda p: p["posted_at"])
    h = Health(state=prev.state, since=prev.since, posts=len(posts))
    if len(posts) < 10:
        return Health("warmup", prev.since if prev.state == "warmup" else now.isoformat(), None, None, len(posts))
    month = [_views(p) for p in posts if p.get("views_7d") and now - _ts(p["posted_at"]) <= timedelta(days=30)]
    month = [v for v in month if v is not None]
    if not month:  # no 7-day numbers yet: hold the state
        return h
    base = statistics.median(month)
    measured = [v for v in (_views(p) for p in posts) if v is not None]
    last10 = measured[-10:]
    med = statistics.median(last10)
    h.baseline, h.last10_median = base, med
    ratio = med / base if base else 1.0
    two_low = len(measured) >= 2 and all(v < 0.25 * base for v in measured[-2:])
    if ratio < 0.5:
        target = "recover"
    elif ratio < 1.0 or two_low:
        target = "watch"
    else:
        target = "healthy"
    cur = "healthy" if prev.state == "warmup" else prev.state
    if prev.state == "warmup":
        new = target                                   # first real reading after warm-up
    elif ORDER.index(target) < ORDER.index(cur):
        new = target                                   # down as soon as a trigger fires
    elif ORDER.index(target) > ORDER.index(cur):
        streak = measured[-5:]
        up = len(streak) == 5 and all(v >= base for v in streak)
        new = ORDER[ORDER.index(cur) + 1] if up else cur   # up one state after 5 in a row
    else:
        new = cur
    h.state = new
    h.since = prev.since if new == prev.state and prev.since else now.isoformat()
    return h


def caps(h: Health, gate: float, now: datetime) -> tuple[int, float]:
    """(posts allowed today, minimum score) for an account in health h."""
    per_day, min_score = STATES[h.state]
    min_score = max(gate, min_score) if min_score is not None else gate
    if h.state == "recover" and h.since and now - _ts(h.since) < timedelta(hours=RECOVER_PAUSE_H):
        per_day = 0
    return per_day, max(min_score, GATE_FLOOR)


# ---------------------------------------------------------------- queue builder

@dataclass
class Candidate:
    clip_id: str
    moment_id: str
    campaign: str
    score: float
    variant: int = 1
    first_seen: str = ""        # date the clip first became eligible (for expiry)
    post_md: str = ""


@dataclass
class Slot:
    account: str
    platform: str
    time: datetime
    clip: Candidate
    min_score: float


def blocked(campaign: Campaign | None, cid: str, stops: dict) -> str | None:
    if cid in stops["campaigns"]:
        return f"paused: {stops['campaigns'][cid]}"
    if campaign is None:
        return "unknown campaign"
    if not campaign.licensed:
        return "not licensed"
    if getattr(campaign, "reach_run", False) and not getattr(campaign, "reach_ai_content_ok", False):
        return "Reach-run campaign: blocked until Aaron sets reach_ai_content_ok: true"
    return None


def build_queue(day: date, candidates: list[Candidate], accounts: list[dict], health: dict[str, Health],
                gates: dict[str, float], campaigns: dict[str, Campaign], stops: dict, posted: set[tuple[str, str]],
                allocation: dict[str, float] | None = None) -> tuple[list[Slot], list[str]]:
    """Fill today's slots. Pure function (tested). Returns (slots, notes)."""
    notes = []
    now = datetime.combine(day, datetime.min.time()).replace(hour=DAY_START_H)
    fresh = []
    for c in candidates:
        if c.first_seen and (day - date.fromisoformat(c.first_seen)).days > EXPIRE_DAYS:
            notes.append(f"expired: {c.clip_id} (in the queue since {c.first_seen})")
            continue
        why = blocked(campaigns.get(c.campaign), c.campaign, stops)
        if why:
            notes.append(f"blocked: {c.clip_id}: {why}")
            continue
        fresh.append(c)
    fresh.sort(key=lambda c: (-c.score, c.clip_id))

    # per-campaign share of today's slots (weekly calibration), if any
    total_slots = 0
    acc_caps = {}
    for a in accounts:
        if not a.get("active", False):
            continue
        if a["id"] in stops["accounts"]:
            notes.append(f"account paused: {a['id']}: {stops['accounts'][a['id']]}")
            continue
        gate = max(GATE_FLOOR, gates.get(a["id"], a.get("gate", DEFAULT_GATE)))
        n, min_score = caps(health.get(a["id"], Health()), gate, now)
        acc_caps[a["id"]] = (n, min_score)
        total_slots += n
    camp_left = None
    if allocation:
        camp_left = {k: math.ceil(v * total_slots) for k, v in allocation.items()}

    slots: list[Slot] = []
    for a in accounts:
        if a["id"] not in acc_caps:
            continue
        n, min_score = acc_caps[a["id"]]
        used_moments, t = set(), now
        for c in fresh:
            if len([s for s in slots if s.account == a["id"]]) >= n:
                break
            if c.campaign not in a.get("campaigns", []):
                continue
            camp_gate = max(GATE_FLOOR, gates.get(c.campaign, getattr(campaigns.get(c.campaign), "gate", DEFAULT_GATE)))
            if c.score < max(min_score, camp_gate):
                continue
            if c.moment_id in used_moments or (a["id"], c.moment_id) in posted:
                continue
            if camp_left is not None and camp_left.get(c.campaign, 0) <= 0:
                continue
            slots.append(Slot(a["id"], a["platform"], t, c, max(min_score, camp_gate)))
            used_moments.add(c.moment_id)
            if camp_left is not None:
                camp_left[c.campaign] -= 1
            t += timedelta(hours=SPACING_H)
    if not slots:
        notes.append("no clip clears today's bar on any active account")
    return slots, notes


# ---------------------------------------------------------------- CLI steps

def import_scores(settings: Settings) -> int:
    """work/revenue/scores.csv (clip_id or moment id, avg) -> clip.score."""
    path = rev_dir(settings) / "scores.csv"
    if not path.exists():
        return 0
    con = db.connect(settings.work_dir)
    n = 0
    with path.open() as f:
        for r in csv.DictReader(f):
            key, score = r.get("clip_id") or r.get("moment"), _num(r.get("avg"))
            if not key or score is None:
                continue
            cur = con.execute("UPDATE clip SET score=? WHERE id=? OR moment_id=?", (score, key, key))
            n += cur.rowcount
    con.commit()
    return n


def _candidates(settings: Settings, state: dict, today: date) -> list[Candidate]:
    con = db.connect(settings.work_dir)
    rows = con.execute(
        "SELECT c.id, c.moment_id, c.variant, c.score, c.path, c.status, s.campaign_id FROM clip c "
        "JOIN moment m ON m.id=c.moment_id JOIN source s ON s.id=m.source_id "
        "WHERE c.status='rendered' AND c.score IS NOT NULL AND c.path IS NOT NULL").fetchall()
    out = []
    for r in rows:
        first = state.get(r["id"], "")   # set once the clip is first eligible (see queue())
        post_md = str(Path(r["path"]).with_suffix(".post.md"))
        out.append(Candidate(r["id"], r["moment_id"], r["campaign_id"], float(r["score"]), r["variant"], first, post_md))
    return out


def _health_all(settings: Settings, accounts: list[dict], ledger: list[dict], now: datetime) -> dict[str, Health]:
    path = rev_dir(settings) / "health.json"
    prev = {k: Health(**v) for k, v in _json(path, {}).items()}
    out = {}
    for a in accounts:
        posts = [r for r in ledger if r.get("account") == a["id"]]
        out[a["id"]] = next_health(posts, prev.get(a["id"], Health()), now)
    path.write_text(json.dumps({k: vars(v) for k, v in out.items()}, indent=1))
    return out


def queue(settings: Settings, camps: dict[str, Campaign], config_path: Path, day: date | None = None) -> Path:
    day = day or date.today()
    rd = rev_dir(settings)
    import_scores(settings)
    accounts, stops, ledger = load_accounts(settings), load_stops(settings), read_ledger(settings)
    health = _health_all(settings, accounts, ledger, datetime.now())
    gates = _json(rd / "gates.json", {})
    allocation = _json(rd / "allocation.json", {}).get("shares")
    qstate = _json(rd / "queue_state.json", {})
    cands = _candidates(settings, qstate, day)
    posted = {(r["account"], r["clip_id"].rsplit("-v", 1)[0]) for r in ledger if r.get("clip_id")}
    # The one-day expiry clock starts when a clip first becomes eligible (unblocked, at or
    # above the floor), so a clip held back by a pause isn't expired by the wait.
    for c in cands:
        if c.score >= GATE_FLOOR and not blocked(camps.get(c.campaign), c.campaign, stops) and not c.first_seen:
            c.first_seen = qstate[c.clip_id] = day.isoformat()
    slots, notes = build_queue(day, cands, accounts, health, gates, camps, stops, posted, allocation)
    (rd / "queue_state.json").write_text(json.dumps(qstate, indent=1))

    md = [f"# Post sheet {day.isoformat()}", "",
          "Post each clip at its time, then submit the post link on the campaign page before the "
          "submit-by time. Add a ledger row (clip_id, account, url, posted_at) right after posting.", ""]
    if not slots:
        md += ["**Nothing to post today.**", ""]
    for s in slots:
        pm = Path(s.clip.post_md)
        text = pm.read_text() if pm.exists() else ""
        hook = re.search(r"Hook on this variant \(v\d\): \*\*(.+?)\*\*", text)
        cap = re.search(r"## Caption to paste\s+```\n(.+?)\n```", text, re.S)
        sub = re.search(r"Submit each post's link at (\S+)", text)
        md += [f"## {s.time:%H:%M} · {s.account} ({s.platform})", "",
               f"- Clip: `{s.clip.clip_id}` (score {s.clip.score:.1f}, bar {s.min_score:.1f})",
               f"- File: {Path(s.clip.post_md).with_suffix('').with_suffix('.mp4')}",
               f"- Hook (v{s.clip.variant}): {hook.group(1) if hook else '?'}",
               f"- Caption: `{cap.group(1).strip() if cap else '?'}`",
               f"- Submit at {sub.group(1) if sub else '(see post checklist)'} by "
               f"**{(s.time + timedelta(minutes=SUBMIT_WITHIN_MIN)):%H:%M}**",
               f"- Checklist: {s.clip.post_md}", ""]
    if notes:
        md += ["## Notes", ""] + [f"- {n}" for n in notes] + [""]
    out = rd / f"queue-{day.isoformat()}.md"
    out.write_text("\n".join(md))
    print(f"[ops] {len(slots)} slot(s) -> {out}")
    for n in notes[:10]:
        print(f"[ops]   {n}")
    return out


def measure(settings: Settings, camps: dict[str, Campaign], now: datetime | None = None) -> int:
    """Public view counts for our posted URLs at 24 h / 72 h / 7 d (metadata only, no sign-in)."""
    import yt_dlp
    now = now or datetime.now()
    rows = read_ledger(settings)
    done = 0
    for r in rows:
        posted = _ts(r.get("posted_at"))
        if not posted or not r.get("url"):
            continue
        due = [k for k, h in (("views_24h", 24), ("views_72h", 72), ("views_7d", 168))
               if not r.get(k) and now - posted >= timedelta(hours=h)]
        if not due:
            continue
        try:
            with yt_dlp.YoutubeDL({"quiet": True, "no_warnings": True, "skip_download": True}) as y:
                info = y.extract_info(r["url"], download=False)
            views = info.get("view_count")
        except Exception as e:
            print(f"[ops] could not read {r['url']}: {str(e)[:100]}")
            continue
        if views is None:
            continue
        r[due[-1]] = str(views)
        for k in due[:-1]:
            r.setdefault(k, "")
        camp = camps.get(r.get("campaign", ""))
        if camp and due[-1] == "views_7d":
            r["payout_usd"] = f"{payout(camp, views):.2f}"
        done += 1
    write_ledger(settings, rows)
    _health_all(settings, load_accounts(settings), rows, now)
    print(f"[ops] measured {done} post(s)")
    return done


def payout(camp: Campaign, views: float) -> float:
    p = camp.payout or {}
    if views < float(p.get("min_views", 0)):
        return 0.0
    return min(float(p.get("max_payout", 1e9)), float(p.get("per_1k_views", 0)) * views / 1000)


def calibrate_gate(gate: float, scored: list[tuple[float, float]], median: float, min_n: int = 4) -> float:
    """scored: (score, 7-day views) for this account/campaign. Pure function (tested)."""
    band = [v for s, v in scored if 7.5 <= s < 8.0]
    if len(band) < min_n:
        return gate
    beat = sum(v > median for v in band) / len(band)
    if beat >= 0.5:
        gate -= GATE_STEP
    elif beat < 1 / 3:
        gate += GATE_STEP
    return max(GATE_FLOOR, round(gate, 2))


def allocate(per_campaign_usd: dict[str, list[float]], explore: float = 0.2, cap: float = 0.5) -> dict[str, float]:
    """Share of next week's slots per campaign by measured $/clip. Pure function (tested)."""
    avg = {k: (sum(v) / len(v) if v else 0.0) for k, v in per_campaign_usd.items()}
    pool = 1.0 - explore
    tot = sum(avg.values())
    if tot <= 0:
        n = max(1, len(avg))
        shares = {k: min(cap, pool / n) for k in avg}
    else:
        shares = {k: min(cap, pool * v / tot) for k, v in avg.items()}
    shares["_explore"] = round(1.0 - sum(shares.values()), 4)
    return {k: round(v, 4) for k, v in shares.items()}


def calibrate(settings: Settings, camps: dict[str, Campaign]) -> dict:
    rd = rev_dir(settings)
    rows = read_ledger(settings)
    gates = _json(rd / "gates.json", {})
    changes = {}
    for a in load_accounts(settings):
        posts = [r for r in rows if r.get("account") == a["id"] and r.get("views_7d") and r.get("score")]
        if not posts:
            continue
        med = statistics.median(float(r["views_7d"]) for r in posts)
        old = gates.get(a["id"], a.get("gate", DEFAULT_GATE))
        new = calibrate_gate(old, [(float(r["score"]), float(r["views_7d"])) for r in posts], med)
        if new != old:
            changes[a["id"]] = (old, new)
        gates[a["id"]] = new
    (rd / "gates.json").write_text(json.dumps(gates, indent=1))
    per_camp: dict[str, list[float]] = {}
    for r in rows:
        if r.get("campaign") and r.get("payout_usd"):
            per_camp.setdefault(r["campaign"], []).append(float(r["payout_usd"]))
    shares = allocate(per_camp) if per_camp else {}
    (rd / "allocation.json").write_text(json.dumps({"week_of": date.today().isoformat(), "shares": shares or None},
                                                   indent=1))
    print(f"[ops] gate changes: {changes or 'none'}; slot shares: {shares or 'not enough data'}")
    return {"gates": changes, "shares": shares}


def status(settings: Settings, camps: dict[str, Campaign], config_path: Path) -> None:
    stops, accounts = load_stops(settings), load_accounts(settings)
    ledger = read_ledger(settings)
    health = _health_all(settings, accounts, ledger, datetime.now())
    gates = _json(rev_dir(settings) / "gates.json", {})
    print("Accounts:")
    for a in accounts:
        h = health.get(a["id"], Health())
        gate = max(GATE_FLOOR, gates.get(a["id"], a.get("gate", DEFAULT_GATE)))
        n, m = caps(h, gate, datetime.now())
        flag = "" if a.get("active") else " (inactive)"
        flag += f" PAUSED: {stops['accounts'][a['id']]}" if a["id"] in stops["accounts"] else ""
        print(f"  {a['id']:14} {a['platform']:16} {h.state:8} posts={h.posts:3} today: {n} post(s), bar {m:.2f}{flag}")
    print("Campaigns:")
    for cid, c in camps.items():
        print(f"  {cid:22} {blocked(c, cid, stops) or 'open'}")
    import_scores(settings)
    q = _candidates(settings, {}, date.today())
    print(f"Scored clips: {len(q)}; at or above 7.5: {sum(c.score >= 7.5 for c in q)}")
    print(f"Ledger posts: {len(ledger)}")
