"""Command line: `python -m clipper <command>`."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import config, ingest, moments, render, signals, transcribe


def _section(s: str | None):
    if not s:
        return None
    a, b = s.split("-")

    def secs(x: str) -> float:
        parts = [float(p) for p in x.split(":")]
        return sum(p * 60 ** i for i, p in enumerate(reversed(parts)))

    return secs(a), secs(b)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="clipper", description="VOD -> ranked moments -> 9:16 clips")
    p.add_argument("-c", "--config", default="campaigns.yaml")
    sub = p.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("run", help="ingest + transcribe + signals + moments + render")
    a.add_argument("campaign")
    a.add_argument("url")
    a.add_argument("--section", help="only this part of the VOD, e.g. 1:02:00-1:32:00")
    a.add_argument("--engine", choices=["whisper", "youtube"], default="whisper")
    a.add_argument("--scenes", action="store_true", help="also detect scene cuts (slower)")
    a.add_argument("-n", type=int, help="moments to render")
    a.add_argument("--no-claude", action="store_true")

    a = sub.add_parser("ingest"); a.add_argument("campaign"); a.add_argument("url")
    a.add_argument("--section"); a.add_argument("--force", action="store_true")

    a = sub.add_parser("ingest-file", help="register a local video you have rights to")
    a.add_argument("campaign"); a.add_argument("path"); a.add_argument("--id", required=True)

    a = sub.add_parser("transcribe"); a.add_argument("source")
    a.add_argument("--engine", choices=["whisper", "youtube"], default="whisper")
    a.add_argument("--force", action="store_true")

    a = sub.add_parser("signals"); a.add_argument("source")
    a.add_argument("--scenes", action="store_true"); a.add_argument("--force", action="store_true")

    a = sub.add_parser("moments"); a.add_argument("source"); a.add_argument("-n", type=int)
    a.add_argument("--no-claude", action="store_true")
    a.add_argument("--manual", help="JSON list of {start,end,title,hooks,rationale} to use instead")

    a = sub.add_parser("render"); a.add_argument("source")
    a.add_argument("--only", type=int, nargs="*", help="moment ranks to render")
    a.add_argument("--force", action="store_true")
    a.add_argument("--no-guide", action="store_true", help="ignore the tastemaker's style guide")
    a.add_argument("--tag", default="", help="write to clips/<source>-<tag>/ (for side-by-side comparisons)")

    a = sub.add_parser("taste", help="tastemaker: learn a style guide from top clips")
    a.add_argument("step", choices=["collect", "measure", "analyze", "report", "all"])
    a.add_argument("campaign")
    a.add_argument("--source", action="append", help="yt-dlp URL or search; repeatable (default: campaign taste_sources)")
    a.add_argument("-n", type=int, default=24, help="clips to collect")
    a.add_argument("--max-dur", type=float, default=90.0)
    a.add_argument("--min-views", type=int, default=10_000)
    a.add_argument("--force", action="store_true")

    a = sub.add_parser("scan", help="scan several POV VODs of one group stream for the best moments")
    a.add_argument("config", help="scan config JSON (see clipper/scan.py)")
    a.add_argument("--top", type=int, default=24, help="peaks to shortlist")

    sub.add_parser("campaigns", help="list campaigns and whether they are licensed")

    args = p.parse_args(argv)
    settings, camps = config.load(args.config)

    def camp_for(source_id: str):
        from . import db
        return camps[db.get_source(db.connect(settings.work_dir), source_id)["campaign_id"]]

    if args.cmd == "campaigns":
        for c in camps.values():
            print(f"{c.id:28} {'LICENSED' if c.licensed else 'not licensed':13} {c.creator:20} {c.campaign_url}")
        return 0
    if args.cmd == "run":
        camp = camps[args.campaign]
        sid = ingest.ingest(settings, camp, args.url, _section(args.section))
        transcribe.transcribe(settings, sid, engine=args.engine)
        signals.compute(settings, sid, scenes=args.scenes)
        moments.rank(settings, camp, sid, n=args.n, use_claude=False if args.no_claude else None)
        render.render_source(settings, camp, sid)
        print(f"\nReview: {settings.work_dir / 'clips' / sid / 'review.html'}")
    elif args.cmd == "ingest":
        print(ingest.ingest(settings, camps[args.campaign], args.url, _section(args.section), force=args.force))
    elif args.cmd == "ingest-file":
        print(ingest.ingest_local(settings, camps[args.campaign], Path(args.path), args.id))
    elif args.cmd == "transcribe":
        transcribe.transcribe(settings, args.source, engine=args.engine, force=args.force)
    elif args.cmd == "signals":
        signals.compute(settings, args.source, scenes=args.scenes, force=args.force)
    elif args.cmd == "moments":
        moments.rank(settings, camp_for(args.source), args.source, n=args.n,
                     use_claude=False if args.no_claude else None,
                     manual=Path(args.manual) if args.manual else None)
    elif args.cmd == "render":
        render.render_source(settings, camp_for(args.source), args.source, force=args.force, only=args.only,
                             use_guide=not args.no_guide, tag=args.tag)
        name = f"{args.source}-{args.tag}" if args.tag else args.source
        print(f"Review: {settings.work_dir / 'clips' / name / 'review.html'}")
    elif args.cmd == "scan":
        from . import scan
        scan.run(settings, Path(args.config), top=args.top)
    elif args.cmd == "taste":
        from . import taste
        camp = camps[args.campaign]
        if args.step in ("collect", "all"):
            taste.collect(settings, camp, args.source, n=args.n, max_dur=args.max_dur, min_views=args.min_views)
        if args.step in ("measure", "all"):
            taste.measure(settings, camp, force=args.force)
        if args.step in ("analyze", "all"):
            taste.analyze(settings, camp, force=args.force)
        if args.step == "report":
            taste.report(settings, camp)
    return 0


if __name__ == "__main__":
    sys.exit(main())
