# Clipper: build step 1 (Stages 2–4)

Takes one VOD from a licensed clipping campaign and produces ranked, captioned
1080×1920 clips ready for a person to review. No posting, no submission: that
is build step 2 and 3 of the blueprint.

```
VOD URL ──► ingest (yt-dlp) ──► transcribe (faster-whisper, word timings)
        ──► signals (loudness, chat replay velocity, optional scene cuts)
        ──► moments (heuristic shortlist ──► Claude picks + writes hooks)
        ──► render (9:16 reframe, word-by-word captions, hook, #ad, -14 LUFS)
        ──► work/clips/<source>/review.html
```

## Setup (on your computer)

Needs Python 3.10+ (3.12 is safest for faster-whisper) and an ffmpeg built with
libass, which the captions need. Homebrew's plain `ffmpeg` no longer has it, so on a Mac:

```bash
brew install ffmpeg-full
export PATH="/opt/homebrew/opt/ffmpeg-full/bin:$PATH"   # put this in every new shell
ffmpeg -hide_banner -filters | grep -q " ass " && echo "libass OK"
```

```bash
cd clipper
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export ANTHROPIC_API_KEY=...        # optional; without it ranking is heuristic only
```

The first transcription downloads the Whisper model (~500 MB for `small`).

## Before the first run: pick and license a campaign

Open `campaigns.yaml`. For the campaign you want (ArthurTV, Bobby VanRiper and
Lauren Hamden are stubbed from the blueprint's addendum):

1. Paste the campaign page URL and the creator's channel URL(s).
2. Read the campaign terms and remaining budget on that page.
3. Paste the brief, prohibited content and required handle/hashtags.
4. Set `licensed: true`.

The pipeline refuses to download anything whose channel isn't on a licensed
campaign's list. `python -m clipper campaigns` shows what is licensed.

## Run

```bash
# Whole pipeline, on a 30-minute slice of a long stream (good first test)
python -m clipper run arthurtv-gta6 "https://www.youtube.com/watch?v=..." --section 1:00:00-1:30:00

# Or stage by stage (each is idempotent; --force redoes it)
python -m clipper ingest arthurtv-gta6 URL [--section a-b]
python -m clipper transcribe <source-id> [--engine youtube]   # youtube = use auto-captions, instant
python -m clipper signals <source-id> [--scenes]
python -m clipper moments <source-id> [-n 5] [--no-claude] [--manual picks.json]
python -m clipper render <source-id> [--only 1 3] [--force]
```

Open `work/clips/<source-id>/review.html` to watch every clip next to why it
was picked. Each clip also gets a `.jpg` contact sheet and automatic checks
(size, duration, audio, black or frozen frames, loudness) in `work/state.db`.

## Knobs worth knowing

- `reframe.mode`: `auto` detects a small fixed facecam (→ cam on top,
  gameplay below), a talking head (→ crop that follows the face), or neither
  (→ full frame over a blurred fill). If a creator's facecam is always in the
  same place, set `facecam: [x, y, w, h]` in source pixels for a reliable split.
- `whisper_model`: `small` is a good CPU default; `large-v3` is noticeably
  better on shouting and slang if you have a GPU or patience.
- `variants_per_moment: 2` renders an A and B per moment: different hook text,
  both open on the same spoken line.
- Without an API key, `moments` saves the exact ranking prompt to
  `work/sources/<id>/rank_prompt.md`, so you can paste it into Claude and feed
  the answer back with `--manual`.

## Tastemaker: learn the style from top clips

The tastemaker studies the most-viewed short clips of similar content and writes
a style guide that the clipper then follows: what moments to pick, how to write
hooks and subcaptions, and how captions should look.

```bash
python -m clipper taste all sypherpk-nov15      # collect -> measure -> analyze
python -m clipper taste report sypherpk-nov15   # re-validate and rewrite style_guide.md
```

1. **collect** searches the campaign's `taste_sources` (channel Shorts tabs and
   yt-dlp searches) and downloads the top `-n` clips under 90 s by views.
2. **measure** records length, scene cuts per 10 s, time to first word, words
   per second, loudness, the opening line, and a 12-frame grid of each clip.
3. **analyze** has Claude look at each clip's frames and transcript and
   describe its hook, caption style, subcaption, layout and edits. It then
   weights the results by views and writes
   `work/taste/<campaign>/style_guide.json` and `style_guide.md`. Without an API
   key, it writes `ANALYZE.md` so Claude Code can do this step by hand.

After that, `moments` adds the guide's rules to the ranking prompt and asks for
a subcaption for each moment. `render` styles captions, hooks and subcaptions
from the guide. Use `render <id> --no-guide --tag plain` and
`render <id> --tag guide` to compare clips with and without the guide.

Downloaded reference clips are for analysis only and stay in `work/taste`.

## What is not built yet

Stage 5 vision QA and perceptual-hash dedup, posting (Stage 6), campaign
discovery and submission (Stages 1 and 7), and the scheduled orchestrator.
Twitch chat replay isn't read (yt-dlp doesn't fetch it); YouTube's is.
