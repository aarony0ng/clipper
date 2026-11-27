# Revenue plan: views without wrecking the account

Owner: the revenue strategist (`.claude/agents/revenue-strategist.md`), with the
tastemaker as the quality gate. Implemented in `clipper/ops.py` (`clipper ops ...`).

## The objective

Maximize net clipping revenue per hour of Aaron's time, over months:

    revenue per clip = sum over platforms of  min(cap, CPM × views / 1000)   if views ≥ the campaign minimum, else 0

Two facts drive everything:

1. **Payouts are all-or-nothing at the minimum** (10,000 views on the Sypher campaign).
   A clip at 9,000 views earns $0. Many mediocre posts earn nothing; a few strong ones earn
   most of the money.
2. **The account is the asset.** TikTok, Reels and Shorts first show a post to a small
   test audience, then push it further only if people watch to the end, rewatch and share.
   An account whose recent posts lose that test gets smaller test audiences next time. Weak
   posts don't just earn $0; they make the next good post earn less.

So the strategy is **fewer, stronger posts, spread across every allowed platform**, not more posts.

## The loop (runs every day there's new footage)

1. **Find money.** Rank open campaigns by expected value: CPM × chance a clip clears the
   minimum × remaining budget share, adjusted for source energy (how big and lively the
   stream is), allowed platforms and competition. Drop campaigns under the floor in the policy.
2. **Find moments.** `clipper scan` the eligible windows (all POVs, viewer clips, chat once it's
   on). Take the top peaks.
3. **Big-idea gate.** Every moment needs a concept of 5 words or fewer naming a creator plus a
   superlative, a number or stakes. No concept, no render.
4. **Render** with the style guide, fast cuts and text-safe framing. Automatic checks must pass.
5. **Tastemaker score.** Each clip is scored 0-10 (below). Under the bar, it's never posted.
6. **Queue.** `clipper ops queue` fills today's slots from each account's health caps,
   highest score first, and writes the post sheet.
7. **Post and submit** (Aaron, or a posting step he has approved): every allowed platform,
   the post checklist's caption, the campaign submit link within 30 minutes.
8. **Measure.** Ledger rows at 24 h, 72 h and 7 days: views, and where the platform shows
   them, watch-through, likes and shares.
9. **Learn.** Weekly: compare scores with results, move the thresholds, retire hooks and
   formats that don't earn, and report $ earned, $ per hour and what changes next week.

## Quality bar and account health (implemented in `clipper ops`)

Every clip gets the tastemaker bar score (0-10: hook in the first 1.5 s, payoff, energy, makes
sense without context, visual quality, fit with the top Shorts), stored on the clip. A clip is
never queued below the bar: **7.5 by default, never below 7.0**, set per campaign (`gate` in
campaigns.yaml) and per account (work/revenue/accounts.yaml), and moved weekly by calibration.

Each posting account (one per platform handle) has a health state. Baseline = the median 7-day
views of the account's last 30 days of posts.

| State | Trigger | Posts a day | Minimum score |
|---|---|---|---|
| Warm-up | fewer than 10 posts | 1 | 8.0 |
| Healthy | median of the last 10 at or above baseline | up to 3 | the bar (7.5) |
| Watch | last-10 median at 50-99% of baseline, or 2 in a row under 25% | 1 | 8.5 |
| Recover | last-10 median under 50% of baseline | 0 for 48 h, then 1 | 9.0 |

Down as soon as a trigger fires; up one state after 5 posts in a row at or above baseline.

Queue rules: highest score first, at least 3 hours apart on an account, never two clips of one
moment on an account on the same day, and a clip that waits a day without a slot expires.

Stop rules (`work/revenue/stops.yaml`): a campaign rejection or strike warning pauses that
campaign; a reach-limit or unoriginal-content notice pauses that account; Reach-run campaigns
(SypherPK included) stay blocked until Aaron sets `reach_ai_content_ok: true`.

Weekly calibration: if clips scored 7.5-8.0 beat the account median at least half the time, the
bar drops 0.25; under a third of the time, it rises 0.25; never below 7.0. Next week's slots are
split across campaigns by measured $ per clip, at most 50% to one campaign, 20% kept for exploring.

Commands: `clipper ops queue` (writes work/revenue/queue-YYYY-MM-DD.md), `clipper ops measure`
(public view counts at 24 h / 72 h / 7 d via yt-dlp metadata, no sign-in), `clipper ops calibrate`
(weekly), `clipper ops status`.

## Money rules

- Post fast: budgets drain first-come, first-served. A clip above the bar from a fresh stream is
  worth more today than tomorrow.
- Front-load: when a campaign's budget is over 70% spent, only clips well above the bar that can clear the
  minimum within 48 hours are worth the time.
- A clip that has hit the per-clip cap ($500 at $2 CPM = 250,000 views) earns nothing more.
  Spend the effort on the next clip.
- Hours count: if a campaign's expected $ per hour falls below the floor in the policy for two
  weeks running, drop it.

## Calibration

The tier thresholds start as judgment calls based on the tastemaker's 24 reference Shorts
(median 24M views). After 30 posted clips with 7-day results, replace them with what the
ledger shows actually cleared the minimum. Until then, every score is a prediction, and the
plan says so in every report.

## What stays with Aaron

The strategist plans, ranks, renders, queues and reports on its own. Posting, submitting to
campaigns, creating accounts, accepting terms and spending money need Aaron's approval each
time. No bought views, bots, engagement pods, reposting other clippers' work or misleading
hooks, ever.

## Where we are now (2026-11-26)

- Sypher Nov 15 campaign: 9 concept-gated moments rendered, **0 at or above the 7.5 bar**. The best,
  "Sypher's first ever hole-in-one", scores about 6.5. The campaign is also Reach-run, so it stays
  blocked until Reach answers the AI-content question.
- The source was calm. The biggest lever is better source: the next high-energy stream on a
  licensed campaign, with chat replay data (waiting on TwitchDownloaderCLI approval).
- The GTA 6 campaigns ($1.50 CPM, $11k combined budget) can't be priced until their campaign
  pages and channels are added to campaigns.yaml.
