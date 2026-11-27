# Revenue plan: views without wrecking the account

Owner: the revenue strategist (`.claude/agents/revenue-strategist.md`), with the
tastemaker as the quality gate. Machine-readable rules: `revenue_policy.yaml`.

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
5. **Tastemaker score.** Each clip is scored 0-10 on six things: hook in the first 1.5 s,
   payoff, energy, makes sense without context, visual quality, fit with the top Shorts.
   The scores put it in a tier (below).
6. **Queue.** Tier A clips go in the posting queue, best first, within the cadence limits.
   Tier B clips can only go out as tests, within the test limits. Tier C is archived.
7. **Post and submit** (Aaron, or a posting step he has approved): every allowed platform,
   the post checklist's caption, the campaign submit link within 30 minutes.
8. **Measure.** Ledger rows at 24 h, 72 h and 7 days: views, and where the platform shows
   them, watch-through, likes and shares.
9. **Learn.** Weekly: compare scores with results, move the thresholds, retire hooks and
   formats that don't earn, and report $ earned, $ per hour and what changes next week.

## Tiers (the quality/volume balance)

| Tier | Rule | What happens |
|---|---|---|
| A | average ≥ 7.0, no score under 6, hook ≥ 7, visual ≥ 8 | Post on every allowed platform, in queue order |
| B | average 6.0–6.9, hook ≥ 6, visual ≥ 8 | Test only: one platform, at most 1 test per week, never two posts in a row that are tests |
| C | anything else | Never posted |

Hook and visual have their own floors because the hook decides the first test audience
and a sloppy frame (cut text, smudges) reads as a low-quality account.

## Cadence and account health

- At most 2 posts per platform per day, at least 4 hours apart. Quality over count.
- Keep a **rolling health score** per platform: the median 24-hour views of the last 5 posts,
  compared with the account's own 30-day median.
  - Healthy (≥ 60% of the 30-day median): follow the queue.
  - Warning (30–60%): Tier A only, at most 1 post a day, no tests.
  - Pause (< 30%, or 3 posts in a row under 1,000 views at 24 h): stop posting on that
    platform for 48 hours, review the last 5 posts, and raise the Tier A bar by 0.5 until
    two posts in a row clear the campaign minimum.
- A new account starts in "warning" until it has 10 posts.
- Never delete posts to tidy the numbers, and never repost the same clip to the same account.

## Money rules

- Post fast: budgets drain first-come, first-served. A Tier A clip from a fresh stream is
  worth more today than tomorrow.
- Front-load: when a campaign's budget is over 70% spent, only Tier A clips that can clear the
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

- Sypher Nov 15 campaign: 9 concept-gated moments rendered, **0 in Tier A**. The best, "Sypher's first
  ever hole-in-one", scores about 6.5: Tier B, so one test post on one platform at most.
- The source was calm. The biggest lever is better source: the next high-energy stream on a
  licensed campaign, with chat replay data (waiting on TwitchDownloaderCLI approval).
- The GTA 6 campaigns ($1.50 CPM, $11k combined budget) can't be priced until their campaign
  pages and channels are added to campaigns.yaml.
