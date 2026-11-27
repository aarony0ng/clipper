"""Unit tests for the revenue loop's health state machine, queue builder and calibration."""
import unittest
from datetime import date, datetime, timedelta

from clipper import ops
from clipper.config import Campaign

NOW = datetime(2026, 12, 1, 9, 0)


def posts(views, days_ago_start=20):
    """Ledger rows, oldest first, one a day, each with 7-day views."""
    out = []
    for i, v in enumerate(views):
        t = NOW - timedelta(days=days_ago_start - i * 0.5)
        out.append({"posted_at": t.isoformat(), "views_7d": str(v)})
    return out


def camp(cid="c1", **kw):
    return Campaign(id=cid, creator="Sypher", licensed=True, channels=[], **kw)


class HealthTests(unittest.TestCase):
    def test_warmup_until_ten_posts(self):
        h = ops.next_health(posts([1000] * 9), ops.Health(), NOW)
        self.assertEqual(h.state, "warmup")
        self.assertEqual(ops.caps(h, 7.5, NOW), (1, 8.0))

    def test_leaves_warmup_to_healthy(self):
        h = ops.next_health(posts([1000] * 12), ops.Health(), NOW)
        self.assertEqual(h.state, "healthy")
        self.assertEqual(ops.caps(h, 7.5, NOW), (3, 7.5))

    def test_two_posts_under_quarter_baseline_is_watch(self):
        h = ops.next_health(posts([1000] * 12 + [100, 100]), ops.Health("healthy", NOW.isoformat()), NOW)
        self.assertEqual(h.state, "watch")
        self.assertEqual(ops.caps(h, 7.5, NOW), (1, 8.5))

    def test_median_under_half_is_recover_and_pauses_48h(self):
        h = ops.next_health(posts([1000] * 10 + [200] * 10), ops.Health("healthy", "2026-11-01T00:00:00"), NOW)
        self.assertEqual(h.state, "recover")
        self.assertEqual(ops.caps(h, 7.5, NOW)[0], 0)              # just entered: paused
        self.assertEqual(ops.caps(h, 7.5, NOW + timedelta(hours=49)), (1, 9.0))

    def test_moves_up_one_state_only_after_five_at_baseline(self):
        prev = ops.Health("recover", "2026-11-01T00:00:00")
        # last-10 median well above baseline, but only 4 recent posts at/above it
        h = ops.next_health(posts([400] * 6 + [100] * 4 + [5000] * 4), prev, NOW)
        self.assertEqual(h.state, "recover")
        h = ops.next_health(posts([400] * 6 + [100] * 4 + [5000] * 5), prev, NOW)
        self.assertEqual(h.state, "watch")                        # one step, not straight to healthy

    def test_gate_never_below_floor(self):
        self.assertEqual(ops.caps(ops.Health("healthy"), 6.0, NOW)[1], 7.0)


class QueueTests(unittest.TestCase):
    def setUp(self):
        self.accounts = [{"id": "tt", "platform": "tiktok", "campaigns": ["c1"], "active": True}]
        self.health = {"tt": ops.Health("healthy", NOW.isoformat())}
        self.camps = {"c1": camp()}
        self.stops = {"campaigns": {}, "accounts": {}}
        self.day = date(2026, 12, 1)

    def cand(self, cid, score, moment=None, first=None):
        return ops.Candidate(cid, moment or cid + "-m", "c1", score, first_seen=first or "2026-12-01")

    def build(self, cands, **kw):
        args = dict(day=self.day, candidates=cands, accounts=self.accounts, health=self.health, gates={},
                    campaigns=self.camps, stops=self.stops, posted=set())
        args.update(kw)
        return ops.build_queue(**args)

    def test_below_bar_never_queued(self):
        slots, notes = self.build([self.cand("a", 7.4), self.cand("b", 6.5)])
        self.assertEqual(slots, [])

    def test_highest_score_first_capped_and_spaced(self):
        slots, _ = self.build([self.cand(x, s) for x, s in [("a", 7.6), ("b", 9.0), ("c", 8.1), ("d", 7.9)]])
        self.assertEqual([s.clip.clip_id for s in slots], ["b", "c", "d"])   # healthy cap 3
        gaps = [(b.time - a.time) for a, b in zip(slots, slots[1:])]
        self.assertTrue(all(g >= timedelta(hours=3) for g in gaps))

    def test_one_clip_per_moment_per_day(self):
        slots, _ = self.build([self.cand("a-v1", 9, "m1"), self.cand("a-v2", 8.9, "m1"), self.cand("b", 8, "m2")])
        self.assertEqual([s.clip.clip_id for s in slots], ["a-v1", "b"])

    def test_leftovers_expire_after_a_day(self):
        slots, notes = self.build([self.cand("old", 9.5, first="2026-11-29")])
        self.assertEqual(slots, [])
        self.assertTrue(any("expired" in n for n in notes))

    def test_stops_and_reach_block(self):
        slots, _ = self.build([self.cand("a", 9)], stops={"campaigns": {"c1": "strike warning"}, "accounts": {}})
        self.assertEqual(slots, [])
        slots, _ = self.build([self.cand("a", 9)], stops={"campaigns": {}, "accounts": {"tt": "reach limit"}})
        self.assertEqual(slots, [])
        slots, notes = self.build([self.cand("a", 9)], campaigns={"c1": camp(reach_run=True)})
        self.assertEqual(slots, [])
        self.assertTrue(any("Reach-run" in n for n in notes))
        slots, _ = self.build([self.cand("a", 9)], campaigns={"c1": camp(reach_run=True, reach_ai_content_ok=True)})
        self.assertEqual(len(slots), 1)

    def test_warmup_account_needs_8(self):
        slots, _ = self.build([self.cand("a", 7.9), self.cand("b", 8.2)], health={"tt": ops.Health("warmup")})
        self.assertEqual([s.clip.clip_id for s in slots], ["b"])

    def test_already_posted_moment_skipped(self):
        slots, _ = self.build([self.cand("a-v1", 9, "m1")], posted={("tt", "m1")})
        self.assertEqual(slots, [])


class CalibrationTests(unittest.TestCase):
    def test_lower_when_band_beats_median_half_the_time(self):
        self.assertEqual(ops.calibrate_gate(7.5, [(7.6, 200), (7.7, 300), (7.8, 50), (7.9, 40)], 100), 7.25)

    def test_raise_when_band_rarely_beats(self):
        self.assertEqual(ops.calibrate_gate(7.5, [(7.6, 20), (7.7, 30), (7.8, 50), (7.9, 400)], 100), 7.75)

    def test_floor(self):
        self.assertEqual(ops.calibrate_gate(7.0, [(7.6, 900)] * 5, 100), 7.0)

    def test_allocation_caps_and_explore(self):
        sh = ops.allocate({"a": [100, 100], "b": [1, 1]})
        self.assertLessEqual(sh["a"], 0.5)
        self.assertGreaterEqual(sh["_explore"], 0.2)
        self.assertAlmostEqual(sum(sh.values()), 1.0, places=3)


if __name__ == "__main__":
    unittest.main()
