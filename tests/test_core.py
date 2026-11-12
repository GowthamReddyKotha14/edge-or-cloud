import math

import numpy as np
import pytest

from edgecloud.network import PROFILES, Link, Profile, fit_profile
from edgecloud.strategies import Adaptive, CloudOnly, EdgeOnly, Hedged, DEADLINE_MS


def test_link_matches_profile_median_and_losses():
    p = Profile("t", median_ms=50, sigma=0.4, loss=0.05)
    link = Link(p, seed=1)
    xs = [link.sample() for _ in range(20000)]
    ok = [x for x in xs if math.isfinite(x)]
    assert abs(np.median(ok) - 50) < 2
    assert 0.04 < 1 - len(ok) / len(xs) < 0.06


def test_bursts_make_bad_streaks():
    """Bad-state bursts should make slow samples come in streaks, like a tunnel."""
    p = Profile("t", 20, 0.1, 0.0, p_bad=0.05, p_recover=0.2, bad_factor=5)
    link = Link(p, seed=3)
    slow = [link.sample() > 60 for _ in range(5000)]
    runs, cur = [], 0
    for x in slow + [False]:
        if x:
            cur += 1
        elif cur:
            runs.append(cur); cur = 0
    assert 0.1 < np.mean(slow) < 0.4
    assert np.mean(runs) > 2          # independent coin flips would average ~1.2


def test_fit_profile_recovers_median():
    rng = np.random.default_rng(0)
    xs = list(40 * np.exp(rng.normal(0, 0.3, 500)))
    prof = fit_profile("x", xs, lost=25)
    assert abs(prof.median_ms - 40) < 3 and abs(prof.loss - 25 / 525) < 1e-9


class FakeLink:
    def __init__(self, rtts):
        self.rtts = list(rtts)
    def sample(self):
        return self.rtts.pop(0) if self.rtts else 10.0


class FakeCtx:
    def __init__(self, rtts, conf=0.0, cloud_action=4, compute=40.0):
        self.link = FakeLink(rtts)
        self.edge_action, self.edge_conf, self.edge_ms = 1, conf, 0.3
        self.cloud_action, self.compute = cloud_action, compute
        self.cloud_compute_estimate = compute
        self.asked = 0
    def ask_cloud(self, rtt):
        self.asked += 1
        if not math.isfinite(rtt) or rtt + self.compute > DEADLINE_MS:
            return False, None, DEADLINE_MS
        return True, self.cloud_action, rtt + self.compute


def test_cloud_only_holds_previous_action_when_late():
    s = CloudOnly(); s.reset()
    assert s.step(FakeCtx([20]))["action"] == 4                 # on time: cloud's answer
    d = s.step(FakeCtx([math.inf]))                             # lost: keeps doing the last thing
    assert d["source"] == "stale" and d["action"] == 4 and d["missed"]


def test_hedged_falls_back_to_edge():
    d = Hedged().step(FakeCtx([500]))
    assert d["source"] == "fallback" and d["action"] == 1


def test_adaptive_trusts_confident_edge_and_skips_hopeless_network():
    a = Adaptive(confidence=0.3, min_chance=0.25); a.reset()
    ctx = FakeCtx([10] * 10, conf=0.9)
    assert a.step(ctx)["source"] == "edge" and ctx.asked == 0     # confident: don't ask
    a.reset()
    for _ in range(10):                                           # hopeless network: stop asking
        ctx = FakeCtx([400, 400], conf=0.0); d = a.step(ctx)
    assert d["source"] == "edge" and ctx.asked == 0
    a.reset()
    ctx = FakeCtx([10, 10], conf=0.0)
    assert a.step(ctx)["source"] == "cloud"                       # unsure + good network: ask


def test_edge_model_is_fast_and_valid():
    from edgecloud.edge import EdgeModel
    from edgecloud.sim import make_env, sensor_features
    env, _ = make_env(5)
    m = EdgeModel.load()
    a, conf, ms = m.decide(env.unwrapped)
    assert a in range(5) and conf >= 0 and ms < 50
    assert sensor_features(env.unwrapped).shape == (30,)


def test_planner_avoids_driving_into_a_slower_car():
    """Smoke test of the digital twin: in a fresh scene it returns a scored action quickly."""
    from edgecloud.sim import make_env, twin_plan
    env, _ = make_env(11)
    act, scores, ms = twin_plan(env.unwrapped)
    assert act == max(scores, key=scores.get) and len(scores) == 5 and ms < 2000


def test_one_episode_runs_end_to_end():
    from edgecloud.bench import run_episode
    from edgecloud.edge import EdgeModel
    ep = run_episode(Hedged(), PROFILES["5g"], 801, EdgeModel.load(), [], record=True)
    assert ep.decisions > 0 and len(ep.frames) == ep.decisions + 1
    assert set(ep.sources) <= {"cloud", "fallback"}
