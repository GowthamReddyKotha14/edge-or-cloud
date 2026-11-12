"""Closed-loop benchmark: every strategy drives the same traffic scenarios
(same seeds) under every network condition. Paired seeds mean differences
come from the strategy, not from luckier traffic."""
from __future__ import annotations

import math
import statistics as st
from dataclasses import dataclass, field

import numpy as np

from .edge import EdgeModel
from .network import PROFILES, Link, Profile
from .sim import make_env, snapshot, twin_plan
from .strategies import ALL, DEADLINE_MS


class Ctx:
    """What a strategy sees at one decision."""
    def __init__(self, u, obs, edge, link, compute_hist):
        self.u, self.obs, self.link = u, obs, link
        self.edge_action, self.edge_conf, self.edge_ms = edge.decide(u)
        self._hist = compute_hist

    @property
    def cloud_compute_estimate(self):
        return st.median(self._hist[-50:]) if self._hist else 45.0

    def ask_cloud(self, rtt):
        """Send the request. Returns (on_time, action, total_ms)."""
        if not math.isfinite(rtt) or rtt >= DEADLINE_MS:
            return False, None, DEADLINE_MS          # lost, or can't possibly make it
        act, _, compute = twin_plan(self.u)
        self._hist.append(compute)
        total = rtt + compute
        return total <= DEADLINE_MS, act, total


@dataclass
class Episode:
    crashed: bool
    seconds: float
    speed: float
    sources: dict
    latencies: list
    offloads: int
    misses: int
    decisions: int
    frames: list = field(default_factory=list)


def run_episode(strategy, profile: Profile, seed: int, edge: EdgeModel, compute_hist, record=False) -> Episode:
    env, obs = make_env(seed)
    u = env.unwrapped
    link = Link(profile, seed=seed * 7919 + 1)
    strategy.reset()
    speeds, lats, src = [], [], {}
    off = miss = n = 0
    frames = []
    while True:
        ctx = Ctx(u, obs, edge, link, compute_hist)
        d = strategy.step(ctx)
        n += 1
        off += d["offloaded"]; miss += d["missed"]
        src[d["source"]] = src.get(d["source"], 0) + 1
        lats.append(d["latency"])
        if record:
            frames.append({"cars": snapshot(u), "src": d["source"], "lat": round(d["latency"], 1),
                           "a": d["action"], "conf": round(ctx.edge_conf, 2),
                           "x": round(float(u.vehicle.position[0]), 2), "v": round(float(u.vehicle.speed), 2)})
        obs, _, term, trunc, info = env.step(d["action"])
        speeds.append(float(u.vehicle.speed))
        if term or trunc:
            if record:
                frames.append({"cars": snapshot(u), "src": "end", "lat": 0, "a": None, "conf": None,
                               "x": round(float(u.vehicle.position[0]), 2), "v": round(float(u.vehicle.speed), 2)})
            return Episode(bool(info["crashed"]), round(float(u.time), 1), float(np.mean(speeds)), src, lats,
                           off, miss, n, frames)


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def summarize(eps: list[Episode]) -> dict:
    crashes = sum(e.crashed for e in eps)
    lats = [x for e in eps for x in e.latencies]
    decisions = sum(e.decisions for e in eps)
    offloads = sum(e.offloads for e in eps)
    src = {}
    for e in eps:
        for k, v in e.sources.items():
            src[k] = src.get(k, 0) + v
    lo, hi = wilson(crashes, len(eps))
    return {
        "episodes": len(eps), "crashes": crashes, "crash_rate": crashes / len(eps), "crash_ci": [lo, hi],
        "speed": float(np.mean([e.speed for e in eps])),
        "km_per_crash": (sum(e.speed * e.seconds for e in eps) / 1000) / max(crashes, 1),
        "offload_rate": offloads / decisions,
        "miss_rate": sum(e.misses for e in eps) / max(offloads, 1),
        "p50_ms": float(np.percentile(lats, 50)), "p99_ms": float(np.percentile(lats, 99)),
        "sources": {k: v / decisions for k, v in src.items()},
    }


def run_bench(seeds, networks=None, log=print, record_seeds=()):
    edge = EdgeModel.load()
    networks = networks or PROFILES
    hist: list[float] = []
    results, replays = {}, {}
    for nkey, prof in networks.items():
        for cls in ALL:
            strat = cls()
            eps = []
            for s in seeds:
                ep = run_episode(strat, prof, s, edge, hist, record=s in record_seeds)
                eps.append(ep)
                if ep.frames:
                    replays[f"{nkey}|{strat.name}|{s}"] = {"frames": ep.frames, "crashed": ep.crashed}
            results[(nkey, strat.name)] = summarize(eps)
            r = results[(nkey, strat.name)]
            log(f"{prof.name:12} {strat.name:20} crashes {r['crashes']:>2}/{len(eps)}  speed {r['speed']:.1f}  "
                f"offload {r['offload_rate']:.0%}  late {r['miss_rate']:.0%}  p99 {r['p99_ms']:.0f}ms")
    return results, replays, {"compute_ms_median": st.median(hist) if hist else None,
                              "compute_ms_p99": float(np.percentile(hist, 99)) if hist else None}


RESULTS_DIR = __import__("pathlib").Path(__file__).resolve().parent.parent / "results"
STRATS = {cls.name: cls for cls in ALL}


def run_combo(nkey: str, strat_name: str, seeds, record_seeds=(), log=print) -> dict:
    """Run one network x strategy cell and save it, so a long benchmark can be
    built up in pieces and resumed."""
    import json
    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / f"{nkey}__{strat_name.replace(' ', '_').replace('+', 'plus')}.json"
    edge, hist = EdgeModel.load(), []
    prof, strat = PROFILES[nkey], STRATS[strat_name]()
    eps, replays = [], {}
    for s in seeds:
        ep = run_episode(strat, prof, s, edge, hist, record=s in record_seeds)
        eps.append(ep)
        if ep.frames:
            replays[str(s)] = {"frames": ep.frames, "crashed": ep.crashed}
    r = summarize(eps)
    r.update(network=nkey, strategy=strat_name, compute_ms=hist[:2000],
             per_episode=[{"seed": s, "crashed": e.crashed, "speed": round(e.speed, 2)} for s, e in zip(seeds, eps)])
    out.write_text(json.dumps({"summary": r, "replays": replays}))
    log(f"{prof.name:12} {strat_name:20} crashes {r['crashes']:>2}/{len(eps)}  speed {r['speed']:.1f}  "
        f"offload {r['offload_rate']:.0%}  late {r['miss_rate']:.0%}  p99 {r['p99_ms']:.0f}ms")
    return r
