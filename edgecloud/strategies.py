"""Four ways to decide where each driving decision is computed.

Every decision has a hard deadline (DEADLINE_MS). The car must act at the
deadline with whatever answer it has.
"""
from __future__ import annotations

import math
from collections import deque

import numpy as np

DEADLINE_MS = 150.0     # of each 200 ms decision period; the rest is for actuation


class Strategy:
    name = ""
    def reset(self): ...


class EdgeOnly(Strategy):
    name = "Edge only"
    def step(self, ctx):
        return dict(action=ctx.edge_action, source="edge", latency=ctx.edge_ms, offloaded=False, missed=False)


class CloudOnly(Strategy):
    """Always ask the cloud and wait. If the answer is late or lost, keep doing
    whatever the car was doing (what a naive client does on timeout)."""
    name = "Cloud only"
    def reset(self):
        self.prev = 1
    def step(self, ctx):
        ok, act, total = ctx.ask_cloud(ctx.link.sample())
        if ok:
            self.prev = act
            return dict(action=act, source="cloud", latency=total, offloaded=True, missed=False)
        return dict(action=self.prev, source="stale", latency=DEADLINE_MS, offloaded=True, missed=True)


class Hedged(Strategy):
    """Ask the cloud on every decision, but compute the edge answer too and use
    it if the cloud is late. Safe, but pays for a request every 200 ms."""
    name = "Cloud + edge backup"
    def step(self, ctx):
        ok, act, total = ctx.ask_cloud(ctx.link.sample())
        if ok:
            return dict(action=act, source="cloud", latency=total, offloaded=True, missed=False)
        return dict(action=ctx.edge_action, source="fallback", latency=DEADLINE_MS, offloaded=True, missed=True)


class Adaptive(Strategy):
    """Ask the cloud only when it's worth it AND has a fair chance of arriving:
      * worth it   - the edge model is unsure: its best and second-best
                     maneuver scores are within `confidence` of each other
      * a chance   - from the last 20 heartbeat pings (tiny messages sent every
                     decision), estimate how often round trip + server compute
                     would have fit the deadline; ask if that's >= `min_chance`
    Late answers fall back to the edge model.

    Design note: an earlier version asked only when the 90th-percentile round
    trip fit the deadline. On busy 4G it then never asked at all, giving up the
    cloud even though half the answers would have arrived. With a backup in
    place, a late answer costs only bandwidth, so the right question is "is
    there a fair chance?" rather than "is it almost certain?"."""
    name = "Adaptive switcher"

    def __init__(self, confidence: float = 0.3, min_chance: float = 0.25, window: int = 20):
        self.tau, self.min_chance, self.window = confidence, min_chance, window

    def reset(self):
        self.pings = deque(maxlen=self.window)

    def chance_on_time(self, compute_ms: float) -> float:
        if len(self.pings) < 5:
            return 1.0                       # no evidence yet: try the cloud
        return sum(math.isfinite(r) and r + compute_ms <= DEADLINE_MS for r in self.pings) / len(self.pings)

    def step(self, ctx):
        self.pings.append(ctx.link.sample())                    # heartbeat
        unsure = ctx.edge_conf < self.tau
        chance = self.chance_on_time(ctx.cloud_compute_estimate)
        if not unsure or chance < self.min_chance:
            return dict(action=ctx.edge_action, source="edge", latency=ctx.edge_ms, offloaded=False, missed=False)
        ok, act, total = ctx.ask_cloud(ctx.link.sample())
        if ok:
            return dict(action=act, source="cloud", latency=total, offloaded=True, missed=False)
        return dict(action=ctx.edge_action, source="fallback", latency=DEADLINE_MS, offloaded=True, missed=True)


ALL = [EdgeOnly, CloudOnly, Hedged, Adaptive]
