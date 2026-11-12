"""Network conditions between the car and the cloud.

Round-trip times are drawn from a log-normal distribution (network latency is
right-skewed: usually fine, occasionally terrible). A two-state "good/bad"
Markov chain adds bursts, because real signal problems come in streaks
(a tunnel, a congested cell), not as independent coin flips. Lost requests
never arrive.

The three built-in profiles are modeling assumptions for illustration. Replace
them with real measurements from your own connection: `edgecloud measure`.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, asdict

import numpy as np


@dataclass
class Profile:
    name: str
    median_ms: float          # typical round trip in the good state
    sigma: float              # log-normal spread (tail heaviness)
    loss: float               # probability a request is lost
    p_bad: float = 0.0        # chance per decision of entering a bad burst
    p_recover: float = 0.3    # chance per decision of leaving it
    bad_factor: float = 4.0   # latency multiplier during a burst

    def to_dict(self):
        return asdict(self)


PROFILES = {
    "5g": Profile("Good 5G", 22, 0.30, 0.002, p_bad=0.01, p_recover=0.5, bad_factor=3),
    "4g": Profile("Busy 4G", 60, 0.45, 0.01, p_bad=0.04, p_recover=0.3, bad_factor=3),
    "weak": Profile("Weak signal", 110, 0.60, 0.04, p_bad=0.08, p_recover=0.25, bad_factor=4),
}


class Link:
    def __init__(self, profile: Profile, seed: int):
        self.p = profile
        self.rng = np.random.default_rng(seed)
        self.bad = False

    def sample(self) -> float:
        """One round trip in ms; math.inf if the request is lost."""
        if self.bad:
            self.bad = self.rng.random() > self.p.p_recover
        else:
            self.bad = self.rng.random() < self.p.p_bad
        if self.rng.random() < self.p.loss * (3 if self.bad else 1):
            return math.inf
        rtt = self.p.median_ms * math.exp(self.rng.normal(0, self.p.sigma))
        return rtt * (self.p.bad_factor if self.bad else 1.0)


def fit_profile(name: str, samples_ms: list[float], lost: int = 0) -> Profile:
    """Build a profile from real measurements (median and log-spread)."""
    ok = np.asarray([s for s in samples_ms if math.isfinite(s) and s > 0])
    total = len(ok) + lost
    logs = np.log(ok)
    return Profile(name, float(np.exp(np.median(logs))), float(max(np.std(logs), 0.05)),
                   lost / max(total, 1))


def load_profile(path: str) -> Profile:
    return Profile(**json.load(open(path)))
