"""Driving simulator setup and the digital-twin planner (the "cloud brain").

The planner copies the traffic around the car into a twin, test-drives each of
the five possible maneuvers ~1.6 s into the future, and picks the one with the
best mix of progress and safety margin. It is accurate but compute-heavy,
which is exactly why you'd run it on a server rather than in the car.
"""
from __future__ import annotations

import copy
import time

import gymnasium as gym
import highway_env  # noqa: F401  (registers the environments)
import numpy as np
from highway_env.vehicle.controller import MDPVehicle

ACTIONS = {0: "LANE_LEFT", 1: "IDLE", 2: "LANE_RIGHT", 3: "FASTER", 4: "SLOWER"}
DECISION_PERIOD_MS = 200          # 5 decisions per second

CONFIG = {
    "policy_frequency": 5,
    "duration": 12,               # seconds of driving per episode
    "vehicles_density": 1.6,      # dense traffic
    "lanes_count": 3,
    # Default highway-env cars can't go below 20 m/s while traffic drives ~21 m/s,
    # which makes many crashes physically unavoidable. Let the car brake.
    "action": {"type": "DiscreteMetaAction", "target_speeds": [10, 15, 20, 25, 30]},
    "observation": {"type": "Kinematics", "vehicles_count": 7,
                    "features": ["presence", "x", "y", "vx", "vy"],
                    "normalize": True, "absolute": False},
}


def make_env(seed: int):
    env = gym.make("highway-fast-v0", config=CONFIG)
    obs, _ = env.reset(seed=seed)
    return env, obs


def features(obs) -> np.ndarray:
    return np.asarray(obs, dtype=np.float32).ravel()


def _rollout(twin, first, follow, horizon, freq, dt):
    r = copy.deepcopy(twin)
    e = next(v for v in r.vehicles if isinstance(v, MDPVehicle))
    e.act(ACTIONS[first])
    min_gap = 99.0
    for k in range(horizon * freq):
        if k and k % freq == 0:
            e.act(ACTIONS[follow])
        r.act()
        r.step(dt)
        if e.crashed:
            return -10 + k * 0.02          # crashing later is slightly less bad
        for v in r.vehicles:
            if v is e:
                continue
            if e.lane_index in (v.lane_index, v.target_lane_index) or v.lane_index == e.target_lane_index:
                d = v.position[0] - e.position[0]
                if 0 < d < min_gap:
                    min_gap = d
    progress = np.clip((e.speed - 10) / 20, 0, 1)
    safety = np.clip((min_gap - 8) / 20, -1, 1)
    return 0.6 * progress + 0.4 * safety


def twin_plan(u, horizon: int = 8, radius: float = 70.0):
    """Return (action, scores, compute_ms). `u` is env.unwrapped."""
    t0 = time.perf_counter()
    ego, road = u.vehicle, u.road
    saved = road.vehicles
    road.vehicles = [v for v in saved if v is ego or abs(v.position[0] - ego.position[0]) < radius]
    twin = copy.deepcopy(road)          # the digital twin: nearby traffic only
    road.vehicles = saved
    freq = u.config["simulation_frequency"] // u.config["policy_frequency"]
    dt = 1 / u.config["simulation_frequency"]
    scores = {a: max(_rollout(twin, a, f, horizon, freq, dt) for f in (1, 4)) for a in ACTIONS}
    best = max(scores, key=scores.get)
    return best, scores, (time.perf_counter() - t0) * 1000


def snapshot(u) -> list:
    """Positions of every car, for replay on the showcase site."""
    ego = u.vehicle
    return [[round(float(v.position[0] - ego.position[0]), 2), round(float(v.position[1]), 2),
             round(float(v.heading), 3), int(v is ego), int(bool(v.crashed))]
            for v in u.road.vehicles if abs(v.position[0] - ego.position[0]) < 90]


def sensor_features(u) -> np.ndarray:
    """What a car's sensors report, per lane (left, current, right): gap to the
    nearest car ahead and behind, closing speeds, and time-to-collision. Cars
    that are moving INTO a lane count for that lane too, so a cut-in is visible
    before it completes. Plus the car's own state, including whether it is
    already part-way through a lane change."""
    ego = u.vehicle
    lanes = u.road.network.all_side_lanes(ego.lane_index)
    n, li = len(lanes), ego.lane_index[2]
    lane = u.road.network.get_lane(ego.lane_index)
    _, lat = lane.local_coordinates(ego.position)
    tgt = ego.target_lane_index[2] - li
    out = [ego.speed / 30.0, li / max(n - 1, 1), lat / 4.0, ego.heading * 5, tgt,
           float(getattr(ego, "speed_index", 0)) / 4]
    for off in (-1, 0, 1):
        j = li + off
        if not 0 <= j < n:
            out += [0] + [0] * 7
            continue
        front = rear = None
        cutin = 0.0
        for v in u.road.vehicles:
            if v is ego or j not in (v.lane_index[2], v.target_lane_index[2]):
                continue
            if v.lane_index[2] != j:
                cutin = 1.0
            d = v.position[0] - ego.position[0]
            if d > 0 and (front is None or d < front[0]):
                front = (d, ego.speed - v.speed)
            if d <= 0 and (rear is None or -d < rear[0]):
                rear = (-d, v.speed - ego.speed)
        fg, fc = front if front else (80.0, 0.0)
        rg, rc = rear if rear else (80.0, 0.0)
        ttc_f = fg / fc if fc > 0.1 else 10.0
        ttc_r = rg / rc if rc > 0.1 else 10.0
        out += [1, min(fg, 80) / 80, np.clip(fc / 10, -2, 2), min(ttc_f, 10) / 10,
                min(rg, 80) / 80, np.clip(rc / 10, -2, 2), min(ttc_r, 10) / 10, cutin]
    return np.asarray(out, dtype=np.float32)
