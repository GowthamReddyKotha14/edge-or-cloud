"""The "edge brain": a small neural network distilled from the planner.

Two choices make it work:
  * It learns the planner's SCORE for every maneuver, not just which one won.
    A crash scores -10, so the network learns how bad each wrong move is,
    not merely that it lost. The gap between its best and second-best score
    is a built-in confidence signal for the adaptive switcher.
  * DAgger: after an initial pass on planner driving, the small network
    drives itself and the planner labels the situations it gets into.
    A model trained only on expert driving never learns to recover from its
    own mistakes, because the expert never makes them.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from .sim import ACTIONS, make_env, sensor_features, twin_plan

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
SCORE_FLOOR = -3.0      # squash the -10 crash scores so they don't swamp training


class EdgeModel:
    def __init__(self, reg):
        self.reg = reg

    def scores(self, u):
        return self.reg.predict(sensor_features(u)[None, :])[0]

    def decide(self, u):
        """Return (action, confidence margin, compute_ms)."""
        t0 = time.perf_counter()
        s = self.scores(u)
        order = np.argsort(s)[::-1]
        return int(order[0]), float(s[order[0]] - s[order[1]]), (time.perf_counter() - t0) * 1000

    def save(self, path=MODEL_DIR / "edge.joblib"):
        Path(path).parent.mkdir(exist_ok=True)
        joblib.dump(self.reg, path)

    @classmethod
    def load(cls, path=MODEL_DIR / "edge.joblib"):
        return cls(joblib.load(path))


def _collect(seeds, driver, explore, rng):
    X, Y, crashes = [], [], 0
    for s in seeds:
        env, _ = make_env(s)
        u = env.unwrapped
        while True:
            best, sc, _ = twin_plan(u)
            X.append(sensor_features(u))
            Y.append([max(sc[a], SCORE_FLOOR) for a in ACTIONS])
            a = driver(u, best)
            if rng.random() < explore:
                a = int(rng.integers(0, 5))
            _, _, term, trunc, info = env.step(a)
            if term or trunc:
                crashes += bool(info["crashed"])
                break
    return X, Y, crashes


def _fit(X, Y, seed):
    reg = make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(48, 24), max_iter=1500,
                                                       random_state=seed, early_stopping=True, alpha=1e-3))
    reg.fit(np.array(X), np.array(Y))
    return EdgeModel(reg)


def train(n_expert=40, dagger_rounds=3, n_dagger=25, seed=0, log=print) -> dict:
    """Resumable: each round's labeled data is saved, so an interrupted run
    picks up where it stopped instead of re-running the slow planner."""
    t0 = time.time()
    ckpt = MODEL_DIR / "train_data.npz"
    MODEL_DIR.mkdir(exist_ok=True)
    done, X, Y, history = 0, [], [], []
    if ckpt.exists():
        z = np.load(ckpt, allow_pickle=True)
        X, Y, done, history = list(z["X"]), list(z["Y"]), int(z["done"]), list(z["history"])
        log(f"resuming after stage {done} with {len(Y)} states")
    save = lambda d: np.savez(ckpt, X=np.array(X), Y=np.array(Y), done=d, history=np.array(history, dtype=object))
    if done < 1:
        rng = np.random.default_rng(seed)
        X, Y, _ = _collect(range(1000, 1000 + n_expert), lambda u, best: best, 0.15, rng)
        save(1); done = 1
        log(f"expert pass: {len(Y)} states ({time.time() - t0:.0f}s)")
    model = _fit(X, Y, seed)
    for r in range(done - 1, dagger_rounds):
        rng = np.random.default_rng(seed + r + 1)
        base = 2000 + 100 * r
        X2, Y2, crashes = _collect(range(base, base + n_dagger), lambda u, best: model.decide(u)[0], 0.05, rng)
        X += X2; Y += Y2
        history.append(f"round {r + 1}: +{len(Y2)} states, crashed {crashes}/{n_dagger}")
        save(r + 2)
        model = _fit(X, Y, seed)
        log(f"DAgger round {r + 1}: +{len(Y2)} states, small model crashed {crashes}/{n_dagger} ({time.time() - t0:.0f}s)")
    model.save()
    Yarr, Xarr = np.array(Y), np.array(X)
    agree = float(np.mean(np.argmax(model.reg.predict(Xarr), 1) == np.argmax(Yarr, 1)))
    mlp = model.reg[-1]
    meta = {"states": len(Y), "agreement_with_planner": round(agree, 3),
            "params": int(sum(w.size for w in mlp.coefs_) + sum(b.size for b in mlp.intercepts_)),
            "features": int(Xarr.shape[1]), "dagger": [str(h) for h in history]}
    (MODEL_DIR / "edge_meta.json").write_text(json.dumps(meta, indent=2))
    log(f"agreement with planner: {agree:.1%}, {meta['params']} parameters")
    return meta
