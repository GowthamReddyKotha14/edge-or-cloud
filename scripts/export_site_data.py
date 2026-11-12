"""Write docs/data.js from the saved benchmark results. Every number on the
showcase site comes from here."""
import glob, json, math
from pathlib import Path
import numpy as np
from edgecloud.bench import RESULTS_DIR
from edgecloud.network import PROFILES, Link
from edgecloud.strategies import DEADLINE_MS

ROOT = Path(__file__).resolve().parent.parent
grid = {}
for f in sorted(RESULTS_DIR.glob("*__*.json")):
    if f.name.startswith("sweep"):
        continue
    s = json.loads(f.read_text())["summary"]
    grid[f"{s['network']}|{s['strategy']}"] = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in s.items()
                                               if k in ("crashes", "episodes", "crash_rate", "crash_ci", "speed", "offload_rate", "miss_rate", "sources")}
    if s["strategy"] == "Cloud only" and s["network"] == "5g":
        compute = s["compute_ms"]
sweep = [json.loads(Path(f).read_text()) for f in glob.glob(str(RESULTS_DIR / "sweep__5g__*.json"))]
sweep = sorted(({"ask": round(x["offload_rate"], 3), "crashes": x["crashes"], "tau": x["tau"]} for x in sweep), key=lambda x: x["ask"])
sweep = [{"ask": 0.0, "crashes": grid["5g|Edge only"]["crashes"], "tau": None, "label": "Edge only"}] + sweep + \
        [{"ask": 1.0, "crashes": grid["5g|Cloud + edge backup"]["crashes"], "tau": None, "label": "Always ask"}]
hist = {}
bins = np.linspace(0, 400, 41)
for k, p in PROFILES.items():
    link = Link(p, seed=42)
    xs = np.array([link.sample() for _ in range(20000)])
    ok = xs[np.isfinite(xs)]
    hist[k] = {"name": p.name, "counts": np.histogram(np.clip(ok, 0, 399), bins)[0].tolist(),
               "median": round(float(np.median(ok)), 1), "p95": round(float(np.percentile(ok, 95)), 1),
               "lost": round(float(1 - len(ok) / len(xs)), 4),
               "on_time": round(float(np.mean(np.isfinite(xs) & (xs + np.median(compute) <= DEADLINE_MS))), 3)}
meta = json.loads((ROOT / "models" / "edge_meta.json").read_text())
replays = json.loads((RESULTS_DIR / "replays.json").read_text())
data = {"grid": grid, "sweep": sweep, "hist": hist, "bins": bins.tolist(), "deadline": DEADLINE_MS,
        "compute": {"median": round(float(np.median(compute)), 1), "p99": round(float(np.percentile(compute, 99)), 1)},
        "edge": meta, "replays": replays, "networks": {k: p.name for k, p in PROFILES.items()}}
(ROOT / "docs").mkdir(exist_ok=True)
(ROOT / "docs" / "data.js").write_text("window.EC = " + json.dumps(data, separators=(",", ":")) + ";\n")
print("data.js", (ROOT / "docs" / "data.js").stat().st_size, "bytes; compute median", data["compute"],
      {k: (v["median"], v["on_time"]) for k, v in hist.items()})
