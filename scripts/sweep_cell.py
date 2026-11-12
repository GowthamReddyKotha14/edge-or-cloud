"""Sweep the adaptive switcher's confidence gate:  python scripts/sweep_cell.py 5g 0.1 0.3"""
import json, sys
from edgecloud import bench
from edgecloud.network import PROFILES
from edgecloud.strategies import Adaptive
from edgecloud.edge import EdgeModel
SEEDS = range(800, 830)
nkey = sys.argv[1]
edge, hist = EdgeModel.load(), []
for tau in map(float, sys.argv[2:]):
    eps = [bench.run_episode(Adaptive(confidence=tau), PROFILES[nkey], s, edge, hist) for s in SEEDS]
    r = bench.summarize(eps)
    bench.RESULTS_DIR.mkdir(exist_ok=True)
    (bench.RESULTS_DIR / f"sweep__{nkey}__{tau}.json").write_text(json.dumps({"tau": tau, "network": nkey, **{k: r[k] for k in ("crashes", "episodes", "speed", "offload_rate", "miss_rate")}}))
    print(f"{nkey} tau={tau}: crashes {r['crashes']}/30 speed {r['speed']:.1f} asks cloud {r['offload_rate']:.0%} late {r['miss_rate']:.0%}", flush=True)
