"""Re-record the showcase replays (same seeds as the benchmark) and check they
end the same way the benchmark did."""
import json
from edgecloud.bench import RESULTS_DIR, STRATS, run_episode
from edgecloud.edge import EdgeModel
from edgecloud.network import PROFILES
SEEDS = (803,)
edge, hist, out, mismatch = EdgeModel.load(), [], {}, 0
for nkey in PROFILES:
    for name, cls in STRATS.items():
        ref = json.loads((RESULTS_DIR / f"{nkey}__{name.replace(' ', '_').replace('+', 'plus')}.json").read_text())["summary"]["per_episode"]
        for s in SEEDS:
            ep = run_episode(cls(), PROFILES[nkey], s, edge, hist, record=True)
            bench_crash = next(e["crashed"] for e in ref if e["seed"] == s)
            mismatch += ep.crashed != bench_crash
            out[f"{nkey}|{name}|{s}"] = {"crashed": ep.crashed, "frames": ep.frames}
            print(f"{nkey:5} {name:22} seed {s}: crashed={ep.crashed} (benchmark: {bench_crash})", flush=True)
(RESULTS_DIR / "replays.json").write_text(json.dumps(out, separators=(",", ":")))
print("mismatches:", mismatch)
