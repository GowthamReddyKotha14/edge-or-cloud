"""edgecloud: command-line entry point.

  edgecloud train                      distill the edge model from the planner (resumable)
  edgecloud bench NET STRATEGY ...     run benchmark cells, e.g. 5g "Cloud only"
  edgecloud summary                    print the results grid
  edgecloud measure URL [--n 60]       record REAL round trips from this machine
                                       and save them as a network profile
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
import urllib.request
from pathlib import Path


def cmd_measure(a) -> int:
    from .network import fit_profile
    samples, lost = [], 0
    for i in range(a.n):
        t0 = time.perf_counter()
        try:
            req = urllib.request.Request(a.url, method="HEAD", headers={"User-Agent": "edgecloud-measure"})
            urllib.request.urlopen(req, timeout=a.timeout).close()
            samples.append((time.perf_counter() - t0) * 1000)
        except Exception:
            lost += 1
        time.sleep(a.interval)
    if not samples:
        print("No successful requests. Check the URL and your connection.", file=sys.stderr)
        return 2
    prof = fit_profile(a.name, samples, lost)
    Path(a.out).write_text(json.dumps(prof.to_dict(), indent=2))
    s = sorted(samples)
    print(f"{len(samples)} ok, {lost} lost. median {s[len(s)//2]:.0f} ms, p95 {s[int(.95*(len(s)-1))]:.0f} ms -> {a.out}")
    return 0


def cmd_summary(a) -> int:
    from .bench import RESULTS_DIR
    rows = {}
    for f in sorted(RESULTS_DIR.glob("*__*.json")):
        if f.name.startswith("sweep"):
            continue
        s = json.loads(f.read_text())["summary"]
        rows[(s["network"], s["strategy"])] = s
    for (n, st), s in sorted(rows.items()):
        lo, hi = s["crash_ci"]
        print(f"{n:5} {st:22} crashes {s['crashes']:>2}/{s['episodes']} (95% CI {lo:.0%}-{hi:.0%})  "
              f"speed {s['speed']:.1f}  asks cloud {s['offload_rate']:.0%}  late {s['miss_rate']:.0%}")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="edgecloud", description="Edge-or-cloud decision offloading for autonomous driving")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("train").set_defaults(func=lambda a: (__import__("edgecloud.edge", fromlist=["train"]).train(), 0)[1])
    b = sub.add_parser("bench")
    b.add_argument("cells", nargs="+", help='pairs: network strategy, e.g. 5g "Cloud only"')
    b.set_defaults(func=lambda a: (__import__("edgecloud.bench", fromlist=["run_combo"]) and
                                   [__import__("edgecloud.bench", fromlist=["run_combo"]).run_combo(
                                       a.cells[i], a.cells[i + 1], range(800, 830), record_seeds=(800, 803))
                                    for i in range(0, len(a.cells), 2)], 0)[1])
    sub.add_parser("summary").set_defaults(func=cmd_summary)
    m = sub.add_parser("measure")
    m.add_argument("url")
    m.add_argument("--n", type=int, default=60)
    m.add_argument("--interval", type=float, default=0.2)
    m.add_argument("--timeout", type=float, default=3.0)
    m.add_argument("--name", default="My network")
    m.add_argument("--out", default="my_network.json")
    m.set_defaults(func=cmd_measure)
    a = p.parse_args(argv)
    return a.func(a)


if __name__ == "__main__":
    sys.exit(main())
