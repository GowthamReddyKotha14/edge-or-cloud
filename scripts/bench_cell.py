"""Run benchmark cells:  python scripts/bench_cell.py 5g "Cloud only" [more cells...]"""
import sys
from edgecloud.bench import run_combo
SEEDS = range(800, 830)
REPLAY = (800, 803)
args = sys.argv[1:]
for i in range(0, len(args), 2):
    run_combo(args[i], args[i + 1], SEEDS, record_seeds=REPLAY)
