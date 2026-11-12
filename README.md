# Edge or Cloud

**Where should a self-driving car think: on the car, or in the cloud? I measured it over 360 simulated drives.**

[![ci](https://github.com/GowthamReddyKotha14/edge-or-cloud/actions/workflows/ci.yml/badge.svg)](https://github.com/GowthamReddyKotha14/edge-or-cloud/actions/workflows/ci.yml)

**[▶ Watch the replays](https://gowthamreddykotha14.github.io/edge-or-cloud/)**: same traffic, different strategy, real recorded drives.

A car makes five driving decisions a second, each with a hard 150 ms deadline. It has two brains:

- **Edge (on the car):** a small neural network. It answers in under 1 ms but is less reliable.
- **Cloud (on a server):** a digital-twin planner that copies the surrounding traffic and test-drives every possible maneuver 1.6 s into the future. It's much safer, but takes about 70 ms to think, plus the trip across the mobile network.

Four strategies drove the **same 30 traffic scenarios** under three network conditions.

## Results

Crashes in 30 drives (95% confidence range in brackets):

| Strategy | Good 5G | Busy 4G | Weak signal |
|---|---|---|---|
| Edge only | 7 (12–41%) | 7 (12–41%) | 7 (12–41%) |
| Cloud only | **1** (1–17%) | 12 (25–58%) | **24** (63–90%) |
| Cloud + edge backup | **1** (1–17%) | 4 (5–30%) | **6** (10–37%) |
| Adaptive switcher | **1**, asks 91% | **2**, asks 90% | 7, asks 43% |

**What the results show:**

- **Relying only on the cloud is a trap.** It's the best strategy on 5G and the worst on a weak signal. 75% of answers arrive late, the car keeps doing whatever it was already doing, and 24 of 30 drives crash. That's 3× worse than having no cloud at all.
- **An edge backup is what makes the cloud safe to use.** With one, using the cloud is never worse than going without it.
- **The adaptive switcher** asks the cloud only when the edge model is unsure *and* an answer has a fair chance of arriving. It matches the best safety on 5G and 4G with about 10% fewer requests. On a weak signal it sends 57% fewer requests for similar safety.
- **Small differences are noise.** With 30 drives per cell, 2 vs 4 crashes is within the confidence range; 24 vs 6 is not.

## How it works

| Piece | What it does |
|---|---|
| `edgecloud/sim.py` | highway-env setup (3 lanes, dense traffic, 5 decisions/s), the **digital-twin planner**, and sensor features |
| `edgecloud/edge.py` | The edge model: a 2,789-parameter network **distilled** from the planner's scores, refined with **DAgger** |
| `edgecloud/network.py` | Network model: log-normal round trips, burst "bad streaks" (a tunnel, a busy cell), and lost packets |
| `edgecloud/strategies.py` | Edge only, cloud only, cloud + edge backup, and the adaptive switcher |
| `edgecloud/bench.py` | Closed-loop benchmark with paired traffic seeds; each cell is saved so long runs can resume |

## Engineering notes: what went wrong and what fixed it

1. **The car couldn't brake.** highway-env's default controlled car can't go below 20 m/s while traffic drives at about 21 m/s, so many crashes were unavoidable whatever the decision. Allowing speeds down to 10 m/s cut planner crashes from 9/24 to 2/24.
2. **The digital twin was too slow.** Copying the whole simulator per look-ahead took 170 ms, longer than a decision is allowed. Twinning only traffic within 70 m and stepping the physics directly made it **12× faster**.
3. **The first edge model learned "just drive fast"** and crashed in 12 of 12 test drives. Five model designs all barely beat driving straight, so the issue wasn't model size; the inputs couldn't show the danger. Adding *mid-lane-change* and *car cutting in* signals plus time-to-collision, learning the planner's **scores** rather than just its choice, and three DAgger rounds brought it to 9 crashes in 24 drives.
4. **My first adaptive switcher was too cautious to help.** It asked only when the 90th-percentile round trip fit the deadline, so on busy 4G it *never* asked and did no better than having no cloud. With a backup in place, a late answer only costs bandwidth, so the right question is "is there a fair chance?" Asking when there's at least a 25% chance cut 4G crashes from 7 to 2.
5. **Crash rate alone is misleading.** "Always slow down" never crashes, at 11 m/s. Every result reports speed too.

## Run it

```bash
pip install -e ".[dev]"
pytest -q                                        # 9 tests
edgecloud summary                                # print the saved results grid
edgecloud bench 4g "Adaptive switcher"           # re-run a cell (about 2 minutes)
edgecloud measure https://www.google.com --n 60  # record YOUR real network latency as a profile
edgecloud train                                  # re-distill the edge model (about 10 minutes, resumable)
python scripts/export_site_data.py               # rebuild the showcase site's data
```

## Honest limits

- The three network profiles are **modeling assumptions** for illustration, not measurements. `edgecloud measure` records your real connection and fits a profile from it.
- 30 drives per cell, so only large differences are conclusive (see the confidence ranges above).
- Cloud compute (about 70 ms) was measured on one laptop-class CPU. A real server would be faster, which favors cloud strategies.
- highway-env is a simplified simulator: good for comparing strategies, not for certifying a vehicle.

## Related research

This connects to my master's research at BGSU: *Agentic AI precision control using digital twins in edge AI for autonomous vehicles*.

## License

MIT
