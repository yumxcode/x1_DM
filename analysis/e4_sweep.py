#!/usr/bin/env python3
"""E4 sim2sim sweep runner (idear-0002 protocol, layer 3 subset + friction).

Grid: pd_scale {0.5, 1.0, 2.0} x latency {0, 2, 5, 10 ms} x friction {1.0}
      + baseline row with friction {0.4, 0.6, 0.8}
Baseline (pd=1.0, lat=0, fric=1.0) run with 3 episodes; other cells 1 episode.

Usage: python e4_sweep.py --policy <final.pt> --out e4_results.json
"""
import argparse
import itertools
import json
import os
import subprocess
import sys

ANALYSIS = os.path.dirname(os.path.abspath(__file__))


def run_cell(policy, xml, pd_scale, latency_ms, friction, duration, episodes, tag,
             init="rsi"):
    out_json = os.path.join(ANALYSIS, f"_e4_{tag}.json")
    cmd = [sys.executable, os.path.join(ANALYSIS, "sim2sim_validate.py"),
           "--policy", policy, "--xml", xml,
           "--pd-scale", str(pd_scale), "--latency-ms", str(latency_ms),
           "--friction", str(friction),
           "--duration", str(duration), "--episodes", str(episodes),
           "--init", init,
           "--out", out_json]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    if r.returncode != 0:
        print(f"[{tag}] FAILED rc={r.returncode}\n{r.stdout[-500:]}\n{r.stderr[-500:]}")
        return None
    with open(out_json) as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True)
    ap.add_argument("--xml", default=os.path.join(ANALYSIS, "x1_train_sim.xml"))
    ap.add_argument("--out", default=os.path.join(ANALYSIS, "e4_results.json"))
    ap.add_argument("--duration", type=float, default=20.0)
    args = ap.parse_args()

    # idear-0006 I21 protocol: RSI init default; friction axis prioritized
    # (known-unaligned item) before latency; home-pose control arm quantifies
    # transient pollution.
    cells = []
    cells.append(dict(pd=1.0, lat=0.0, fric=1.0, eps=3, tag="base"))         # RSI baseline
    cells.append(dict(pd=1.0, lat=0.0, fric=1.0, eps=1, tag="home",
                      init="home"))                                           # control arm
    for fric in (0.6, 0.8, 1.2):                                             # friction axis first
        cells.append(dict(pd=1.0, lat=0.0, fric=fric, eps=1, tag=f"fric{fric}"))
    for pd in (0.5, 2.0):
        cells.append(dict(pd=pd, lat=0.0, fric=1.0, eps=1, tag=f"pd{pd}"))
    for lat in (8.3, 16.7, 41.7):                                             # phys-step aligned (1/2/5 steps)
        cells.append(dict(pd=1.0, lat=lat, fric=1.0, eps=1, tag=f"lat{lat}"))

    results = []
    for c in cells:
        print(f"=== cell {c['tag']} (pd={c['pd']} lat={c['lat']} fric={c['fric']} init={c.get('init','rsi')}) ===")
        r = run_cell(args.policy, args.xml, c["pd"], c["lat"], c["fric"],
                     args.duration, c["eps"], c["tag"], init=c.get("init", "rsi"))
        if r is not None:
            entry = dict(cell=c, episodes=r)
            results.append(entry)
            for ep in r:
                print(f"  dur={ep['duration_s']}s fell={ep['fell']} "
                      f"v={ep['fwd_vel_mean']} duty={ep.get('duty_L')}")

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print("saved", args.out, f"({len(results)}/{len(cells)} cells)")

    # summary table
    print("\ncell | eps | mean_dur | fall_rate | v_mean | duty_L")
    for entry in results:
        eps = entry["episodes"]
        durs = [e["duration_s"] for e in eps]
        falls = sum(1 for e in eps if e["fell"])
        vs = [e["fwd_vel_mean"] for e in eps]
        duty = [e.get("duty_L") for e in eps if e.get("duty_L") is not None]
        print(f"{entry['cell']['tag']:8s} | {len(eps)} | "
              f"{sum(durs)/len(durs):.1f} | {falls}/{len(eps)} | "
              f"{sum(vs)/len(vs):.2f} | {duty[0] if duty else '-'}")


if __name__ == "__main__":
    main()
