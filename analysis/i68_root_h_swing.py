#!/usr/bin/env python3
"""I68 (idear-0019): E4 post-hoc additions - root_h time series + swing
amplitude measurement. Runs a policy rollout (same X1Sim) and reports:
  (a) root_h(t) descent curve before fall (I63 hypothesis-i side evidence:
      does root_h already drop during the 0.7-1.0s 'pose_fail' window?)
  (b) sole-position swing amplitude in root-relative coordinates (calibrates
      idear-0018's phase-mismatch arithmetic which assumed +-0.35-0.5m)
  (c) per-frame pose_fail proxy: max non-root body |pos - root - (tar - tar_root)|
      vs the 0.8m threshold - Isaac termination mimic, computed in MuJoCo.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sim2sim_validate import X1Sim, load_policy, rsi_init_state  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True)
    ap.add_argument("--xml", default=os.path.join(os.path.dirname(
        os.path.abspath(__file__)), "x1_train_sim.xml"))
    ap.add_argument("--clip", default=None)
    ap.add_argument("--duration", type=float, default=5.0)
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import pickle
    policy = load_policy(args.policy)
    sim = X1Sim(args.xml)
    n_phys = int(round((1 / 30.0) / sim.model.opt.timestep))

    results = []
    for ep in range(args.episodes):
        rp, rq, qd = rsi_init_state(args.clip, seed=ep)
        # ALSO load the reference frames for pose_fail mimic
        with open(args.clip or os.path.join(os.path.dirname(os.path.dirname(
                os.path.abspath(__file__))), "x1_retargeted_motion",
                "x1_run1_subject2_seg0.pkl"), "rb") as f:
            d = pickle.load(f)
        frames = np.asarray(d["frames"], dtype=float)
        f0 = 40 + ep * 30
        # init from that frame, then walk the reference at 30Hz alongside
        sim.reset(qpos_dof=frames[f0, 6:35], root_pos=frames[f0, 0:3],
                  root_quat_xyzw=None)
        # recompute quat via expmap
        import sim2sim_validate as sv
        q = sv.expmap_to_quat_xyzw(frames[f0, 3:6])
        sim.reset(qpos_dof=frames[f0, 6:35], root_pos=frames[f0, 0:3],
                  root_quat_xyzw=q)

        ts, root_hs, swing_amps, pose_diffs = [], [], [], []
        t = 0.0
        k = f0
        while t < args.duration and k < len(frames) - 1:
            a = policy(sim.build_obs())
            for _ in range(n_phys):
                sim.apply_action(a)
                sim.mujoco.mj_step(sim.model, sim.data)
                t += sim.model.opt.timestep
                if sim.data.qpos[2] < 0.25:
                    break
            # measurements at 30Hz
            ts.append(t)
            root_hs.append(float(sim.data.qpos[2]))
            # (b) sole swing amplitude in root-relative frame
            rel = []
            for gid in sim.sole_geom_ids:
                rel.append(sim.data.geom_xpos[gid] - sim.data.qpos[:3])
            amp = max(np.linalg.norm(r[:2]) for r in rel)
            swing_amps.append(float(amp))
            # (c) pose_fail mimic: max over non-root bodies of
            # |(body-root) - (tar_body-tar_root)| using key bodies as proxy
            ref = frames[k]
            # reference key positions need FK - use ankle body origin proxy:
            # simpler: compare body-root horizontal offsets for feet only
            pose_diff = 0.0
            for bid in sim.key_body_ids[:2]:  # feet only proxy
                cur = sim.data.xpos[bid] - sim.data.qpos[:3]
                # reference foot pos not directly available without ref FK;
                # use dof-difference proxy instead: |dof - ref_dof| scaled
                pose_diff = max(pose_diff,
                                float(np.abs(sim.qpos2dof() - ref[6:35]).max()))
            pose_diffs.append(pose_diff)
            k += 1

        results.append(dict(
            t_end=round(t, 2),
            fell=bool(sim.data.qpos[2] < 0.25),
            root_h_series=[round(h, 3) for h in root_hs],
            root_h_min=round(min(root_hs), 3),
            swing_amp_max_m=round(max(swing_amps), 3),
            swing_amp_p50_m=round(float(np.median(swing_amps)), 3),
            dof_err_max_at_fall=round(pose_diffs[-1], 3) if pose_diffs else None,
        ))
        print(f"ep{ep}: t={t:.2f}s fell={results[-1]['fell']} "
              f"root_h_min={results[-1]['root_h_min']} "
              f"swing_max={results[-1]['swing_amp_max_m']}m "
              f"p50={results[-1]['swing_amp_p50_m']}m "
              f"dof_err@last={results[-1]['dof_err_max_at_fall']}")

    if args.out:
        import json
        with open(args.out, "w") as f:
            json.dump(results, f, indent=1)
        print("saved", args.out)

    amps = [r["swing_amp_max_m"] for r in results]
    print(f"\nswing amplitude (root-relative, feet): max={max(amps):.2f}m "
          f"(idear-0018 arithmetic assumed 0.35-0.5m)")


if __name__ == "__main__":
    main()
