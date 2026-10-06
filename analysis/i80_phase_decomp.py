"""I80-1 (idear-0022): posture-drift decomposition - oscillatory (phase
mismatch) vs monotonic (pose error) - using i68 rollout data (r23 weights).

Uses the saved i68_r23.json root_h series + fresh rollouts capturing
body-root horizontal offsets (the pose_fail proxy) and analyzes:
  - zero-crossing rate of the detrended offset (oscillation indicator)
  - monotonic-fraction (fraction of steps moving away from ref)
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sim2sim_validate import X1Sim, load_policy, rsi_init_state, expmap_to_quat_xyzw  # noqa: E402

import pickle

POLICY = "/Users/yumx/code/x1_DM/checkpoints/r23_final.pt"


def rollout_with_ref(policy, sim, n_phys, clip_path, f0, duration=3.0):
    with open(clip_path, "rb") as f:
        d = pickle.load(f)
    frames = np.asarray(d["frames"], dtype=float)
    q = expmap_to_quat_xyzw(frames[f0, 3:6])
    sim.reset(qpos_dof=frames[f0, 6:35], root_pos=frames[f0, 0:3], root_quat_xyzw=q)
    # walk reference at 30Hz; record per-step: dof tracking error vector sign
    # structure: compare LEFT-KNEE (gait-phase-sensitive joint) error sign
    # over time - oscillating sign = phase mismatch; constant sign = drift
    import sim2sim_validate as sv
    knee_l = sv.X1_DOF_ORDER.index("left_hip_pitch_joint")
    hip_l = sv.X1_DOF_ORDER.index("left_hip_pitch_joint")
    knee_errs, ref_dk = [], []
    t = 0.0
    k = f0
    while t < duration and k < len(frames) - 1:
        a = policy(sim.build_obs())
        for _ in range(n_phys):
            sim.apply_action(a)
            sim.mujoco.mj_step(sim.model, sim.data)
            t += sim.model.opt.timestep
            if sim.data.qpos[2] < 0.25:
                break
        dof = sim.qpos2dof()
        knee_errs.append(float(dof[knee_l] - frames[k, 6 + knee_l]))
        ref_dk.append(float(frames[k, 6 + knee_l]))
        k += 1
    return np.array(knee_errs), np.array(ref_dk), t


def analyze(err, ref, tag):
    if len(err) < 20:
        print(f"{tag}: too short ({len(err)} steps)")
        return
    # detrend
    det = err - np.linspace(err[0], err[-1], len(err))
    # zero crossings of detrended signal
    zc = np.sum(np.diff(np.sign(det)) != 0)
    zc_rate = zc / (len(err) / 30.0)  # crossings per second
    # monotonic fraction of raw error
    mono = np.mean(np.sign(np.diff(err)) == np.sign(err[-1] - err[0]))
    # oscillation of the REFERENCE itself (gait cycle baseline)
    ref_det = ref - np.linspace(ref[0], ref[-1], len(ref))
    ref_zc = np.sum(np.diff(np.sign(ref_det)) != 0)
    print(f"{tag}: n={len(err)} zc_rate={zc_rate:.1f}/s mono_frac={mono:.2f} "
          f"| ref_zc_rate={ref_zc/(len(ref)/30.0):.1f}/s (gait baseline)")
    return zc_rate, mono


def main():
    policy = load_policy(POLICY)
    sim = X1Sim("/Users/yumx/code/x1_DM/analysis/x1_train_sim.xml")
    n_phys = int(round((1 / 30.0) / sim.model.opt.timestep))
    clip = "/Users/yumx/code/x1_DM/x1_retargeted_motion/x1_run1_subject2_seg0.pkl"
    results = []
    for f0 in (40, 100, 160):
        err, ref, t = rollout_with_ref(policy, sim, n_phys, clip, f0)
        r = analyze(err, ref, f"r23 f0={f0}")
        results.append(dict(f0=f0, t_end=round(t, 2), err=list(map(float, err)),
                            ref=list(map(float, ref))))
    json.dump(results, open("/Users/yumx/code/x1_DM/analysis/i80_phase_decomp.json", "w"))
    print("saved analysis/i80_phase_decomp.json")


if __name__ == "__main__":
    main()
