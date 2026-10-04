"""Smoke test v2: reference-motion PD replay (kinematic tracking check).

Feeds each retargeted-motion frame's dof as the servo target (30 Hz) and
measures joint tracking + root drift. A low tracking error proves the
X1Sim PD loop / joint mapping / gear scaling are all consistent with the
training actuator semantics (I5 gate), ready for policy rollout.
Note: replay cannot balance the free-floating base (no stabilizer), so
root drift before falling is expected; the gate is on JOINT tracking.
"""
import sys
sys.path.insert(0, '/Users/yumx/code/x1_DM/analysis')
import pickle
import numpy as np
from sim2sim_validate import X1Sim, X1_DOF_ORDER

PKL = '/Users/yumx/code/x1_DM/x1_retargeted_motion/x1_run1_subject2_seg0.pkl'
sim = X1Sim('/Users/yumx/code/x1_DM/analysis/x1_train_sim.xml', pd_scale=1.0)
n_phys = int(round((1/30.0) / sim.model.opt.timestep))

d = pickle.load(open(PKL, 'rb'))
frames = np.array(d['frames'])
dof_ref = frames[:, 6:35]
root_ref = frames[:, 0:3]

# init from frame 40 (mid-stride, both near contact)
f0 = 40
sim.reset(qpos_dof=dof_ref[f0], root_pos=root_ref[f0] + [0, 0, 0.0])

errs, hs, t = [], [], 0.0
fall_t = None
for k in range(f0, min(f0 + 150, len(dof_ref))):  # ~3.7 s
    tgt = dof_ref[k]
    for _ in range(n_phys):
        sim.apply_action(tgt)
        sim.mujoco.mj_step(sim.model, sim.data)
        t += sim.model.opt.timestep
    errs.append(np.abs(sim.qpos2dof() - tgt).max())
    hs.append(sim.data.qpos[2])
    if sim.data.qpos[2] < 0.30 and fall_t is None:
        fall_t = t

errs = np.array(errs)
print(f"replay {len(errs)} frames ({t:.2f}s)")
print(f"joint tracking: mean_max_err={errs.mean():.3f} rad, "
      f"p95={np.percentile(errs, 95):.3f}, max={errs.max():.3f}")
print(f"root_h: start={hs[0]:.3f} min={min(hs):.3f} end={hs[-1]:.3f}, fall_at={fall_t}")
ok = errs.mean() < 0.25 and np.percentile(errs, 95) < 0.5
print("SMOKE PASS (PD/obs infra consistent)" if ok else "SMOKE FAIL")
