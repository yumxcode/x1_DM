"""Policy loader smoke test with r4 initial checkpoint (untrained actor).
Expect: forward produces sane 29-dim actions; rollout runs; robot falls
(untrained policy) without numpy/torch errors. Gate = pipeline correctness.
"""
import sys
sys.path.insert(0, '/Users/yumx/code/x1_DM/analysis')
import numpy as np
from sim2sim_validate import X1Sim, HOME_QPOS, load_policy

policy = load_policy('/Users/yumx/code/x1_DM/checkpoints/r4_initial_model.pt')
sim = X1Sim('/Users/yumx/code/x1_DM/analysis/x1_train_sim.xml')
n_phys = int(round((1/30.0) / sim.model.opt.timestep))

sim.reset(qpos_dof=HOME_QPOS)
obs = sim.build_obs()
a = policy(obs)
print("action dim:", a.shape, "range:", a.min(), a.max(), "finite:", np.isfinite(a).all())

acts, hs = [], []
t = 0.0
while t < 8.0:
    a = policy(obs)
    acts.append(a.copy())
    for _ in range(n_phys):
        sim.apply_action(a)
        sim.mujoco.mj_step(sim.model, sim.data)
        t += sim.model.opt.timestep
        if sim.data.qpos[2] < 0.25:
            break
    obs = sim.build_obs()
    hs.append(sim.data.qpos[2])
    if sim.data.qpos[2] < 0.25:
        break

A = np.array(acts)
print(f"rollout {t:.2f}s, {len(acts)} ctrl steps, fell={t < 7.9}")
print(f"action stats: |a| mean={np.abs(A).mean():.3f} max={np.abs(A).max():.3f} std={A.std():.4f}")
print(f"root_h trajectory: {hs[0]:.3f} -> {hs[-1]:.3f}, min={min(hs):.3f}")
print("PASS (pipeline ok; fall expected for untrained policy)" if np.isfinite(A).all() else "FAIL")
