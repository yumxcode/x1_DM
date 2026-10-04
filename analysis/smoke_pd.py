import sys
sys.path.insert(0, '/Users/yumx/code/x1_DM/analysis')
import numpy as np
from sim2sim_validate import X1Sim, HOME_QPOS

sim = X1Sim('/Users/yumx/code/x1_DM/analysis/x1_train_sim.xml', pd_scale=1.0)
print("timestep:", sim.model.opt.timestep, "kp[:3]:", sim.kp[:3], "kd[:3]:", sim.kd[:3])
n_phys = int(round((1/30.0) / sim.model.opt.timestep))
print("phys steps per ctrl step:", n_phys)

sim.reset(qpos_dof=HOME_QPOS)
target = HOME_QPOS.copy()
t = 0.0
hs, errs = [], []
for step in range(int(5 * 30)):
    for _ in range(n_phys):
        sim.apply_action(target)
        sim.mujoco.mj_step(sim.model, sim.data)
        t += sim.model.opt.timestep
    err = np.abs(sim.qpos2dof() - target).max()
    hs.append(sim.data.qpos[2]); errs.append(err)
print(f"after 5s PD-hold: root_h={hs[-1]:.4f} (start 0.6016), min={min(hs):.4f}, max_dof_err={max(errs):.4f} rad")
print("PASS" if hs[-1] > 0.5 and max(errs) < 0.1 else "FAIL")

obs = sim.build_obs()
print("obs dim:", obs.shape, "finite:", np.isfinite(obs).all())
