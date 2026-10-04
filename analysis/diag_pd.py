import sys
sys.path.insert(0, '/Users/yumx/code/x1_DM/analysis')
import numpy as np
from sim2sim_validate import X1Sim, HOME_QPOS, X1_DOF_ORDER

sim = X1Sim('/Users/yumx/code/x1_DM/analysis/x1_train_sim.xml', pd_scale=1.0)
n_phys = int(round((1/30.0) / sim.model.opt.timestep))

sim.reset(qpos_dof=HOME_QPOS)
print("after reset: root_h =", sim.data.qpos[2])
print("feet z:", [round(sim.data.xpos[i][2], 3) for i in sim.key_body_ids[:2]])

# free settle WITHOUT control for 1s to see natural pose
for _ in range(30 * n_phys):
    sim.data.ctrl[:] = 0
    sim.mujoco.mj_step(sim.model, sim.data)
print("after 1s free fall: root_h =", round(sim.data.qpos[2], 3))

# PD hold with per-joint tracking printout
sim.reset(qpos_dof=HOME_QPOS)
target = HOME_QPOS.copy()
worst = []
for step in range(int(3 * 30)):
    for _ in range(n_phys):
        sim.apply_action(target)
        sim.mujoco.mj_step(sim.model, sim.data)
    if step in (30, 60, 89):
        dof = sim.qpos2dof()
        err = dof - target
        idx = np.argsort(-np.abs(err))[:6]
        print(f"t={step/30:.1f}s root_h={sim.data.qpos[2]:.3f} worst joints:")
        for i in idx:
            print(f"   {X1_DOF_ORDER[i]:32s} dof={dof[i]:+.3f} tgt={target[i]:+.3f} err={err[i]:+.3f} kp={sim.kp[i]:.0f} gear={sim.gear[i]:.0f}")
        # torque saturation check
        dof_vel = sim.qvel2dofvel()
        tau = sim.kp * (target - dof) - sim.kd * dof_vel
        sat = np.abs(tau) > sim.gear
        print(f"   saturated joints: {int(sat.sum())}/29")
