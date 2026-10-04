import sys
sys.path.insert(0, '/Users/yumx/code/x1_DM/analysis')
import numpy as np
from sim2sim_validate import X1Sim, HOME_QPOS, X1_DOF_ORDER

sim = X1Sim('/Users/yumx/code/x1_DM/analysis/x1_train_sim.xml', pd_scale=1.0)
n_phys = int(round((1/30.0) / sim.model.opt.timestep))

sim.reset(qpos_dof=HOME_QPOS)
target = HOME_QPOS.copy()
print("t     root_h  pitch(deg) feet_L_z feet_R_z knee_L knee_R hipL_p hipR_p contactL contactR")
for step in range(int(2.5 * 30)):
    for _ in range(n_phys):
        sim.apply_action(target)
        sim.mujoco.mj_step(sim.model, sim.data)
    if step % 5 == 0:
        q = sim.data.qpos
        # pitch from quat (wxyz)
        w, x, y, z = q[3], q[4], q[5], q[6]
        pitch = np.degrees(np.arctan2(2*(w*y - z*x), 1 - 2*(y*y + x*x)))
        dof = sim.qpos2dof()
        i = lambda n: X1_DOF_ORDER.index(n)
        ncf = sim.data.ncon
        fl = sim.data.xpos[sim.key_body_ids[0]][2]
        fr = sim.data.xpos[sim.key_body_ids[1]][2]
        print(f"{step/30:5.2f} {q[2]:7.3f} {pitch:9.1f} {fl:8.3f} {fr:8.3f} "
              f"{dof[i('left_knee_pitch_joint')]:+.3f} {dof[i('right_knee_pitch_joint')]:+.3f} "
              f"{dof[i('left_hip_pitch_joint')]:+.3f} {dof[i('right_hip_pitch_joint')]:+.3f} ncon={ncf}")
