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

f0 = 40
sim.reset(qpos_dof=dof_ref[f0], root_pos=root_ref[f0])
hist = []
for k in range(f0, f0 + 90):  # 3s
    tgt = dof_ref[k]
    for _ in range(n_phys):
        sim.apply_action(tgt)
        sim.mujoco.mj_step(sim.model, sim.data)
    hist.append(np.abs(sim.qpos2dof() - tgt))

H = np.array(hist)  # (T, 29)
pre = H[:25]  # before falling (~0.8s, root still high)
names = X1_DOF_ORDER
print("pre-fall per-joint mean err (rad), sorted:")
order = np.argsort(-pre.mean(0))
for i in order[:12]:
    grp = 'leg' if ('hip' in names[i] or 'knee' in names[i] or 'ankle' in names[i]) else 'arm/waist'
    print(f"  {names[i]:30s} {pre[:, i].mean():.3f}  ({grp}, kp={sim.kp[i]:.0f}, gear={sim.gear[i]:.0f})")
print("\nlegs mean:", pre[:, 17:].mean(), " arms/waist mean:", pre[:, :17].mean())
print("overall pre-fall mean_max_err:", pre.max(1).mean())
