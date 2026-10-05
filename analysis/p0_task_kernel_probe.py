"""P0 root-cause probe (offline): compute the DeepMimic task reward kernel
on REAL data windows to separate kernel/config issues from runtime wiring.

Cases:
  A. perfect tracking (state == ref, same frame)   -> expect ~1.0
  B. one-step-ahead (state=f_k, ref=f_{k+1})       -> expect 0.3-0.7 (jogging)
  C. ref frozen at episode start (state=f_k, ref=f_0), k=30 -> jogging drift
Uses the actual compute_reward jit fn from the shim + smp_x1_env.yaml scales.
"""
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, "/Users/yumx/code/x1_DM/.repos/mimickit_shim/mimickit")
os.chdir("/Users/yumx/code/x1_DM/.repos/mimickit_shim")
from envs import task_reward_fn as dm_env

exec(open("/Users/yumx/code/x1_DM/analysis/e6b_reprojection_test.py")
     .read().split("def main()")[0], globals())

fk = FK(XML)
fr = load_clip(sorted(__import__("glob").glob(os.path.join(REPO, "x1_retargeted_motion", "*.pkl")))[0])
pos, quat, vel, ang, dof, dof_vel = build_sequences(fr)

# joint weights from smp_x1_env.yaml joint_err_w (29 entries, same order)
JW = [1.0,1.0,1.0, 0.5,0.5,0.5,0.5,0.5,0.25,0.25, 0.5,0.5,0.5,0.5,0.5,0.25,0.25,
      1.0,1.0,1.0,1.0,1.0,1.0, 1.0,1.0,1.0,1.0,1.0,1.0]
jw = torch.tensor(JW, dtype=torch.float32)
dw = torch.tensor(JW, dtype=torch.float32)

def frame_tensors(k):
    kp, _ = fk.frame(pos[k], quat[k], dof[k])
    return dict(
        root_pos=torch.tensor(pos[k], dtype=torch.float32),
        root_rot=torch.tensor(quat[k], dtype=torch.float32),
        root_vel=torch.tensor(vel[k], dtype=torch.float32),
        root_ang_vel=torch.tensor(ang[k], dtype=torch.float32),
        joint_rot=torch.tensor(fk.joint_rots(dof[k]), dtype=torch.float32),
        dof_vel=torch.tensor(dof_vel[k], dtype=torch.float32),
        key_pos=torch.tensor(kp, dtype=torch.float32))

def task_reward(st, rf):
    return dm_env.compute_reward(
        root_pos=st["root_pos"][None], root_rot=st["root_rot"][None],
        root_vel=st["root_vel"][None], root_ang_vel=st["root_ang_vel"][None],
        joint_rot=st["joint_rot"][None], dof_vel=st["dof_vel"][None],
        key_pos=st["key_pos"][None],
        tar_root_pos=rf["root_pos"][None], tar_root_rot=rf["root_rot"][None],
        tar_root_vel=rf["root_vel"][None], tar_root_ang_vel=rf["root_ang_vel"][None],
        tar_joint_rot=rf["joint_rot"][None], tar_dof_vel=rf["dof_vel"][None],
        tar_key_pos=rf["key_pos"][None],
        joint_rot_err_w=jw, dof_err_w=dw,
        track_root_h=True, track_root=False,
        pose_w=0.5, vel_w=0.1, root_pose_w=0.15, root_vel_w=0.1, key_pos_w=0.15,
        pose_scale=0.25, vel_scale=0.01, root_pose_scale=5.0,
        root_vel_scale=1.0, key_pos_scale=10.0)

res = {}
f50 = frame_tensors(50)
res["A_perfect(f50 vs f50)"] = float(task_reward(f50, f50))
f51 = frame_tensors(51)
res["B_one_step(f50 vs f51)"] = float(task_reward(f50, f51))
f80 = frame_tensors(80)
res["C_frozen_ref(f80 vs f50, 1s drift)"] = float(task_reward(f80, f50))
for k in (100, 150, 200, 250):
    fk_k = frame_tensors(k)
    res[f"C_frozen_ref(f{k} vs f50)"] = float(task_reward(fk_k, f50))
# average over consecutive-frame pairs (steady-state tracking difficulty)
vals = [float(task_reward(frame_tensors(k), frame_tensors(k + 1)))
        for k in range(60, 160, 5)]
res["B_avg_consecutive_20pairs"] = float(np.mean(vals))

for k, v in res.items():
    print(f"{k:38s} {v:.4f}")
json.dump(res, open("/Users/yumx/code/x1_DM/analysis/p0_task_kernel_probe.json", "w"), indent=1)
print("saved analysis/p0_task_kernel_probe.json")
