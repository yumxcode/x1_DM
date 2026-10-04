# E15: pose_scale genealogy (idear-0013 I47) — verdict

- 报告 ID: E15-dm-pose-scale-genealogy-20261005_0700
- 作者: dm_worker
- 方法: gh api 遍历 x1_mimicKit main 的 data/envs/*.yaml（25 个配置）grep pose_scale/root_vel_w/reward_pose_w

## 结果（已观测）

**全部 25 个 env 配置（add/amp/ase/deepmimic/smp/vault × g1/go2/humanoid/pi_plus/smpl/x1）完全一致**：
- reward_pose_scale: 0.25
- reward_root_vel_w: 0.1
- reward_pose_w: 0.5

## 判定

pose_scale=0.25 是 **MimicKit 全库统一传统**（含 g1 与 DeepMimic 系），非 x1 配置失误。idear-0013 的预警成立（"若 g1 也 0.25 则为 MimicKit 传统而非 x1 失误，改动需更谨慎"）。

对 I46 的含义：
- 方案 β（pose_scale 0.25→3.87 DeepMimic 原版公式）= 偏离全库惯例的单点改动，正当性下调，仅在态 B 且 α 无效且轨迹级核算支持时考虑
- 方案 α（root_vel_w 0.1→0.3）保持首选（同样偏离惯例但幅度小、直指速度弱惩罚）
