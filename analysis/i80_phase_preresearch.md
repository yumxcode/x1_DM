# I80 预研：盲跟踪（phase_obs）改造可行性（idear-0022 I80，零算力）

- 作者: dm_worker; 时间: 2026-10-07; 依据: deepmimic_env.py @ dm/smp-reward-fix

## I80-2 prior 兼容断言核对（已观测，源码级）

1. **`_check_prior_env_config` 断言集**（smp_agent.py L89-95）：仅含
   `global_obs / root_height_obs / enable_tar_obs / num_disc_obs_steps / disc_dof_vel_obs`
   —— **不含 phase**。phase_obs 开关不触发断言。
2. **phase 不进 disc_obs**（amp_env.py 全文无 phase 引用；disc_obs=compute_tar_obs+compute_disc_vel_obs，
   二者均无 phase 输入）→ **prior（201/帧通道）无需重训**。
3. **`enable_phase_obs` 默认 True**（deepmimic_env.py L19：`env_config.get("enable_phase_obs", True)`）
   ——但 smp_x1_env.yaml L14 显式 `enable_phase_obs: False`。
   即：框架原生支持 phase obs（含 `compute_phase_obs` 编码函数 L585-600 与
   `num_phase_encoding` 位置编码），x1 只是配置关闭。**改造 = 改一行 yaml + obs 维数联动**。
4. obs 维数影响：phase_obs = 1 + 2×num_phase_encoding 维（num_phase_encoding 默认 0 → 1 维）。
   policy obs 228→229。actor 输入层自动适配（net_builder 按输入维建网）；**sim2sim 侧需同步**（228→229，
   MuJoCo 侧 phase 计算=参考时间归一化，需 E4 runner 携带参考相位——idear-0022 I80-3 成本点确认）。

## I80-1 构型偏差振荡/单调分解（见 i68_phase_decomp.py 输出）

判定：漂移时序若为周期振荡增长 → 相位失配机制（I61 改造有效预期高）；
若单调发散 → 姿态错误主导（phase 无效，转 task 权重/跟踪精度）。

## 结论（r27 方向备料）

- I61 改造工程上轻（yaml 一行+sim2sim 对齐）；prior 零重训。
- 若 r26（461M 第二种子）仍 ~23-40 档且 I80-1 显示振荡型漂移 → r27 = phase_obs 单变量为首选；
  若单调型 → r27 = task 权重消融或 DR（I81 条件触发）。
