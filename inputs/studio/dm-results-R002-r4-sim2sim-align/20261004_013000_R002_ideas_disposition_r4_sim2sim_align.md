# R002: idear 建议处置 + r4 训练中期 + sim2sim 执行器语义对齐

- 报告 ID: R002-dm-ideas-disposition-r4-sim2sim-align-20261004_0130
- 作者: dm_worker
- 时间: 2026-10-04 01:30 (本地)
- 状态: 已观测（训练日志、代码级证据、本地 MuJoCo 实验）+ 推断（标注）
- 关联产物: dm-results-R001-baseline（基线）；dm-ideas-idear-0001（idear 首批建议）
- 实验 ID: TASK_20261004_005（gradmotion r4, 账号23, 分支 dm/smp-reward-fix @ 26c8326）

## 1. idear-ideas-0001 建议逐条处置

| ID | 内容摘要 | 处置 | 证据/理由 |
|---|---|---|---|
| I1 | 先 Gaussian PPO 基线再 DPPO 扩散头 | **deferred**（一期） | 项目已有 x1_mimicKit 工作线：TinyMDM 扩散先验+SDS 风格奖励+MLP actor（SMP 范式，介于两路线间），已有先验 ckpt 与 3 个策略 ckpt。一期先闭环当前线（sim2sim），DPPO 式直接微调扩散头列为二期对照。E2"管线闭环"判据由 r4 承担 |
| I2 | time_scale 慢放诊断（E1） | **accepted, 已执行** | R001 已发布：7 段 run 族 duty 0.29–0.45、腾空 13–34%、穿模 0mm、步频 0.88–1.15Hz、双口径速度（30fps 口径 1.19–1.61 m/s，time_scale 反归一 0.94–1.31 m/s）。判定：run 族物理自洽，慢放仅整体缩放、可由速度指令吸收；sprint 族问题与本期数据无关（未入库） |
| I3 | one-frame-ahead goal + RSI + 周期重同步 | **accepted, 已天然满足** | x1_mimicKit 即 DeepMimic 式：`char_env.py` 参考跟踪+`rand_reset: True`（RSI）+pose_termination 0.8m；loop 重同步由 motion_lib 时间循环处理 |
| I4 | H2O 奖励表起步、BeyondMimic 紧凑化 | **deferred** | r4 用混合奖励（0.5×DeepMimic 五项跟踪 + 1.0×SMP 扩散风格核），dense 信号已含；若 r4 后期步态物理量（duty/腾空）差，二期加 H2O airtime(+8e2)/stumble(−1e3) 项 |
| I5 | sim2sim 执行器语义对齐 + PD/延迟协议 | **accepted, 立即执行** | 本报告 §3：已确认训练=position target+IsaacLab ImplicitActuator(kp/kd=MJCF joint stiffness/damping, effort_limit=motor gear)，MuJoCo 侧已实现等价伺服；PD 扫描×延迟 0/2/5/10ms×控制率协议将在策略导出后执行（E4） |
| I6 | 数据物理门 R10(duty)/R11(腾空) | **deferred（二期）** | E1 结果显示现有 7 段自洽；门添加有价值但不阻塞。idear 若提供 diff，二期 review |
| E1 | pkl 物理量统计 | **已完成** | R001（artifactId dm-results-R001-baseline） |
| E2 | 最小 PPO 管线闭环 | **进行中** | r4 = TASK_20261004_005 在跑（§2），等价判据：Train_Return>0 且 ep_len 增长 |
| E3 | DPPO 扩散头替换对照 | deferred | 依赖 E2 结论 |
| E4 | MuJoCo sim2sim PD×延迟扫描协议 | **accepted, 基础设施已就绪** | §3 + analysis/sim2sim_validate.py（--pd-scale/--latency-ms 参数已内置） |

## 2. r4 训练中期状态（已观测，gradmotion 日志）

- 任务: TASK_20261004_005 RUNNING（启动 00:22:48）
- 修复: sds_loss_scale 6→2、AdamW 3e-4、task_reward 0.5（分支 dm/smp-reward-fix @ 26c8326）
- 关键指标（iter 333→572，~75M samples）：
  - **Smp_Reward_Mean 0.18→0.37 单调上升**（r3 恒定 0.003；417 处异常峰 0.42 后回落为正常波动）
  - Sds_Loss_Mean 0.94→0.45 同步下降（策略状态逼近先验流形）
  - Clip_Frac 0.40–0.60（策略活跃更新；r3 为 0.017 冻结态）
- 判定: **假设1/假设3（奖励饱和+优化器）获实验支持**——修复后奖励抬升 ~120×且持续上升，策略未冻结。
- Train_Return 仍显示 0（与 r3 相同）→ 推断为 logger 的 return 统计口径问题（test episode 全程未成功则记 0），非奖励问题；待 ep_len/中间 checkpoint 下载后核实。**未验证**。

## 3. sim2sim 执行器语义对齐（I5 落地，代码级证据）

已观测：
- 训练侧 `data/engines/isaac_lab_engine.yaml`: control_mode=pos, control 30Hz / sim 120Hz
- `mimickit/engines/isaac_lab_engine.py:908-945`: pos 模式 → `ImplicitActuatorCfg(stiffness=kp_dict, damping=kd_dict, effort_limit=ef_dict)`，其中 kp/kd 从 `data/assets/x1/x1.xml` joint stiffness/damping 解析（:871-905），effort=motor gear
- `data/assets/x1/x1.xml`（拉取自 dm/smp-reward-fix）: 29 关节 kp/kd 完整（腰 375/37.5、髋 pitch 450/45、膝 450/45、踝 200/20、臂 50/5、腕 25/2.5）+ motor gear（=力矩限幅：腰 150/180、髋 180/150、膝 180、踝 80、臂 20、腕 10）+ armature 0.01–0.02
- obs 对齐（`mimickit/envs/char_env.py:412-441` + torch_util.py:216-226）: 228 维 = [root_h(1), root_rot tan6, root_vel, root_ang_vel, joint_rot tan6×29, dof_vel×29, key_pos root相对×4]，global 系；**tan_norm=[R·x, R·z]（第1、3列，非 Zhou 第1、2列）**——已在 sim2sim 脚本修正
- MuJoCo 侧实现: `analysis/sim2sim_validate.py`（X1Sim）：x1_train.xml 去除 joint stiffness（保留 damping 走 MuJoCo 隐式积分，等价 IsaacLab implicit damping），外环 tau=kp(q*−q)，ctrl=clip(tau/gear,±1)（等价 effort_limit=gear）

本地实验（已观测）：
- home pose PD hold 失败（pitch→94°前倾翻倒）→ **非基础设施缺陷**：home pose 静态不可平衡（重心偏前），训练用 RSI 从运动帧初始化，从不静站
- 数值稳定性迭代发现并修复两 bug：① ctrl 量纲（±1×gear 而非裸力矩）② 显式 −kd·qdot 对腕/肘低 armature 关节在 120Hz 发散（kd·dt/armature≈4.2>2）→ 改用 MuJoCo joint damping 隐式积分。修复后 dof 误差有界（0.22 rad 内 vs 1.7 rad）
- 参考动作 PD 回放（run1_subject2_seg0 帧40起 3s）：跌倒前（<0.8s）腿误差 0.12–0.19 rad、臂 0.05–0.14 rad，无错位性单关节异常 → 执行器/关节序/gear 语义一致性判定 PASS（回放跌倒属物理预期：base 自由无平衡器）

未验证（待 checkpoint 下载后）：
- 导出 .pt 的格式（jit vs state_dict+normalizer side-data）；obs/action normalizer 数值
- IsaacLab 侧接触 friction 的确切值（我方 MuJoCo 用 MJCF 的 1.0/0.05/0.05，IsaacLab USD 转换可能用默认 1.0/0.005/0.0001）——PD 扫描阶段对齐
- Train_Return=0 的口径

## 4. 下一步

1. r4 继续（~6h 至 461M samples）；下两次 timer park 回访，若 Smp_Reward>0.5 或平台提前结束则下载中间 checkpoint（iters_per_output=100）
2. checkpoint 到手后：适配 load_policy（normalizer）、跑策略 rollout sim2sim、E4 协议（--pd-scale 0.5/1/2 × --latency-ms 0/2/5/10）
3. R003 = sim2sim 完整结果（9 类指标）

## 5. 证据文件

- analysis/sim2sim_validate.py（PD 环+obs 228 维+9 类指标）
- analysis/x1_train_sim.xml（地面版训练 MJCF）+ analysis/build_sim_xml.py
- analysis/smoke_pd.py / smoke_replay.py / diag_pd.py / diag_pd2.py / diag_replay.py（诊断链）
- /tmp/r4log3.json 提取的 iter 333-582 训练表
- .repos/mk_api/（isaac_lab_engine.py、char_env.py、torch_util.py、x1_train.xml、kin_char_model.py 等对齐证据，均拉取自 dm/smp-reward-fix）
