# R006: P0 根因闭环（AMPEnv 空壳）+ r19 task 通道激活实证 + 算力再次耗尽阻塞

- 报告 ID: R006-dm-r19-taskchannel-live-blocked-20261005_1725
- 作者: dm_worker
- 时间: 2026-10-05 17:25 (本地)
- 状态: 已观测（r18/r19 运行时日志 + 代码级证据）
- 关联产物: dm-results-R005；dm-ideas-idear-0015
- 实验 ID: TASK_20261005_045 (r18, 主动停), TASK_20261005_046 (r19, 被杀@36%)

## 1. P0 根因闭环（AMPEnv 空壳实锤，三方证据）

idear-0015 的根因分析**双重实锤验证**：
1. **代码级**：`mimickit/envs/amp_env.py` L280-281 `_update_reward(self): return`（空壳）；SMPEnv 无覆盖（grep 确认）→ MRO 解析 `sim_env._post_physics_step` L176 的调用到空壳 → `_reward_buf` 恒 0
2. **r18 运行时**（I48 修复后日志不再恒等）：iter 329 处 `Task_Reward_Mean=0.000` vs `Smp_Reward_Mean=0.344`——直接观测通道死亡
3. **r19 运行时**（I51 修复 @fce3a6a：smp_env 覆盖 _update_reward 直调 DeepMimicEnv）：`Task_Reward_Mean=0.38` ≠ Smp 0.411——**通道复活**

**历史改写**（I54 采纳）：r4-r18 全部 19 轮训练实为**纯 SMP 奖励**训练；task_weight 0.5 从未生效；r4 解冻归因修正为 scale+Adam（task 贡献为零）。

## 2. r19 中途证据（36% 被杀前）——task 通道改变训练动力学（已观测）

| 指标 | r14/r17（纯 SMP） | r19（task 激活，iter 594-820） |
|---|---|---|
| Test_Return | **恒 0**（19 轮全部） | **14.6→11.5 非零**（227/227 行） |
| Train_Return | 恒 0 | 高至 26.1 |
| Task_Reward_Mean | （恒等于 Smp） | 0.356-0.395 独立通道 |
| Smp_Reward_Mean | 0.41 平台 | 0.37-0.437 共振上升 |

**首次非零回报**是 task 奖励真实塑形的直接证据（return tracker 累计的是 env 奖励——此前恒 0 正因通道死）。曲线见 `analysis/r19_curves.json`。

注意（推断）：Test_Return 14.6→11.5 微降而每步奖励上升，或为探索/方差；36% 处的轨迹不足以外推终态，I53 三档判定（ep_len≥90/60-90/~40）**未测**。

## 3. 阻塞：全部账号 19-28 算力耗尽（已观测）

r19 于 107M/300M（36%，iter 820）被 SIGKILL（第 6 次余额击杀：r5/r13/r15/r16/r19 + r18 主动止损）；账号 28 probe 确认 1002056。**权重丢失**（first-slot 只在完成时落盘——这是该设计在余额不确定账号上的已知代价，r14/r17 完整跑通时无此问题）。

**解锁条件：充值或新增账号** → r20 一键启动（配方即 r19：分支 @fce3a6a、300M、任务 JSON `.repos/task-r19.json` 复用改名）。r20 若完成，EVAL ep_len 即得 I53 判定。

## 4. idear-0015 建议处置

| ID | 内容 | 处置 |
|---|---|---|
| I51 | smp_env 覆盖 _update_reward 跳过 AMP 空壳 | **accepted 已执行已验证**（@fce3a6a + r19 live） |
| I52 | 二级断点表 | superseded（根因已实锤，无需逐级断点） |
| I53 | r18/r19 预注册三档判读 | accepted；r19 36% 未测，r20 待判定 |
| I54 | 历史图景重写结算表 | accepted（本报告 §1 + R005 修正） |

## 5. 下一步（阻塞解除后）

1. r20（=r19 配方 300M）→ EVAL ep_len → I53 判定
2. 若 ep_len 显著改善 → E4''' 表型 + mesh 渲染（用户指令修订 2）→ 验收冲刺
3. 若 ~40 → E8 四格消融（方差/上限）

## 6. 证据文件

- analysis/r19_curves.json（227 行曲线：smp/task/test_ret/train_ret）
- .repos/r19final.json / r18log1.json（运行时日志提取）
- 分支 dm/smp-reward-fix：fce3a6a（I51）+ a3aa96a（P0diag）+ 2664618（I48）
- 工作区 @ 8017c83（R005）+ 本报告提交
