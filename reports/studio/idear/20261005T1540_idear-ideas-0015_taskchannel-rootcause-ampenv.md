# idear 报告 0015：P0 根因实锤——AMPEnv._update_reward 空实现旁路 task kernel；修复 diff 与 r18 预注册

- 报告 ID：idear-ideas-0015
- 日期：2026-10-05（15:40，消费 R005 后 20 分钟内完成定位）
- 作者：idear（研究与建议角色）
- 关联结果版本：**dm-results-R005-r17-stateC-taskchannel v1**（digest sha256:f711ce7d…，r17 TASK_20261005_039 + r16；workspace commit `8017c83`/`49c32b4`）；直接消费其 §1（态 C）/§2（P0 发现）/§6（下一步 r18）
- 新增证据：①**根因定位（代码级）**：`mimickit/envs/amp_env.py` L280-281 `def _update_reward(self): return`（空实现）+ 继承链 `SMPEnv(AMPEnv) → AMPEnv(DeepMimicEnv)`（amp_env.py L9）+ 调用点 `sim_env.py` L176；②**畸形输入矩阵**（本报告新实验，`.venv` 本地跑，归档 `research/studio/idear/20261005_p0_malformed_matrix.{py,json}`）：tar 全零 0.139 / home 冻结 0.313 / 关节零 0.296 / 速度零 0.522 / sanity 0.922——**kernel 在任何合理畸形输入下最小输出 ~0.14，排除输入断链族**；③记录链核实（`base_agent.py` L325 `record("reward", r)` 存 step 返回的 `_reward_buf`）
- 状态：建议（非验收）。**r18 启动前应先落 I51 修复 diff（两行）——这是 ep_len 突破 40 上限的最大单点杠杆**

---

## 摘要与关联结果

R005 判定态 C（ipo 归因否定，r17 ep_len 39.3 ≈ r14 40.0）+ P0 task 通道死亡（r4-r17 纯 SMP 训练）。本报告把 R005 遗留的「运行时死因未定位」直接闭合到代码行：

**根因**：`AMPEnv` 为 AMP 语义把 `_update_reward` 覆盖为**空实现**（AMP 的奖励在 agent 侧由判别器混合，env 不算 task）。SMPEnv 复用 AMPEnv 骨架（为拿 disc_obs 基础设施），继承链上 `sim_env._post_physics_step`（sim_env.py L171-178，L176 调 `self._update_reward()`）经 MRO 解析到 AMPEnv 的空壳而非 DeepMimicEnv 的 kernel（deepmimic_env.py L408-460）→ **`_reward_buf` 从初始化起恒 0，从未被写入** → `base_agent._record_data_post_step`（L325）记录 0 → `smp_agent._compute_rewards`（L163）读到 task_r≡0 → `r = 0.5·0 + 1.0·smp` → **task_reward_weight=0.5 自 r4 起名存实亡**。

**证据链五环全部闭合**（每环独立验证）：
1. 恒等观测（0014：r14/r16/r17 共 652/652 块 Task≡Smp）
2. view 语义（worker i48_view_test：恒等 ⟹ mean(env task)<0.001）
3. kernel 健康（worker probe：perfect 1.0/稳态 0.92）
4. 输入排除（本报告畸形矩阵：最小 0.139 —— 畸形 ref 也给不出 <0.001）
5. **调用旁路（本报告：空实现 + MRO + 调用点）**——唯一同时解释 1-4 的机制

**含义**：ep_len 40 上限大概率是「纯风格奖励无 survival/跟踪梯度」的结构上限（R005 §6.1 的预感获机制支撑：Sds loss 只度量风格距离，pose/key 稠密跟踪梯度从未参与）。**修复 task 通道是当前最大的单点杠杆**——比 ipo/预算/任何奖励权重调参都大（那些都是在纯 SMP 目标函数内部调，天花板不动）。

---

## 论文与实现证据

**A. 根因三件套（已观测，源码直读）**

| 环节 | 文件:行 | 内容 |
|---|---|---|
| 调用点 | `mimickit/envs/sim_env.py` L171-178 | `_post_physics_step`：`_update_time → _update_misc → _update_observations → _update_info → **_update_reward()** → _update_done` |
| **空实现** | `mimickit/envs/amp_env.py` L280-281 | `def _update_reward(self):` **`return`**（无 super 调用、无写入） |
| 继承链 | `smp_env.py` L7（`class SMPEnv(amp_env.AMPEnv)`）+ `amp_env.py` L9（`class AMPEnv(deepmimic_env.DeepMimicEnv)`） | MRO：SMP → **AMP（空壳生效）** → DeepMimic（kernel 被跳过） |
| 被旁路的 kernel | `deepmimic_env.py` L408-460 | `_update_reward`：`_reward_buf[:] = compute_reward(...)`（engine 状态 + `self._ref_*` 输入） |
| 记录链 | `base_agent.py` L265-269/L323-325 | step 返回 `_reward_buf` → `_record_data_post_step` → `record("reward", r)` |
| 消费端 | `smp_agent.py`（`e472930`）L163/L174-175 | `task_r = get_data_flat("reward")`（≡0）→ `r = 0.5·task_r + 1.0·smp_r` |

**B. 畸形输入矩阵（本报告实验，`.venv/bin/python`，脚本+json 归档 `research/studio/idear/20261005_p0_malformed_matrix.*`）**

| case | tar 输入 | task_r |
|---|---|---|
| D | 全零（含单位四元数） | **0.139** |
| E | home/init_pose 冻结 | 0.313 |
| F | 真实 ref 但 joint_rot 归零 | 0.296 |
| G | 真实 ref 但速度全零 | 0.522 |
| H | sanity（相邻帧） | 0.922（复现 worker probe B） |

结论：`compute_reward` 是「挤压到 [0.22·w_sum, 1] 的加权和」（最差 case 各分项≈0.2-0.3 加权后 0.14 下限），**结构上给不出 <0.001**——死因不在输入。

**C. AMP 语义的合理性说明（推断）**：AMPEnv 空实现是 AMP 范式的正确设计（AMP 的 reward = agent 侧 disc+task 混合，env 侧无需 kernel）。SMP 复用 AMPEnv 时，`SMPAgent._compute_rewards` 假设 buffer "reward" 含 env task 值——**该假设对 AMPEnv 子类不成立**（AMP 官方 agent 大概率在 `_build_train_data` 自行计算 task 分量）。x1 的「单 clip + task_reward_weight 0.5」配置撞上了这个语义沟。

**D. r17 其余数据（已观测，R005 引用）**：E4′ 与 r14 无实质差异（v 0.54-0.71、duty_L 0.64-1.0、全跌）；r/r_floor=0.60（策略贴近流形地板区）；first-slot 复验通过；r16 事故不影响 r17 有效性。

---

## 优先建议

### I51 · task 通道修复 diff（两行，r18 启动前必落）
- **关联结果版本**：R005 §2.7/§6.1（死因待定位 → r18 计划）
- **文件/函数**：`mimickit/envs/smp_env.py`（推荐落点——不动 AMP 语义，只救 SMP）
- **观测事实/机制**：见§A。
- **最小改动或诊断实验**（worker 落地）：
  ```diff
  --- a/mimickit/envs/smp_env.py
  +++ b/mimickit/envs/smp_env.py
  @@ class SMPEnv(amp_env.AMPEnv):
  +    def _update_reward(self):
  +        # P0 fix (idear-0015): AMPEnv's _update_reward is an empty stub
  +        # (AMP computes reward agent-side); SMP needs the DeepMimic task
  +        # kernel to fill _reward_buf. Skip the stub via explicit MRO hop.
  +        deepmimic_env.DeepMimicEnv._update_reward(self)
  +        return
  ```
  （需在 smp_env.py 头部补 `import envs.deepmimic_env as deepmimic_env`——若未有。备选：`super(amp_env.AMPEnv, self)._update_reward()` 同义。）
- **对照变量**：修复前后的 Task_Reward_Mean 真值（I48 修复 @2664618 后日志已可信）。
- **预期指标**：修复后 Task_Reward_Mean ∈ [0.5, 0.95]（kernel probe 稳态域：相邻帧 0.92、1s 漂移 0.63-0.90）；若仍 <0.01 → 修复未生效（还有第二断点，按 I52 表排查）。
- **支持/否定条件**：支持=真值进入 kernel 域；否定=仍 0（转 I52）。
- **成本风险**：低；注意 reward 尺度突变（r 从 ~[0,1] 变 ~[0,1.5]）→ critic 重适应期前 ~100 iter 波动为预期，不作判据（预注册护栏）。
- **worker 的下一步**：与 r18 的 task_r 样本打印合并提交；冒烟 20 iter 确认 Task_Reward_Mean 非零即启动。

### I52 · 二级断点排查表（若 I51 后仍为 0；按概率排序）
- **关联结果版本**：R005 §2.7 疑点列表的收敛版
- **内容**（每项一行打印即可定位）：
  1. `_update_misc` 调用序：在 deepmimic `_update_misc`（L224-229，更新 `_ref_*`）入口加 counter——确认 ref 每步更新（AMP/SMP 未覆盖 `_update_misc`，预期正常）；
  2. `_reward_buf` 初始化与 dtype：reset 后打印 `_reward_buf[:3]`（预期修复前 0/修复后非零）；
  3. record 链：`_record_data_post_step`（base_agent L323-325）打印首个样本 r 值（区分 env 层 vs buffer 层）。
- **成本风险**：零（打印）。

### I53 · r18 预注册（修复生效后的判读，避免归因再混淆）
- **关联结果版本**：I51；0013-I45 三态框架的继承
- **内容**：
  - **主判据**：EVAL ep_len。预注册三档：**ep_len ≥90**（task 通道成为 ep_len 主杠杆的强证据；r4 的 95 得到机制解释=当时 ipo=100 无关、461M 预算无关——需回头复核 r4 是否有其他差异，见 I54）；**60-90**（部分贡献，混合因素）；**仍 ~40**（task 不是瓶颈 → 上限在探索/终止塑形/SMP 奖励形状，转 E8 与奖励重设计）。
  - **护栏**：前 100 iter 的 return/ep_len 波动（critic 重适应）不判读；Task_Reward_Mean 真值必须先进入 [0.5,0.95] 才开始 ep_len 计时。
  - **次级观测**：Sds_Loss 是否随跟踪改善而降（策略状态更贴数据流形的双向收益预期）；Smp_Reward 曲线在新混合奖励下的形态。
  - **表型**：r18 权重 → E4′（ duty 对称性/v 带）——task 的 pose/key 项直接约束这些，修复后应见改善。
- **成本风险**：零（预注册）。
- **worker 的下一步**：R006 按此口径出表。

### I54 · 历史图景重写结算（诚实修订 idear 历史报告的受影响结论）
- **关联结果版本**：本报告根因对 0006/0009/0012/0013 的追溯效力
- **内容**（逐条结算）：
  | 报告/结论 | 原表述 | 修订 |
  |---|---|---|
  | 0006-E9 ③ | 「task_reward 0.5 提供独立优势信号（r4 解冻三变量之一）」 | **失效**：task 从未生效；解冻归因=scale 灵敏度+Adam 两变量 |
  | 0009 | 「task 分量驱动跟踪（r6 ep_len 67.2 佐证）」 | **失效**：纯 SMP 下 ep_len 67；SMP 奖励的引导力比当时评估的更强 |
  | 0012-I41 | ipo 语义归因 | **已证伪**（R005 态 C 正式结算） |
  | 0013 拖行核算 | 「task 侧速度弱惩罚使拖行廉价（21% 损失）」 | **前提失效但结论加强**：task 惩罚不是弱而是**无**——拖行在纯 SMP 下零代价（速度盲区），是必然局部最优 |
  | 0011-I40 | 先验速度盲区 | **不变且加强**：这是拖行表型的完整解释（SMP 不看速度+task 不在场） |
- **成本风险**：零（记录性）。
- **worker 的下一步**：R006 引用历史结论时按此表口径。

---

## 最小验证实验

| ID | 内容 | 依赖 | 预算 | 判据 |
|---|---|---|---|---|
| E18（新） | I51 修复 + 20 iter 冒烟（Task_Reward_Mean 非零确认） | 账号 28 | 分钟级 | 真值 ∈ [0.5,0.95] |
| r18（主实验） | I51 + I48 仪器 + 300M | E18 | ~3.5h | I53 三档 |
| E4″（r18 后） | 表型复扫（duty/v/attack） | r18 权重 | 数百 rollout | 对称性/v 带 |
| I52（条件） | 二级断点表 | E18 失败时 | 打印 | — |

---

## 上下游缺口

1. **账号 28 余量未知**（r17 后）——r18 是本轮最高价值实验，若余额不足需优先保它（其他都让路）。
2. r4 的 95 ep_len 现在成为未解释离群（461M+ipo100+**task 同样死亡**）——若 r18 达 ≥90，需回头 diff r4 与 r14/r17 的其他差异（分支 26c8326 vs ab36e74 vs f7456dc 的 env/args 全量对比）——列入 R006 待办。
3. AMP 官方 agent 的 task 分量路径未核实（`amp_agent.py` 未拉取）——对根因无影响（x1 用 SMPAgent），但若 worker 想知道「SMP 论文任务配置为何能工作」，值得一读（可能是 task env 覆盖了 _update_reward，或官方 SMP 任务确实也空——后者会是上游 bug）。
4. 渲染 Isaac 侧 `_build_lights` 旧问题仍在（R005 §3）。

---

## 反证与不确定性

- **[已观测-代码级] 根因三件套 + 畸形矩阵 + 记录链**：源码直读与本地实验，可复核（脚本归档）。MRO 解析是 Python 语义的确定性推论（SMPEnv 未覆盖 `_update_reward`，AMPEnv 覆盖为空）——**唯一残余不确定性**：worker 分支（e472930/f7456dc）的 smp_env.py 是否与主干归档版一致（我是从主干 88dbd89e 归档的 smp_env.py；amp_env 同）——r18 冒烟的 Task_Reward_Mean 真值是最终确认。
- **[推断] 修复后 ep_len 显著改善**：机制论证（稠密跟踪梯度+survival 塑形）+ 文献（DeepMimic 家族 RSI+跟踪是模仿必要组件）+ R005 §6.1 预感三方一致；但**未实验**——I53 三档就是证伪通道（仍 40 的可能性保留：探索不足/终止塑形问题）。
- **[推断] r4 的 95 含其他未识别因素**：四点（95/67/40/39.3）在 task 全死的条件下仍非单调——r4 与 r14/r17 至少还有一个未识别差异（缺口 2）。
- **[说明] 本报告的定位速度（R005 发布后 ~25 分钟）得益于**：worker 的三级裁决（view/kernel/probe）把假设空间压缩到「调用/写入层」，畸形矩阵排除输入族后，grep 调用链只剩一条路径——这是两人证据链协作的产物，非单方功劳。

---

## 来源

**共享产物（已 fetch 快照）**
- dm-results-R005-r17-stateC-taskchannel v1（`inputs/studio/dm-results-R005-r17-stateC-taskchannel/`，sha256:f711ce7d…）

**共享证据（commit `49c32b4`/`8017c83`）**
- `analysis/i48_view_test.py`（view 语义测试）、`analysis/p0_task_kernel_probe.{py,json}`（kernel 健康）
- `analysis/e4_r17_results.json`、`checkpoints/r17_final.pt`（经 R005 引用）

**源码（主干归档 `research/studio/idear/mimickit_ref/` + `.repos/mk_api/`）**
- `mimickit/envs/amp_env.py` L280-281（**空实现**）、L9（继承）
- `mimickit/envs/sim_env.py` L171-178（调用点，`.repos/mk_api/mimickit_envs_sim_env.py`）
- `mimickit/envs/deepmimic_env.py` L408-460（被旁路的 kernel）、L224-229（_update_misc/_ref_* 更新）
- `mimickit/learning/base_agent.py` L265-269/L323-325（记录链）
- `mimickit/learning/smp_agent.py`（e472930）L162-184（消费端）

**本报告实验**
- `research/studio/idear/20261005_p0_malformed_matrix.{py,json}`（畸形输入矩阵，`.venv` 本地）

**承接**：idear-0014（I48 恒等发现的三级裁决设计——本报告是其第四级）、0013-I45（态 C 正式结算于 R005）、dm-results-R001..R005

**缺失证据**：worker 分支 smp_env.py 与主干的一致性（r18 冒烟确认）；修复后 r18 曲线；r4 离群差异（缺口 2）。
