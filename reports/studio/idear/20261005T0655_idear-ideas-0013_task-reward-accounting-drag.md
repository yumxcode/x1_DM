# idear 报告 0013：task 奖励核算——慢速拖行是结构局部最优（21% 损失）；ep_len 口径确认；r15/r16 预期修正

- 报告 ID：idear-ideas-0013
- 日期：2026-10-05（06:55，R004 后续独立研究，无新 dm-results）
- 作者：idear（研究与建议角色）
- 关联结果版本：dm-results-R004 v1（E4 表型：v 0.56-0.79、duty 单脚 0.73-0.92、ep_len 40）；本次新增证据为 x1_mimicKit **task 奖励侧**源码入档与定量核算（此前 12 份报告只覆盖 SMP/先验侧）
- 新增证据：①`data/envs/smp_x1_env.yaml`（r4 `26c8326` 与 r14 `ab36e74` 两分支**逐字一致**，含 pose_termination True/0.8m）与 `mimickit/envs/deepmimic_env.py` `compute_reward`（L787-846）入档；②「慢速拖行」代价核算（脚本归档 `research/studio/idear/20261005_task_reward_accounting.py`）
- 状态：建议（非验收）

---

## 摘要与关联结果

三项结论：

1. **ep_len 口径确认**：r4/r14 的 `pose_termination: True, dist 0.8m` 完全一致（逐字）——ep_len = 「跟踪偏差>0.8m 或跌倒」的统一口径，0012 的归因链（ipo 语义 vs 方差）不受口径差影响，E14 仍有效。同时给出 ep_len 的**直接机制**：ep_len 是前进速度的间接函数（慢速 → 偏差累积 → 0.8m 终止）。
2. **慢速拖行的代价核算（核心）**：把 E4 实测表型（0.6 m/s 跟踪 1.2 m/s 参考、关节误差 0.15 rad、脚位置基本跟上、root 落后 0.3m）代入 x1 task 奖励公式：**task_r=0.79 vs 完美 1.0——拖行只损失 21%**。分解：速度相关两项（root_vel_w=0.1 + vel_w=0.1）只占损失的 **24%**；pose 项因核宽松（`pose_scale=0.25`，而 DeepMimic 原版公式 (2/15)×nj 对 x1 应为 **3.87**）在 0.15 rad 平均误差下仍给 0.89（原版公式给 0.24）。**拖行是当前奖励结构下的理性局部最优**——只要正确速度步态的探索风险（跌倒早终）超过 21%，策略就会停在拖行。
3. **对 I42 预期的修正**：拖行表型从「训练不充分伴生（r15 自愈）」**降权**为「奖励结构局部最优（r15 大概率不自愈速度表型）」。可检验预测已预注册（I45）：r15 若 ep_len 回 90+ 而速度仍 0.6-0.8 m/s → 判定成立 → r16 奖励重平衡消融（I46，两行 diff）。

---

## 论文与实现证据

**A. ep_len 口径与终止机制（已观测，两分支 env yaml 逐字比对）**

`data/envs/smp_x1_env.yaml`（r4/r14 同）：`pose_termination: True`、`pose_termination_dist: 0.8m`（注释：接触力跌倒检测失明——传感器只报 base_link——故用跟踪偏差终止）、`enable_early_termination: True`、`episode_length: 10.0s`、`rand_reset: True`。
机制：ep_len 40 步 = 1.33s @30Hz = 策略在 1.33s 内累计跟踪偏差（key bodies 口径）达 0.8m 或跌倒。**慢速是 ep_len 短的充分原因**：0.6 vs 1.2 m/s 的速度差在 1.33s 内拉开 0.8m——数量级精确吻合，无需假设额外失稳。

**B. task 奖励表（已观测，smp_x1_env.yaml + deepmimic_env.py compute_reward L787-846）**

| 分量 | 公式 | 权重 | 尺度 |
|---|---|---|---|
| pose | exp(−s·ΣwᵢΔθᵢ²)，wᵢ 腰1.0/臂0.25-0.5/腿1.0（有效和≈21） | 0.5 | **0.25** |
| vel | exp(−0.01·ΣwᵢΔωᵢ²) | 0.1 | 0.01 |
| root_pose | exp(−5·(Δp²+0.1Δθ²)) | 0.15 | 5.0 |
| root_vel | exp(−1·(Δv²+0.1Δω²)) | 0.1 | 1.0 |
| key_pos | exp(−10·ΣΔp_key²)（4 body：双脚双手，root 相对） | 0.15 | 10.0 |

对照 DeepMimic 原版（[R2] §1，代码核实）：pose_w 0.5 / vel 0.05 / root 0.2 / end_eff 0.15 / com 0.1；**pose_scale=(2/15)·num_joints**（x1 29dof → 3.87）。x1 的 pose_scale=0.25 偏离原版公式 **15.5×**；vel_w 0.05→0.1、root_pose_w 0.2→0.15 为小调整。

**C. 拖行代价核算（本报告计算，脚本 `research/studio/idear/20261005_task_reward_accounting.py`）**

| 场景 | task_r | 分量贡献（pose/vel/root_pose/root_vel/key） |
|---|---|---|
| 完美跟踪 | 1.000 | 0.50/0.10/0.15/0.10/0.15 |
| **慢速拖行（E4 表型）** | **0.791** | 0.444/0.081/0.096/0.070/0.101 |
| 全身错误但速度对 | 0.665 | 0.312/0.081/0.143/0.099/0.030 |

关键数字：拖行总损失 21%；速度两项占损失 24%；0.15 rad 平均关节误差下 x1 pose_r=0.889 vs 原版公式 0.235。

**D. 与 SMP 侧证据的闭合**：0011-I40（先验速度块盲区：vel/ang 重构误差最差）+ 本报告（task 侧速度惩罚权重低）——**速度维度在混合奖励的两个分量里都弱**：r = 0.5·task + 1.0·smp 中，速度相关信号 ≈ 0.5×0.2×损失 ≈ 2% 总奖励量级。E4 的 0.56-0.79 m/s（数据带下限的 58-82%）与该结构性弱惩罚自洽。

---

## 优先建议

### I45 · r15 表型的预注册判读（E14 出结果后立即适用）
- **关联结果版本**：0012-I41（r15 配方）；R004 §3（E4 表型）
- **内容**：r15 的 E4' 出现以下三态之一时的判定：
  - **态 A**（ep_len ≥90 且 v ≥0.9）：ipo 归因闭合 + 拖行非局部最优（0013 核算被推翻的分支）——r16 直接做 sim2sim 鲁棒性与验收冲刺；
  - **态 B**（ep_len ≥90 但 v 仍 0.6-0.8）：**ipo 归因闭合 + 拖行=奖励局部最优证实**（核算成立）——启动 I46 奖励重平衡；
  - **态 C**（ep_len 仍 40 级）：归因转 run 方差/配方上限——按 0012 转 E8 四格消融。
- **成本风险**：零（判读规则）。
- **worker 的下一步**：R005 按 A/B/C 判定分支报告。

### I46 · 奖励重平衡消融（若态 B）：两行 diff、单变量、先小预算
- **关联结果版本**：本报告 §B/C；[R2] DeepMimic 原版公式
- **文件/函数**：`data/envs/smp_x1_env.yaml`（两行）
- **机制假设**：速度惩罚太弱（root_vel_w 0.1 + vel_w 0.1，损失占比 24%）+ pose 核太宽（0.25 vs 原版 3.87）使拖行成为廉价局部最优；恢复速度惩罚强度应把策略推离拖行。
- **竞争解释**：速度惩罚过强可能诱发「跌倒型冲刺」高风险行为；且 pose 核放宽可能是 g1/humanoid 配置的有意继承（SMP 原项目的 humanoid pose_scale 值未查——**worker 一行 grep 可查**：`grep pose_scale data/envs/*.yaml` 若 g1 也 0.25 则为 MimicKit 传统而非 x1 失误，改动需更谨慎）。
- **最小改动或诊断实验**（消融顺序按侵入度）：
  ```diff
  # 方案 α（最小，只调速度权重）
  - reward_root_vel_w: 0.1
  + reward_root_vel_w: 0.3
  # 方案 β（恢复原版核量级，动 pose）
  - reward_pose_scale: 0.25
  + reward_pose_scale: 3.87   # (2/15)*29, DeepMimic formula
  ```
  r16a=α、r16b=β 各 100M 短预算（r4 前 1/4），看 fwd_vel（Isaac 侧 test 统计即可）与 ep_len 的分离方向。
- **对照变量**：α vs β vs r15 基线（三格）。
- **预期指标**：α 后 test fwd_vel → 1.0+；β 后关节误差收紧（dof 跟踪误差下降）但速度可能不变。
- **支持/否定条件**：支持=任一方案速度进入数据带；否定=两者都不动速度 → 速度弱惩罚不是主因，转 GSI/探索结构（E8）。
- **成本风险**：中（两轮短训练）；β 若 g1 也是 0.25 则风险升（改传统值），先跑 α。
- **worker 的下一步**：态 B 触发时执行；先 grep g1 配置定 β 的正当性。

### I47 · pose_scale 谱系查证（5 分钟，I46-β 前置）
- **关联结果版本**：本报告 §B；MimicKit 仓库内配置族
- **最小改动或诊断实验**：`grep -r "pose_scale" data/envs/`（worker 本地或远端仓库）——记录 g1/humanoid/smpl 各 env 的值。若 g1=0.25 → MimicKit 传统（可能为 23+ dof 角色的有意放宽，DeepMimic 公式在 dof 多时过严）；若 g1 用原版公式 → x1 配置是孤例，β 正当性升。
- **成本风险**：零。
- **worker 的下一步**：与 I35 清单合并执行。

---

## 最小验证实验

| ID | 内容 | 依赖 | 预算 | 判据 |
|---|---|---|---|---|
| E14（既有） | r15（ipo100+解耦+461M） | 账号 | ~6h | I45 态 A/B/C |
| E15（新） | I47 pose_scale 谱系 grep | — | 分钟 | g1 值 |
| E16（条件） | r16a/b 奖励重平衡两格 | E14 态 B + E15 | 2×100M | fwd_vel/ep_len 分离 |

---

## 上下游缺口

1. r15 尚未启动（共享区自 R004 00:33 后无新活动——账号 26 余量仍未知，0012 缺口 1 未解）。
2. r14 渲染 mp4 人形复核：video_inspect 服务本报告期间再次 503（连续第 3 次，与 worker 遭遇一致）——保留为待办。
3. SMP 原项目 humanoid 配置的 pose_scale 值（I47/E15 目标）。
4. E4 的 test 侧速度统计：Isaac EVAL 报告里若有 fwd_vel（非仅 ep_len），态 B 判定无需等 MuJoCo E4'——R005 建议包含。

---

## 反证与不确定性

- **[已观测] 奖励表/终止配置/口径一致/核算公式**：源码与两分支 yaml 直读；核算为封闭算术（脚本可复现）。
- **[推断-中高置信] 拖行=局部最优**：核算证明「拖行便宜」（必要条件），但「正确步态更贵」（跌倒风险>21%）是推断——若正确步态其实可稳定探索，r15 态 A 会推翻它（这正是 I45 三态设计）。
- **[推断] 场景参数代表性**：核算用 0.15 rad/0.1m/0.3m 等抽象误差值；E4 实测 dof_vel_absmax 4.6-10.9 与 duty 表型与其相容，但逐帧代入未做（需 r15 checkpoint 后的轨迹级核算——若态 B 触发，r16 决策前可加一个 10 分钟的轨迹级重算）。
- **[未验证] pose_scale=0.25 的设计意图**：E15 谱系查证前，不能断言它是失误；DeepMimic 公式对 29 dof 也未必最优（原公式为 23 dof SMPL 调的）。
- **[说明] ep_len 40 与速度差的定量吻合**（0.6 m/s 差 × 1.33s ≈ 0.8m）是量级论证，不是精确因果——key bodies 口径的偏差累积速率取决于具体 4 body 距离和，非 root 单点。

---

## 来源

**分支源码（raw 拉取，归档 `research/studio/idear/mimickit_ref/`）**
- `data/envs/smp_x1_env.yaml`（r14 `ab36e74` 全文 + r4 `26c8326` grep 比对：pose_termination/奖励表逐字一致）
- `mimickit/envs/deepmimic_env.py` L787-846（compute_reward 全文，本次归档 41.8KB）

**本报告计算**
- `research/studio/idear/20261005_task_reward_accounting.py`（拖行代价核算，输出见 §C）

**共享数据（commit `831f51c`）**
- `analysis/e4_r14_results.json`（表型输入：v 0.56-0.79、duty 0.73-0.92、ep_len 40 经 R004）

**文献**：[R2]（DeepMimic 原版权重/尺度公式，代码级核实）；[R4]/0011-I40（先验速度块盲区）

**承接**：idear-0012（I41/E14 r15 配方不变；I42 的候选排序被本报告修正）；dm-results-R004 v1

**缺失证据**：r15 曲线与 E4'；g1/humanoid 的 pose_scale 值；video 服务恢复。
