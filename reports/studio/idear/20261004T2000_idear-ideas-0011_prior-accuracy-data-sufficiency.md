# idear 报告 0011：prior 准确度与数据充分性裁定（用户指导 2026-10-04）——分口径答案、审计方法复核与零训练修复路径

- 报告 ID：idear-ideas-0011
- 日期：2026-10-04（19:53）
- 作者：idear（研究与建议角色）
- 关联结果版本：**用户实时指导（taskContract r2）：「注意 prior 的训练结果准确度，看看 prior 的训练数据是否足够」**；worker 未发布新 dm-results（R004 未出），但共享区新增三个 commit：`c9a46b2`（E6b 扩展 + per-frame channel 发现 + 自保真地板）、`e9e1d37`（I32 角点复测 + 数据攻角 + 渲染修复）、`27f4eb1`（prior accuracy & data-sufficiency audit）及其证据文件 `analysis/prior_audit.{py,json}`、`analysis/prior_sampling_ab.py`、`analysis/i32_corner_{check,stance}.json`、`analysis/data_attack_angle.json`
- 新增证据：①对 worker 审计的独立方法学复核（3 处修正，§证据 A）；②DDIM 误差放大系数表（§证据 B，脚本 `research/studio/idear/20261004_ddim_amp_table.py`）；③20 源文献调研（[R5]，归档）；④I32 闭环（0010 浮空伪影假设**证实**）
- 状态：建议（非验收）。本报告直接回答用户指导的两个问题

---

## 摘要与关联结果

**问题一：prior 的训练结果准确度如何？——必须分两个口径回答，混用会得出错误结论：**

| 口径 | 证据 | 裁定 |
|---|---|---|
| **使用时（RL 奖励打分，t∈{22,15,8}）** | 重构 MAE（归一化）0.164/0.121/0.079；ε̂ 误差经 DDIM x0 反演放大仅 **1.31×/1.13×/1.04×**（§B 表）；r4/r6 双 PASS（奖励 120× 抬升、ep_len 67.2）实证可用 | **合格**——RL 线与采样损坏相互绝缘 |
| **生成时（全链采样，t 从 49 起）** | gen std 为数据的 100-300×；self-loop SDS 207 vs 数据 0.63；return test 失败（commit 27f4eb1）；ε̂ 误差在 t=49 被放大 **32.1×**（§B）→ 级联发散 | **损坏**——GSI 及任何生成用途不可用 |

**问题二：prior 的训练数据是否足够？——同样分用途：**

| 用途 | 数据现状 | 裁定 |
|---|---|---|
| **风格正则（当前 SMP-RL 用法）** | 92.8s / 2784 唯一帧 / 4 subjects / 单步态 / 速度带 0.97-1.52 m/s；680 窗口 stride-4（data-to-data NN 0.04，窗口空间 ~2.4× 冗余）；策略 raw loss 0.20 已落在数据分布内部（数据 mean 0.34 / p50 0.125 之间） | **当前容量下足够**——策略已达先验自身的数据保真地板，加同质数据不抬升平台（平台=地板，不是覆盖缺口） |
| **生成模型（GSI 初始化 / 泛化 / 复用）** | 同上 vs 文献标尺：人形全身 DP 需 ≥2M transitions（≈11h，且必须多样化，Kaidanov 2411.01349）；文本到运动 70h 起出现可测退化（Kimodo 2603.15546）；SMP 全语料 20h/100 风格（[R4]，折合 ~12min/风格为推导值） | **严重不足（1-2 个数量级）且采样器先坏**——生成路线需先修采样再谈数据 |

**一句话回答用户**：prior 在它现在承担的角色（SDS 风格奖励核）上是准的、数据是够的——r4/r6 解冻与 0.20 平台的归因（0010 机理①「ESM 地板」）已获 worker 自保真测试（p50 0.124）直接证实；它坏掉的是从未被 RL 用到的生成半边（高 t 采样），而那半边要修好，92.8s 数据差文献门槛 1-2 个数量级——**当前瓶颈既不是奖励质量也不是数据量，而是 checkpoint 取回（算力）与 sim2sim 验证（R004）**。

另：I32 角点复测证实 0010 浮空伪影预测（触地帧角点 zmin p50 **1.01mm**，全帧均值 1.22mm；「浮空」是中心口径伪影，R003 §1.1 撤回正确）。

---

## 论文与实现证据

**A. worker 审计的独立复核（prior_audit.py/json + prior_sampling_ab.py，commit 27f4eb1/c9a46b2）**

已观测数字（prior_audit.json）：
- 重构（加噪→单步 DDIM x0_hat，归一化 MAE）：t=22: 0.164（最差块 ang3 0.799、vel3 0.420；最好 jrot174 0.133）；t=15: 0.121（ang3 0.531）；t=8: 0.079（ang3 0.305）。
- 生成：256 样本各块 std 99.9-334 vs 数据 0.40-18.4（100-465×）；self-loop ESM_SDS：gen 206.9 vs data 0.63（前 64 窗）；NN：gen→data 8100.97 vs data→data 0.040（ratio 202727.8）。
- 容量算术：200k iter × batch 512 = 1.02e8 窗抽取 ÷ 680 窗 = 15.06 万遍/窗。

**我的三处方法学修正（本报告新增）：**
1. **gen_stats 块索引错位**：`prior_audit.py` L148 对扁平 (B,2010) 用 `gen[:, a*H : b*H]` 切块，但 frame-major 布局下块 c 的索引是 {f·201+c}（跨帧非连续），该切片实为「第 0 帧的前 30/… 维」——**per-block 归因不可靠**。但发散结论不受影响（self-loop 与 NN 为全向量口径，有效）。
2. **NN-ratio 判词无效**：json 的 `memorization_verdict: "SMOOTHED/GENERALIZED"` 在采样发散时无意义（样本是发散噪声而非「泛化」）；正确记忆化探针是**免采样的 loss-based MIA**（Carlini：t 处 loss 阈值，Goldilocks 区）或 **per-σ 泛化 gap G(σ)**（[R5] §5）。
3. **json verdict 与 commit 结论矛盾需拆分**：json 规则给出「DATA LIKELY INSUFFICIENT」，commit 27f4eb1 给出「data memorized, not data-limited」——两者都对但口径不同：对**生成**用途不充分（多样性/绝对量），对**当前容量+正则用途**是记忆化（不再受限于此数据）。摘要表格是统一裁定。

**B. DDIM x0 误差放大系数（本报告计算，`research/studio/idear/20261004_ddim_amp_table.py`，squaredcos_cap_v2 / T=50 / s=0.008）**

| t | ᾱ | 1/√ᾱ（ε̂ 误差→x0 误差放大） |
|---|---|---|
| 8 / 15 / 22（**RL 打分用**） | 0.933 / 0.787 / 0.587 | **1.04× / 1.13× / 1.31×** |
| 30 / 35 / 40 | 0.341 / 0.203 / 0.094 | 1.7× / 2.2× / 3.3× |
| 45 / 49（**采样起点**） | 0.024 / 0.0010 | 6.4× / **32.1×** |

机理链（[R5] §3 五因互证）：cosine 调度的 ᾱ(T)≈0.001 非零 → 训练时高 t 输入残留 ~3% 信号（模型学会读它）；推理时 x_49=纯噪声 → 首步 ε̂ 的 O(0.1-1) 误差被 32× 放大成 x0_pred 巨值 → 后续步复合发散（Kaiser & Kollmann：前向加噪分布 ≠ 反向轨迹分布，复合误差机制）。而 RL 的 ESM 打分在 t≤22 放大 ≤1.3×，且输入是「真实状态+新鲜噪声」（正是训练分布）——**这就是「打分正常、采样损坏」并存的定量解释**。

**C. 文献标尺（[R5] `research/studio/idear/20261004_small-diffusion-data-memorization-sampling.md`，20 源）**
- 记忆化阈值：56M EDM 在 1k 样本 >90% 复制；维度↑阈值↓（EMM 8k@192 维 → 1k@3072 维；我们窗口 2010 维 → 推导 EMM 远低于 1k）——**680 窗/2784 帧落在必然记忆化区**，与 l1+EMA 的平滑化（重构 err 0.08@t8 非零）自洽。
- 数据量标尺：Kaidanov（人形全身 DP）：500k transitions（≈2.8h）8 种随机化全失败、2M（≈11h）+DR 才成、8M 几乎全成，且「数据量不能补偿多样性缺失」；Kimodo：70h 出现可测退化；SMP 20h/100 风格（[R4]）。
- 采样修复文献：zero-terminal-SNR（Lin 2305.08891：rescale ᾱ 使 √ᾱ_T=0、v-prediction、含 t=T 的网格）；Min-SNR-5 加权（Hang 2303.09556：uniform ε-loss 欠训练高 t，3.4× 收敛）；**截断扩散（Zheng 2202.09671）：从 t'<T 起步反向采样是已发表的原理性方案**。

**D. I32 闭环与数据攻角（worker `e9e1d37`）**
- i32_corner_stance.json：触地帧（n=1743）角点 mean 1.22mm / p50 1.01mm → **浮空证伪**（0010 预测成立；全帧 corner_mean 42.8mm 含腾空相，属正常）。
- data_attack_angle.json：7 段 p50 2.5-7.8°（总体 ~6°）——低于姿态标准 10-25° 带，慢跑平放脚特征（数据特性记录，非缺陷）；与旧线策略 sim2sim 65-76° 病态形成三方对照的第三点。

---

## 优先建议

### I36 · 采样损坏的零训练判定与修复路径（若要走 GSI/生成路线）
- **关联结果版本**：commit 27f4eb1（采样 broken）；本报告 §B
- **文件/函数**：`.repos/mimickit_shim/mimickit/learning/tinymdm/tinymdm_model.py` L133-143（`_sample_with_scheduler` 从纯噪声起步）；`analysis/prior_sampling_ab.py`（现成 A/B 框架）
- **观测事实**：return test 失败；放大系数 t=49 32× vs t≤22 ≤1.3×。
- **机制假设**：高 t 段 ε̂ 不完美 + 32× 放大级联（§B 机理链）；**截断采样可救**：从 t′≈28-32 起步（放大 ≤1.7-2.2×），数学上等价于截断扩散先验 p_trunc（数据邻域合法密度，Zheng 2202.09671 支持）。
- **竞争解释**：若 t′=30 起步仍发散 → 中 t 段复合误差主导（Kaiser 机制 4）→ 需重训侧修复（Min-SNR 加权或 v-prediction，Lin/Hang 路线）。
- **最小改动或诊断实验**（全部本地 CPU 分钟级，扩展 prior_sampling_ab）：
  1. ε̂-err vs t 曲线（t=8..49 步长 3，各 64 窗）：定位误差拐点 t*；
  2. 部分回程：从 t∈{45,40,35,30,25} 加噪真实窗起跑 DDIM 到 0，测终点 L2-to-real——成功最大 t 即安全截断点 t′；
  3. 若需 GSI：`sample_ema` 的 timesteps 改为 `set_timesteps(50)` 后从 t′ 截断（diffusers 支持传子序列），生成后 unnormalize——先做 2 再动 3。
- **对照变量**：起步 t′；EMA vs raw 权重。
- **预期指标**：部分回程在 t′≤30 成功（L2-to-real < 数据 NN 尺度的 3×）。
- **支持/否定条件**：支持=t′ 存在且 GSI 样本统计合理（per-block std 比 ≤3×数据）；否定=t′=22 以下才行 → 截断先验太窄，转重训修复。
- **成本风险**：低（诊断零训练）；GSI 启用仍需 r10 消融验证。
- **worker 的下一步**：阻塞期可做（I35 清单插入为第 0.5 项）；结果并入 R004。

### I37 · 数据扩充目标（按用途分级，先修采样再买数据）
- **关联结果版本**：用户指导；[R5] §4 标尺
- **文件/函数**：`data/datasets/`（新语料 yaml）；`x1_gmr_retargeted_tool/`（重定向管线）
- **观测事实**：现状 92.8s 单步态；生成路线文献门槛 11-70h。
- **机制假设**：正则用途加同质数据边际收益≈0（平台=地板）；生成/泛化用途需多样性（速度带/受试者/步态家族）优先于绝对量（Kaidanov「多样性>量」）。
- **竞争解释**：若一期只求 sim2sim 通过现有慢跑策略，数据可不扩——把算力留给训练。
- **最小改动或诊断实验**：分级目标——L1（维持现状）：仅当 R004 显示策略步态质量不足才考虑；L2（单风格强化）：≥10min 多受试者慢跑（速度带 0.9-2.0，≥8 subjects，含变速/转身），对应 SMP 单风格折合量级；L3（多风格复用）：1-2h 走/慢跑/跑/冲刺家族（对齐 R001 sprint 重整计划）。
- **对照变量**：数据量 × 多样性正交消融（L3 阶段才值得做）。
- **预期指标**：L2 后先验对同风格未见窗口的重构误差下降（用 2 留出段做 G(σ) 曲线，[R5] §5 探针）。
- **支持/否定条件**：支持扩充=留出段 gap 大且策略风格指标差；否定=两者都小则维持。
- **成本风险**：L2 需重定向管线跑新段（服务器端 G1 csv 依赖，R001 已知）；L3 大。
- **worker 的下一步**：R004 之后再决策；本条先立靶。

### I38 · 审计方法修正（把本次审计的三处口径问题固化）
- **关联结果版本**：本报告 §A 修正 1-3
- **文件/函数**：`analysis/prior_audit.py`（若复用）
- **观测事实/机制假设**：见 §A——块索引、NN 判词、verdict 拆分。
- **最小改动或诊断实验**：复用时：①块统计改为 `gen.reshape(B,H,201)[:,:,a:b]`；②记忆化判定换 loss-MIA（训练窗 vs 2 个留出段在 t=15 的 loss 分布分离度）或 G(σ)；③verdict 字段拆 `regularizer_sufficient` / `generative_sufficient` 两键。
- **对照变量**：—。
- **预期指标**：审计输出可直接支撑数据决策。
- **支持/否定条件**：—。
- **成本风险**：零。
- **worker 的下一步**：下次审计脚本更新时并入（不紧急）。

### I39 · r9 预注册三条（算力恢复即生效，避免临场变量混入）
- **关联结果版本**：0010-I34/I20；本报告裁定
- **内容**：① **GSI 保持 False**（采样损坏，r9 冻结配方已是 False——维持；启用 GSI 前必须先过 I36-2 部分回程测试）；② 日志增补三件套合入冻结分支：`task_reward_mean`（0006-I20）、`sds_norm_mu`（0010-I34）、`sds_loss_mean/std` 已有——r9 曲线以此三量判读，Smp_Reward 绝对值仅作参考；③ 主仪表改 **r/r_floor**：r_floor = exp(−2×L_floor/μ)，L_floor 取数据自评 p50 0.125（c9a46b2）——r9 停滞判据从「Smp_Reward>0.5」改为「raw L 是否仍降 + r/r_floor 是否 >0.9」。
- **支持/否定条件**：r9 若 raw L 平台且 r/r_floor>0.9 而 ep_len/task 仍升 → 确认 0010 机理（饱和无害），策略质量交给 R004 九类指标；若 raw L 仍在降 → 继续训练即可，无需干预。
- **成本风险**：零（观测与预注册）。
- **worker 的下一步**：账号恢复后 r9 启动前 10 分钟并入。

### I40 · 奖励盲区备忘（R004 解读用）：先验对速度/角速度块近乎失明
- **关联结果版本**：prior_audit 重构分块表（ang3 0.31-0.80、vel3 0.18-0.42 vs jrot174 0.06-0.13）
- **观测事实/含义**：SMP 奖励几乎看不见 root 角速度与速度细节 → **策略躯干姿态/攻角病态不能归咎于 SMP 奖励失职**（它本来就看不清）；该维度由 task 分量（DeepMimic root 项）负责。R004 若见攻角/俯仰病态，优先查 task 侧 root 权重与终止条件，而非调 SMP。
- **worker 的下一步**：R004 判读时对照本条。

---

## 最小验证实验

| ID | 内容 | 依赖 | 预算 | 判据 |
|---|---|---|---|---|
| E12（新） | ε̂-err vs t 曲线 + 部分回程（I36-1/2） | prior_sampling_ab 框架 | CPU 分钟级 | 安全截断点 t′ 是否 ≥30 |
| E13（新） | （条件）截断采样 GSI 样本质量检查（I36-3） | E12 | CPU 分钟级 | per-block std 比 ≤3× |
| r9 前置 | I39 三条合入冻结分支 | — | 10 分钟 | — |
| R004 | （worker）r9 ckpt sim2sim + 攻角三方对照（数据 6°/旧线 65-76°/新线） | 算力恢复 | — | 九类指标 |

---

## 上下游缺口

1. **算力与 checkpoint 取回仍是第一阻塞**（R003 §3 未变）——prior 侧结论已闭环，项目推进钥匙在 gradmotion 账号。
2. GSI 路线依赖 I36 判定（截断采样可行性）——阻塞期可完成。
3. 留出段缺失：7 段全用于训练，G(σ)/loss-MIA 探针需要 1-2 段留出（L2 扩数据时顺手做）。
4. `tinymdm_model.py` 的 `Normalizer`（z-score/通道）与奖励侧 `DiffNormalizer`（x/mean|x|）同名不同物——文档与日志命名建议区分（避免下个读者混淆，0010 μ 反解已踩过一次）。

---

## 反证与不确定性

- **[已观测] 审计数字/代码路径/放大系数表**：json/源码直读 + 纯数学计算，可复核。
- **[推断] 「策略 raw 0.20 在数据分布内部」**：e6b base=全 680 窗 mean 0.34，audit self-loop=首 64 窗 0.63（clip 间差异大），c9a46b2 p50=0.125——策略 0.20 介于 p50 与 mean，方向稳健；精确分位数未逐窗复算。
- **[推断] 截断采样可救 GSI**：机理（放大系数跳变 32×→1.7×）+ 文献（Zheng）支持，但**未实测**——E12/E13 是证伪通道；若 Kaiser 式中 t 复合误差主导则失效。
- **[推断] L2 数据目标（≥10min）**：SMP 20h/100 风格折合的推导值 + Kaidanov 多样性优先原则；无单风格小先验的直接文献数（[R5] 明确该空白）——目标量级参考而非硬门槛。
- **[结算-0010] 两个假设裁决**：浮空=伪影（**证实**，stance 角点 1.0mm）；奖励平台机理①ESM 地板（**获直接证据** self-fidelity p50 0.124；②棘轮 μ≈0.39 与③速度不敏感（noVel +2.4% + 本审计 vel/ang 块重构最差）同获支持）——0010 三重机理全部成立。
- **[提醒] prior_sampling_ab 的 stdout 未存档**：return test 具体数值（commit 引述「×100 放大」）我只见到放大系数推算（t=49 为 32×；×100 或为多步复合后量级）——E12 重跑时落盘 json。

---

## 来源

**用户指导**：taskContract r2「注意 prior 的训练结果准确度，看看 prior 的训练数据是否足够」（2026-10-04 实时补充）。

**共享证据（commit `c9a46b2`/`e9e1d37`/`27f4eb1`）**
- `analysis/prior_audit.py` + `prior_audit.json`（重构/生成/NN/容量算术）
- `analysis/prior_sampling_ab.py`（ema/raw × ddim/ddpm × return test 框架）
- `analysis/i32_corner_check.json`（全帧）、`analysis/i32_corner_stance.json`（触地帧，浮空证伪）
- `analysis/data_attack_angle.json`（7 段攻角基线）
- `.repos/mimickit_shim/mimickit/learning/tinymdm/tinymdm_model.py` L14-113（架构/normalizer）、L133-143（采样起点纯噪声）、L232-272（ESM_SDS_loss 的 x0 反演路径）

**本报告计算/调研**
- `research/studio/idear/20261004_ddim_amp_table.py`（放大系数表）
- [R5] `research/studio/idear/20261004_small-diffusion-data-memorization-sampling.md`（20 源：Gu 2310.02664、Favero 2505.16959、Lin 2305.08891、Hang 2303.09556、Kaiser&Kollmann 2603.13419、Zheng 2202.09671、Kaidanov 2411.01349、Kimodo 2603.15546、Carlini 2301.13188 等）

**承接**：idear-0010（浮空伪影/三重机理，本报告裁决）、0009-I29（R004 判读顺序）、0006-I20（日志增补）、dm-results-R001/R002/R003

**缺失证据**：prior_sampling_ab stdout 数值（E12 落盘）；留出段 G(σ)；SMP 单风格小时数的直接文献值（[R5] 确认无发表）。
