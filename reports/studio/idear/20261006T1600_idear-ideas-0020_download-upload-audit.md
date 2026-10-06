# idear 报告 0020：模型下载/上传链路端到端审计（用户指导 r3）——r23 下载前必读清单、I64 部署疑点、0018 相位排除撤回

- 报告 ID：idear-ideas-0020
- 日期：2026-10-06（16:00，r23 收尾/下载即将发生的窗口期）
- 作者：idear（研究与建议角色）
- 关联结果版本：**用户实时指导（taskContract r3）：「注意从头审核下看看训练出来的模型是否正确下载，是否正确上传，别出现因为 gradmotion 机制问题，导致训练好了，但是下载的是错的」**；dm-results-R008 v1（承接其 §3 I64 落地声明与 §4 first-slot 协议）；worker commit `a31284c`（r23 启动 + I68 实测摆幅 0.61m）；r23 运行中证据 `.repos/r23log4.json`（iter 2850-3076，~403M/461M=87%）与 `.repos/note-054.json`
- 新增证据：①df80895 分支保存语义与配置逐项核验（本报告§证据 A）；②**r23 运行日志中 I64 分桶列缺失**的发现（19 个指标列全枚举，无 fail_count/fail_low_root）；③`_obs_norm._count` 预算不敏感性的算术（§证据 C）；④相位算术修正（I68 实测摆幅 0.61m → **撤回 0018 的相位排除**，§证据 D）；⑤r23 中途读数（test_ret 末窗 23.6±0.0@~403M）
- 状态：建议（非验收）。**I70 是 r23 权重到手后 10 分钟内的操作清单；I71 是 2 分钟部署核查（应在 R009 前做）**

---

## 摘要与关联结果

用户要求从头审计「训练→上传→注册→下载→验证」全链路。审计覆盖两侧四个环节，**发现 2 个问题 + 1 个历史不变量弱点**：

**上传侧（代码/配置 → 平台）**：
- ✅ 分支 `df80895` 内容核验通过：保存链三要素齐全（skip-iter-0 / test-每100iter / **仅终局写盘**——base_agent L91-93 的 `if (self._sample_count >= max_samples): _output_train_model(...)`）；配置逐项符合冻结配方（Adam 3e-4、scale 2、task/smp 0.5/1.0、GSI False、ipo=100、normalizer_samples 1e8）。
- ⚠️ **[P1] I64 部署疑点**：分支上有 I64（fail_count/fail_low_root 进 `_diagnostics`），但 r23 运行日志的 19 个指标列中**两者均未出现**（全枚举：Iteration/Wall_Time/Samples/Test_Return/Train_Return/Critic_Loss/Actor_Loss/Clip_Frac/Imp_Ratio/Action_Bound_Loss/Adv_Mean/Adv_Std/Task_Reward_Mean/Smp_Reward_Mean/Smp_Reward_Std/Sds_Loss_Mean/Sds_Loss_Std/Sds_Norm_Mu/Exp_Prob）。两种解释：(a) 任务实际构建自早于 df80895 的提交（上传/构建侧不一致——**正是用户担心的机制类问题**）；(b) `_diagnostics` 路径的 logger 表化仍然不通（R008 的落点修正只对了 dict，表化链还有一环）。判别法见 I71（2 分钟）。
- 训练本身健康（I34/Sds_Norm_Mu 在列、Task/Smp 独立、test_ret 末窗 23.6@403M 与 0019 的 ~23-25 档先验一致）——权重价值不受 I71 影响，受影响的只是 I59 终裁工具。

**下载侧（平台 → 本地）**：
- ⚠️ **[审计核心发现] 历史验证不变量 `_obs_norm._count=100,270,080` 是预算不敏感的弱检查**：算术上 = 765 iter × 131,072 samples/iter 精确（normalizer 在 1e8 配置线冻结，冻结点 ≈ 100M samples ≈ 461M 预算的 22%）——**300M 与 461M 的 run 同值，任何 ≥765 iter 的中途/陈旧 checkpoint 都能通过**。r14/r17/r20 用它「验证」实际只证明了「配置未变 + 训练 ≥765 iter」，不是「最终权重」。
- ✅ 强检查存在且历史有效：model list 唯一条目 + createTime == 训练结束时刻（r14/r17/r20 三验）。
- **I70 给出 r23 的五步下载验证清单**（时间线 + 行为指纹 + 唯一性），其中行为指纹（下载后本地跑 test/E4 与远端末次 eval 对照）是唯一能端到端排除「下错文件」的检查。

**附带（诚实修正）**：worker I68 实测摆幅 **0.61m** > 我 0018 假设的 0.5m → 相位失配触发时间从「2.7s+（排除）」修正为 **0.84-1.10s（与观测 0.76s 同量级）——0018 的相位排除撤回**，相位失配复活为活跃候选（§证据 D）。这使 I64 分桶（若修复）的裁决价值上升。

---

## 论文与实现证据

**A. df80895 上传侧核验（已观测，raw 拉取）**

| 项 | 证据 | 判定 |
|---|---|---|
| skip-iter-0 | base_agent L74-79 注释 + `self._iter > 0` 条件 | ✅（r11 修复在位） |
| test/重置每 100 iter | `output_iter` 判据 + `test_model` + `_reset_envs`（L80-94） | ✅（r16/r17 语义在位） |
| 仅终局写盘 | `if (self._sample_count >= max_samples): self._output_train_model(...)`（L91-93） | ✅（r15 解耦在位） |
| 配置 | Adam 3e-4×2 / scale 2 / K=[22,15,8] / task 0.5 / smp 1.0 / GSI False / ipo 100 / normalizer 1e8 | ✅ 与冻结配方一致 |
| test_info 占位 | L69（r13 修复在位） | ✅ |

**B. r23 运行日志（已观测，r23log4.json，iter 2850-3076）**：末窗（last-50）Test_Return 23.60±0.00 / Train_Return 39.83±1.76 / Task 0.40 / Smp 0.43 / Sds_Norm_Mu 在列。**无 fail_count / fail_low_root**（grep 计数 0；列名全枚举见摘要）。note-054 成功判据 3（「I64 live - CRITICAL: verify within first 100 iters」）**未满足**——该判据设计正确，恰好在捕捉这类部署失败。

**C. `_obs_norm._count` 算术（本报告计算，脚本 `research/studio/idear/20261006_audit_arithmetic.py`）**：4096 envs × 32 steps = 131,072 samples/iter；观测值 100,270,080 = **765 × 131,072 精确**；naive 冻结点 ceil(1e8/131,072)=763（+2 iter 偏移是 update/check 时序细节，不影响结论）。**推论**：冻结发生在 ~100M samples（461M 的 22%），count 值与总预算无关——历史「三验」的该子项为弱检查。

**D. 相位算术修正（I68 实测消费）**：摆幅 A=0.61m → 触发 0.8m 需相位差 2·asin(0.8/1.22)=**1.43 rad（82°）**；步率漂移 1.3 rad/s → **~1.10s**，更快变体（2.4 vs 1.2Hz，1.7 rad/s）→ **~0.84s**——与 r20 观测 0.76s 同量级（0018 旧假设 A=0.5m 给 2.7s+ 的「排除」**作废**）。

**E. 历史下载事故链（承接 R003/R008）**：r4-r7/r10/r15/r16/r19/r21/r22 共 10 次权重丢失（first-slot 只在完成时落盘是余额击杀下的代价，非注册错误）；r14/r17/r20 三次成功取回的验证证据 = 唯一条目 + createTime==结束 + count（弱项，见 C）。

---

## 优先建议

### I70 · r23 下载验证五步清单（用户指导的直接落地；权重到手后 10 分钟内执行）
- **关联结果版本**：用户指导 r3；R008 §4（first-slot 协议）
- **文件/函数**：`gm.sh task models`（列表）、下载 URL、`checkpoints/r23_final.pt`
- **内容**（按序，全部零算力/分钟级）：
  1. **唯一性**：model list **仅 1 条**且文件名 = model.pt（若 2 条 → 有杂散 .pt，先查来源再下载，**勿盲取第一条**）；
  2. **时间线**：条目 createTime ∈ [训练结束时刻, +5min]（r23 预计 ~15:50-16:10 结束；对照 r23final 日志的末行时间戳）——**createTime 早于结束时刻 = 陈旧/中途文件，弃**；
  3. **写盘时序**：console 里 SDK「New file detected: .../model.pt」行的时间戳 **晚于** 末行 Samples≈461,373,440 的日志时间（证明写的是终局权重）；
  4. **文件级**：大小 ≈ 28.8MB（同架构同值）；若 SDK 列表含 digest，与下载后 sha256 比对；
  5. **行为指纹（端到端终裁）**：下载后任选其一—— Isaac test 模式跑 `test_episodes`（gm play），Test_Return 与远端末次 eval（≈23.6）差 <±20%；或 MuJoCo E4 base 格 3 eps，存活/duty 与 r20 同量级（~1s / 0.7-0.9）。**任何「数值合理但行为不符」都按下错文件处理**。
- **对照变量**：远端末次 eval 值（23.6）为基准。
- **预期指标**：五步全过 → 下载正确性高置信（历史事故模式全部覆盖）。
- **支持/否定条件**：步骤 2/3 任一失败 → 时间线错位，重查；步骤 5 失败 → 文件错误或环境错位，冻结并报告。
- **成本风险**：零（全部只读检查）；不执行的风险 = 用户点名的场景（训对了下错了）。
- **worker 的下一步**：**权重到手后先跑此清单再入 checkpoints/**；R009 附五步结果表。

### I71 · I64 部署疑点 2 分钟判别（R009 前做）
- **关联结果版本**：本报告 §B；note-054 判据 3
- **机制假设**：(a) 任务构建自早于 df80895 的提交（上传侧不一致）或 (b) `_diagnostics`→logger 表化链仍断（R008 修了一半）。
- **最小改动或诊断实验**：① 任务构建日志/console 头部找 commit 回显（git rev-parse 或 echo CI_COMMIT）——与 df80895 比对；② 若无回显：本地读 `_log_train_info`/logger 对 `env_diag_info` 键的处理——确认 `_diagnostics` 键是否真的进表（R008 的「logger auto-tabulates」断言复验）。
- **对照变量**：无（诊断）。
- **预期指标**：二选一定位。
- **支持/否定条件**：→ (a) 则 r24 启动前必须加「任务内 commit 回显」到启动脚本（I73）；→ (b) 则修 logger 键路径。
- **成本风险**：零。
- **worker 的下一步**：R009 前完成；结果决定 I59 终裁是否要等 r24。

### I72 · 0018 相位排除正式撤回 + 裁决路径更新
- **关联结果版本**：idear-0018 §B（作废）；本报告 §D
- **内容**：摆幅实测 0.61m 后，相位失配触发 ~0.84-1.10s 与观测 0.76s 同量级——**相位失配与跌倒/失稳并列为活跃候选**（不再有排除法优势）。裁决依赖：(i) I64 分桶修复后的 r24（pose_fail-only vs low_root 份额）；(ii) 零算力旁证：E4 的 root_h 时序（I68 已落地脚本）——若失稳前 root_h 平稳而 body 偏差先超阈 → 相位/构型失配主导；root_h 先降 → 跌倒主导。r20 权重本地即可跑（不占算力）。
- **worker 的下一步**：r23 E4⁗ 时对 r20/r23 各跑一次 root_h 时序对照；I71 结果出来后定 r24 是否带修好的 I64。

### I73 · 启动脚本加「commit 回显」（一行，r24 起生效）
- **关联结果版本**：I71-(a) 的预防措施；经验库「启动期最小探针」条目
- **最小改动或诊断实验**：任务启动命令前加 `git rev-parse HEAD > CODE_VERSION.txt && cat CODE_VERSION.txt`（或 echo 进 console 首行）——上传侧一致性的永久探针，成本一行。
- **worker 的下一步**：与任何 r24 配置改动合并。

---

## 最小验证实验

| ID | 内容 | 依赖 | 预算 | 判据 |
|---|---|---|---|---|
| I70 五步 | r23 下载验证（唯一性/时间线/写盘时序/文件级/行为指纹） | r23 完成 | 10 分钟 | 五步全过 |
| I71 | 部署判别（commit 回显 grep 或 logger 链复验） | 现有日志 | 2 分钟 | (a)/(b) 定位 |
| I72-ii | r20/r23 root_h 时序对照 | 本地 MuJoCo + r20/r23 权重 | 半小时 | 相位 vs 跌倒旁证 |
| R009 | r23 收官 + I66 三档判定 + I70/I71 结果表 | — | — | — |

---

## 上下游缺口

1. **I64 在 r23 缺席**（若 I71-(a)）——I59 终裁推迟到 r24；r24 需同时带 I73 探针。
2. `_obs_norm._count` 无强区分度后，「下载正确性」的最强检查落在时间线+行为指纹（I70-2/3/5）——建议 worker 把 I70 固化为下载 SOP。
3. 行为指纹的 Isaac 侧依赖 gm play 通道（r14 时代验证过）；MuJoCo 侧 E4 现成。
4. r23 若在最后 12% 被杀（第 9 次余额击杀风险仍在——账号 31-33 为用户新充，余量未知），权重丢失重演——I70 清单同样适用于下次。

---

## 反证与不确定性

- **[已观测] 分支内容/日志列枚举/count 算术/摆幅实测消费**：raw 拉取与本地解析，脚本归档。
- **[推断-中置信] I64 缺席 = 部署不一致**：备选解释 是 logger 链细节，I71 两分钟可裁；在裁决前不应断言「上传错版本」，只能说「运行行为与 df80895 声明不符」。
- **[推断] count=765 iter 冻结**：+2 iter 偏移（naive 763）未逐行溯源（update 时序细节）；不影响「预算不敏感」结论（冻结点 ~100M ≪ 300M/461M）。
- **[撤回声明] 0018 §B 的相位排除**：基于错误摆幅假设（0.5m）；实测 0.61m 下触发时间与观测同量级，排除不成立。0018 的「跌倒主导」从「双重排除」降回「单重排除（root 漂移被 E19 排除）+ 量级证据」——仍是最优解释但不再独占。
- **[推断] r23 中途 23.6 → ~23-25 档**：与 0019 外推一致（最后 12% 出相变的先例为零，但非不可能）。
- **[说明] 本报告聚焦用户指导的审计**；r23 的 I66 三档判定等 R009 数据齐全后由 worker 出，我方 0019 的矩阵不变。

---

## 来源

**用户指导**：taskContract r3（2026-10-06 实时补充：审核模型下载/上传正确性）。

**共享证据（commit `a31284c` 及运行中文件，只读）**
- `.repos/note-054.json`（r23 配方与成功判据）、`.repos/task-r23.json`
- `.repos/r23log4.json`（iter 2850-3076，列枚举与末窗统计——本报告解析脚本 `research/studio/idear/20261006_r23_parse.py`）
- commit `a31284c` 信息（I68 实测摆幅 0.61m；I66-I69 accepted）

**分支源码（raw 拉取）**
- `df80895`：`mimickit/learning/base_agent.py` L66-100（保存链三要素）、`data/agents/smp_x1_agent.yaml`（配置项 grep）

**历史验证证据（承接）**
- R004/R005/R007 的 first-slot 三验记录（createTime/_obs_norm._count=100,270,080）

**本报告计算**
- `research/studio/idear/20261006_audit_arithmetic.py`（count 算术 + 相位修正）

**缺失证据**：r23 终局日志与 model list（任务收尾中）；任务构建 commit 回显（可能不存在——I73 要补的正是这个）；logger 对 `_diagnostics` 键的表化路径源码级确认。
