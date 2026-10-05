# idear 报告 0014：Task_Reward_Mean 疑似无效观测（r14/r16 双日志 200/200+226/226 恒等）——R005 判读前必须裁决 + 视频复核闭环 + E15 结算

- 报告 ID：idear-ideas-0014
- 日期：2026-10-05（12:55）
- 作者：idear（研究与建议角色）
- 关联结果版本：dm-results-R004 v1；共享区新证据（无新 dm-results）：commit `97289b5`（E15 + r16/r17 状态）、`.repos/r16final.json`（r16 被杀前日志）、`.repos/r14final.json`（r14 日志，本轮新分析）、`.repos/task-r17.json`/`note-039.json`（r17 任务卡）、`analysis/gm.sh`（账号 28/r17 TASK_20261005_039）
- 新增证据：①**Task_Reward_Mean ≡ Smp_Reward_Mean 逐块恒等**的发现与定量（r14: 200/200；r16: 226/226）；②worker 分支 `e472930` smp_agent.py L162-184 的接线核实（补丁本身正确）；③r14 渲染 mp4 的 video_inspect 复核（R004 遗留待办，服务恢复后完成）；④r16 末段仪表读数（含 Sds_Norm_Mu——0010-I34 已生效）
- 状态：建议（非验收）。**R005（r17 判读）前最高优先级：裁决 Task_Reward_Mean 的有效性（I48，5 分钟诊断）**

---

## 摘要与关联结果

四项内容，按优先级：

1. **[P0] Task_Reward_Mean 疑似无效观测**：r14 与 r16 的训练日志中 `Task_Reward_Mean` 与 `Smp_Reward_Mean` **逐块精确恒等**（r14: 200/200 块；r16: 226/226 块；例 iter2265 双 0.413、iter2490 双 0.402）。两个统计独立的量（DeepMimic 跟踪奖励 vs 扩散核奖励，`_calc_smp_rewards` 每次调用重采样噪声）不可能如此。而 worker 分支 `e472930` 的 I20 补丁接线正确（L178 `torch.mean(task_r)`，L163 从 buffer 读 "reward"）。根因二选一：
   - **(A) 渲染层 key 绑定错误**（logger 把两个列绑到同一数据）——无害于训练，但 task 分量曲线从未被观测；
   - **(B) env 原始 reward 通道断链**（buffer "reward" 里 env 记录的值恒 0 或从未被 env 覆盖）——**严重**：若 X≡0，则 `r = 0.5·0 + 1.0·smp = smp`，**task_reward_weight=0.5 名存实亡，r4 以来的训练实际是纯 SMP 奖励驱动**，0012/0013 的归因图景（「task 分量提供跟踪引导」）需重写为「scale 2 + Adam 解冻、smp 独自引导」。
   - 判别诊断 I48（5 分钟）；在裁决前，**R005 不应引用 Task_Reward_Mean 任何数值**。
2. **[P1] r14 渲染视频复核闭环**（R004 待办；video_inspect 第 4 次尝试成功）：本体完整（无头配置=29dof 模型正常）、**无穿模、无高频抖动**；但**未形成步态**——0.2s 起后仰倒地、重置后 1.3s 前扑跌倒。与 E4 数值判读一致（全部跌倒 0.7-1.3s）。注：视频显示「未前向移动」而 E4 报 fwd_vel 0.60-0.73——差异归因于相机跟踪模式（camera_mode: track）下的视觉静止错觉和/或视频恰含早期失败段；不影响「策略未学会行走」的一致结论。
3. **[P2] E15 结算（worker 已执行 0013-I47）**：pose_scale=0.25 是 **MimicKit 全家族 25 个 env 配置的约定**（非 x1 失误）——0013-I46-β（0.25→3.87）正当性降级，正是 0013 预注册的风险分支被证实；**I46-α（root_vel_w 0.1→0.3）升为唯一主推**（α 不动家族约定，只调速度权重）。
4. **[P2] r15/r16 事故与 r17 设计确认**：r15/r16 均死于账号余额中途杀（r5/r13/r15/r16 四连，r16@2491iter/327M/71%，smp_r 0.402 健康）；r17（TASK_20261005_039，账号 28 最后余额，11:38 启动）= **I45 框架的干净单变量实现**：ipo=100 + 预算 300M 恰对齐 r14 → r14 vs r17 只差 ipo 语义。r16 末段仪表（smp 0.398 / raw L 0.222 / μ=0.402↓ / std 0.155）与 r14 相当——无异常信号，r17 结果可期。

---

## 论文与实现证据

**A. 恒等现象（已观测，本轮解析）**

| 日志 | n 块 | Task ≡ Smp | 末段值 |
|---|---|---|---|
| `.repos/r14final.json`（iter 2090-2288） | 200 | **200/200** | 双 0.412 |
| `.repos/r16final.json`（iter 2265-2491） | 226 | **226/226** | 双 0.402/0.398 |

（gm logs 滚动窗口 ~200KB，仅覆盖末段；全程数据在任务 tfevents 归档 `rl-tfevents-logs.tgz`，未下载。）

**B. 接线核实（已观测，分支 `e472930` smp_agent.py L162-184）**：`task_r = get_data_flat("reward")`（L163）→ 混合 `r = 0.5·task_r + 1.0·smp_r`（L174）→ 写回（L175）→ `info["task_reward_mean"]=torch.mean(task_r)`（L178）。补丁正确；恒等不能由该函数的显式逻辑产生。

**C. 根因两种机制推演（推断）**：
- (A) 渲染层：`_log_train_info`/logger 的列映射把两列绑同一数据。完全解释恒等；训练不受影响。
- (B) 通道断链：`_record_data_post_step`→base_agent 的 reward 记录链若对 SMP env 断（env reward 未写入 buffer 或写 0），则 L163 读到的是**上一 iter 写回的混合值** r_prev；递归 r_N = 0.5·r_{N-1} + smp_N 的不动点为 r*=2·smp——数值上应显示 task_mean≈2×smp_mean 或 0，**不精确恒等**。故 (B) 单独不闭合恒等现象，除非叠加「每 iter 独立调用且日志取值时序特殊」——需 I48 实测裁决，不能纸面排除。
- **若 (B) 成立的最深远含义**：X≡0 → 训练奖励 = smp_r → r4 解冻三变量中 task_reward 0.5 无贡献 → 解冻归因=scale 灵敏度+Adam（0006 表格的第①②列）；0013 的「task 分量驱动跟踪」叙述作废，ep_len 67-95 的增长全部归功于 SMP 奖励引导（与 0011「策略比数据更贴流形」自洽）。

**D. r14 视频复核（已观测，video_inspect gemini-3.8-flash，sha256 6237e29a…）**：五项判定——本体完整（置信高）；未前向移动+两次倾倒（0.2s 后仰、1.3s 前扑，置信高）；无步态循环（置信高）；无穿模（中高）；无抖动但姿态控制失效（高）。

**E. r16 末段仪表（已观测，r16final.json 解析）**：Task/Smp 0.398（恒等，同上问题）、Smp_Std 0.155、Sds_Loss_Mean 0.222、**Sds_Norm_Mu 0.402-0.42**（I34 生效的直接证据；μ 随训练缓慢下降与累计棘轮自洽）。r/r_floor 估算：floor=exp(−2×0.125/0.402)=0.537，r≈0.398 → r/r_floor≈0.74（注：raw L=0.214 与 μ=0.402 反解的 r=exp(−2×0.214/0.402)=0.345 与日志 0.398 有 16% 差——μ 的三步均值口径与逐 t 归一化的差异，标注为未闭合细节）。

**F. r17 任务卡（已观测，task-r17.json/note-039.json）**：名称 x1-smp-policy-r17-ipomantics-300M；描述明确 "clean single-variable ipo comparison (r14: ipo=5000/300M/ep_len 40 vs r17: ipo=100/300M) for I45 three-state verdict"；分支 f7456dc（args-only）。

---

## 优先建议

### I48 · Task_Reward_Mean 有效性裁决（P0，R005 前必做，5-15 分钟）
- **关联结果版本**：本报告 §A-C；0006-I20（补丁源起）
- **文件/函数**：`mimickit/learning/smp_agent.py` L162-184；logger 渲染路径（`_log_train_info`/Logger）；env→buffer 的 reward 记录链（base_agent `_record_data`）
- **观测事实**：双日志 426/426 恒等 + 补丁接线正确。
- **机制假设**：(A) 渲染 key 绑定错误 或 (B) env reward 通道断链（含义见 §C）。
- **最小改动或诊断实验**（按成本升序）：
  1. **纸面快查（1 分钟）**：grep worker 分支 `_log_train_info` 与 Logger 的列定义——若两列同名变量/复制粘贴，(A) 即证；
  2. **本地 shim 微跑（10 分钟）**：用 `.repos/mimickit_shim` 构造 2 个 iter 的假数据流（env reward 填已知非零非常数序列，如 [1,2,3…]），跑 `_compute_rewards` 两次并打印 info——若 task_reward_mean 输出 ≠ 注入序列均值 → 逻辑层断链；若输出正确 → 渲染层 (A)；
  3. **tfevents 对照（15 分钟，最权威）**：下载 r14 的 `rl-tfevents-logs.tgz`，比对 Task_Reward_Mean 与 Smp_Reward_Mean 两条 scalar 原始序列——tensorboard 数据绕过文本渲染层，直接区分 (A)/(B)。
- **对照变量**：注入的已知 reward 序列。
- **预期指标**：三选一结论：渲染 bug（训练数据健康，仅修日志）/ 逻辑断链（需修 `_compute_rewards` 或记录链）/ tfevents 本身也恒等（→ 逻辑层断链实锤）。
- **支持/否定条件**：—（诊断）。
- **成本风险**：零-低。**不做此裁决的代价**：若 (B) 成立而 R005 继续按「task 分量在引导」解读，r16/r17 的归因结论全部偏航。
- **worker 的下一步**：R005 之前完成（r17 预计 ~15:00 出结果，时间刚好）。

### I49 · R005 判读守则（r17 出结果即用；对 0012-I45 的增补）
- **关联结果版本**：0012-I45 三态；本报告 §C
- **内容**：三态判定的证据列**剔除 Task_Reward_Mean**（I48 裁决前无效；裁决为 (A) 后可恢复）。修订判据：
  - 态 A（ep_len ≥90 且 E4' 速度 ≥0.9）：ipo 归因闭合 + 拖行局部最优被推翻 → 直接验收冲刺；
  - 态 B（ep_len ≥90、速度仍 0.6-0.8）：拖行局部最优证实 → **r16a=I46-α**（root_vel_w 0.1→0.3，唯一主推——E15 后 β 已降级）；
  - 态 C（ep_len 仍 40 级）：转 E8 四格消融（0006-I15）。
  - **新增第四证据列**：Sds_Norm_Mu（μ）+ raw Sds_Loss + r/r_floor（I34/I44 仪表）——r17 是首个全程带 μ 的 run，R005 应给出 μ(t) 曲线（检验棘轮预期：μ 缓慢单调降）。
- **成本风险**：零。
- **worker 的下一步**：R005 按此守则出表。

### I50 · r16 曲线的补偿分析（可选，若 r17 又被算力杀）
- **关联结果版本**：r16 事故（§F）；gm logs 200KB 窗口限制
- **内容**：r16 若成为唯一 ipo=100 长程数据（r17 再被杀的情形），从 `rl-tfevents-logs.tgz` 提取全程曲线（ep_len 若在 tfevents 有记录）补做 I45 判定；同时在 R005 注明「r16 被杀@71% 无 EVAL ep_len」的数据缺口。
- **成本风险**：低。

---

## 最小验证实验

| ID | 内容 | 依赖 | 预算 | 判据 |
|---|---|---|---|---|
| E17（新） | I48 三级裁决（grep→shim 微跑→tfevents） | 现有材料 | 5-15 分钟 | (A)/(B) 判定 |
| E14/r17（在跑） | ipo 单变量（300M=r14） | 账号 28 | ~15:00 完成 | I49 三态 |
| E16（条件） | r16a = I46-α root_vel_w 0.3 | 态 B | 100M 短训 | fwd_vel 分离 |

---

## 上下游缺口

1. **Task_Reward_Mean 有效性**（I48，本轮最高优先）。
2. gm logs 的 200KB 滚动窗口：iter 级全程曲线只能靠 tfevents（已在任务归档但需下载）——建议 worker 把 tfevents 提取并入 R005 例行。
3. 账号 28 是最后余额（r17 后若再需训练，账号层阻塞重现）——E16（若触发）可能要拼余额。
4. video_inspect 服务已恢复（本轮成功）——后续渲染复核可用；r17 出 mp4 后建议直接复核（5 分钟）。

---

## 反证与不确定性

- **[已观测] 恒等 426/426、接线正确、r16 末段仪表、视频五项判定、E15 结论、r17 任务卡**：共享文件直读/解析，可复核（解析脚本 `research/studio/idear/20261005_r16_parse.py` 归档）。
- **[推断] 根因二选一未闭合**：(B) 单独不能精确解释恒等（不动点 2×smp 或 0），(A) 完全解释但未定位到代码行——I48 三级诊断是裁决通道；不排除 (A)+(B) 并存或第三种机制（如 info 聚合的 key 冲突）。
- **[推断] (B) 的归因重写含义**：若成立，0009 的「task 分量驱动跟踪」弱化为「smp 引导+ep_len 增长」——但 0011 的「策略比数据更贴流形」与 0013 的「拖行 21% 核算」都不依赖该叙述（前者是 raw L 数值事实，后者是公式静态核算），主要损失在 0006-E9 的归因判别方法。
- **[推断] 视频「未前向移动」vs E4 fwd_vel 0.6 的差异**：相机跟踪模式是最可能解释（R001 的 camera_mode: track；渲染器由 worker 配置），未验证；不影响主结论（两证据源都判「未学会行走」）。
- **[未闭合] r/r_floor 的 16% 差**（§E）：μ 口径（三步均值 vs 逐 t）差异推测，R005 有全程 μ 曲线后可复核。
- **[说明] 恒等现象自 I20 生效（r13+）起存在**——r13 被杀@254 iter 也有日志可查证起点（未做，I48-3 tfevents 会覆盖）。

---

## 来源

**共享证据（commit `97289b5` 及未提交文件，只读）**
- `.repos/r14final.json` / `.repos/r16final.json`（训练日志文本，本轮解析）
- `.repos/task-r17.json`、`.repos/note-039.json`、`.repos/run039.json`（r17 启动与设计）
- `analysis/gm.sh`（账号 28/r17/最后余额注释）
- commit `97289b5`（E15 结论：pose_scale 全家族约定）

**分支源码（raw）**
- `e472930`（r17 分支）`mimickit/learning/smp_agent.py` L162-184（接线核实）

**本报告解析/调研**
- `research/studio/idear/20261005_r16_parse.py`（日志解析 + 分段统计）
- video_inspect `analysis/r14_sim2sim_mesh.mp4`（gemini-3.8-flash，sha256 6237e29a…，clip 0-5s@10fps）

**承接**：idear-0012（I45 三态→I49 增补）、idear-0013（I46-β 降级结算、I47 已执行）、0010-I34（μ 日志已生效的直接证据）、dm-results-R004 v1（视频复核为其遗留待办）

**缺失证据**：I48 裁决结果；tfevents 全程曲线；r17 EVAL/E4'（预计 ~15:00）。
