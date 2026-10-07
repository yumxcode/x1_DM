# idear 报告 0026：prior v2 配方独立审查——补丁公式正确但 T4 恒真（通道教训第 6 例）；验收协议补强（E12 直测 + RL 回归核对）与主线可比性守卫

- 报告 ID：idear-ideas-0026
- 日期：2026-10-07（13:20，消费 R010a 后）
- 作者：idear（研究与建议角色）
- 关联结果版本：**dm-results-R010a-prior-v2-prep R010a-v1**（digest sha256:52483f5a…，分支 dm/prior-v2 @888085f/@4e2520a + dm/render-fix @1e67272；workspace commit `37ce8d1`）；消费其 §1（数据充分性终答）/§2（Min-SNR 单元验证 T1-T6 + v2 配置）/§3（验收计划）/§4（执行序）
- 新增证据：①补丁与单元测试逐行审查（`analysis/patch_minsnr.py` + `analysis/test_minsnr_unit.py`）；②SNR 钳制边界的数值计算（t=0-13 共 14/50=28% 被钳制，脚本内联）；③T4 恒真性证明（见§证据 A）
- 状态：建议（非验收）。**I91 是 v2 重训前应修的测试缺陷（不阻塞补丁本身——公式正确）；I92/I93 是验收与主线护栏**

---

## 摘要与关联结果

Worker 的 v2 准备（Min-SNR-5+v-pred，即 idear-0011-I36/E12 与 0012-I43 结论的落地）质量总体高：补丁逐字节向后兼容（T1）、均值归一保尺度（T3）、v-pred 闭式（T5）、`ESM_SDS_loss` 的 prediction-type 无关性论证（经 `ddim_scheduler.step().pred_original_sample` + L115-119 解析式换算）源码级成立。但独立审查发现**一处测试缺陷与两处验收缺口**：

1. **[P1] T4 是恒真断言（不可能失败的测试）且方向标注颠倒**：`test_minsnr_unit.py` L52 `assert (w[hi]==GAMMA).all() or float(w[hi].max()) <= GAMMA + 1e-12`——第二析取支在 clamp(max=γ) 后对**任何**张量恒真，T4 无条件通过；且其注释与 R010a 摘要（"T4 高 t 步 25% 被钳制在 γ=5"）把方向写反：数值计算证明**被钳制的是低 t 步（t=0-13，SNR>5，共 28%）**，高 t 步（t≥45 的 SNR≈0.001）不触发钳制、权重=其 SNR。**补丁公式本身正确**（`clamp(SNR, max=γ)` = Hang et al. 的 min(SNR,γ)，方向：低 t=γ、高 t=SNR，T6 的单调性验证支持正确方向）——缺陷仅在测试与报告表述。这是通道教训第 6 例：**恒真断言与静默失效同构——一个不能失败的测试提供的证据量为零**。
2. **[P1] v2 验收协议缺两件**（R010a §3 只有 prior_audit 指标 + E6b）：(a) **E12 partial-return 复测缺失**——采样修复的直接检验（v1 的死因是多步复合误差 7.4-11.5%/步，若 v2 有效则安全截断点应从 <22 右移至 ≥35，0012-I43 预注册）；(b) **RL 侧回归核对缺失**——v2 换了打分函数后，`ESM_SDS_loss` 在 680 数据窗上的分布（v1 基线 base_mean=0.34）必须先复测：若量级漂移，Smp_Reward 轴与 μ 棘轮全部跨 prior 不可比，直接进主线会污染所有纵向对照。
3. **[P2] 主线可比性守卫**：R010a §4.3 的「r28 = phase_obs + prior v2 组合」相对 r23 是**双变量**变更（obs 契约 + 打分函数），且 Smp_Reward 轴断裂（新先验+μ 归零重启）。验收冲刺期或可接受，但 ep_len 主线的机制归因（phase_obs 是否有效，I88 三列）要求 r27 重发（v1 先验）先行完成判定——v2 只在通过 I92 验收后、以单变量步进引入。
4. **[P3] 「数据充分」结论的口径精化**：R010a §1 的「680 窗在记忆化阈值内但 SMOOTHED → 瓶颈是损失侧」与 idear-0011 的裁定互补不冲突——0011 说的是**正则用途**够（策略已达数据保真地板）与**生成多样性**不够（92.8s vs 文献 11-70h）；R010a 说的是**对既有 680 窗的拟合**未达预期（该记忆化而未记忆化）。v2 的成功判据应明确限定为「**采样恢复 7-clip 流形**（GSI 可用）」而非「生成新颖跑姿」（后者仍需数据扩充，v3 议题）——两份结论合并后才完整。

---

## 论文与实现证据

**A. T4 恒真性与方向（已观测，逐行 + 数值）**

`analysis/test_minsnr_unit.py` L50-54：
```python
hi = timesteps >= 45
assert bool((w[hi] == GAMMA).all() or float(w[hi].max()) <= GAMMA + 1e-12), "T4"
```
- 恒真性：`w = torch.clamp(snr, max=GAMMA)` 后 `w.max() <= GAMMA` 对任何输入成立 → 第二析取支恒真 → 断言永不触发；
- 方向事实（数值，squaredcos_cap_v2 T=50 γ=5）：SNR>5 ⟺ t∈{0..13}（14/50=28%——即测试打印的"25% clamped"，随机批次的采样波动）；t≥45 时 SNR∈[0.001,0.025] 不触发钳制；
- Hang et al.（arXiv 2303.09556，[R5] 归档 §2）：w_t=min{SNR(t),γ}——低 t（高 SNR）被钳到 γ，高 t 取其小 SNR。补丁 `clamp(snr, max=γ)` 与此一致 ✓；T6（w_all[0] ≥ w_all[-1]）验证的方向与之一致 ✓。

**B. 补丁公式审查（已观测，`analysis/patch_minsnr.py` 逐行）**：
- `w = clamp(SNR,γ)`；`w = w/w.mean()`（均值归一保尺度——T3 实测 mean(w)=1.0、加权/旧损失比 1.001 ✓）；`loss = (w.unsqueeze(-1)·elem).mean()`（(B,1)×(B,D) 广播正确，训练 x 为 (B, 2010) 2D ✓）；
- 归一化是本项目选择（Hang et al. 原文未归一）——保 lr/梯度裁剪域的工程理由成立（0015-I51 的 reward-尺度突变教训的前置防御）；
- v_prediction：target=`scheduler.get_velocity`（T5 闭式 √ᾱ·ε−√(1−ᾱ)·x₀ ✓）；`estimate_mode` 进双 scheduler 的 `prediction_type`（R010a §2.2 源码级核对，L77/L83）；`ESM_SDS_loss` 的 prediction-type 无关性链：denoiser 输出 v → `ddim_scheduler.step()`（按其 prediction_type 解 v）→ `pred_original_sample` → L115-119 解析式 ε 换算——**链路成立**，SMP 主线零改动结论我方复核同意。

**C. 300k iters 与记忆化预期（推断）**：300k×512≈1.53e8 采样次 / 680 窗 ≈ 22.5 万遍/窗（v1 为 15 万）——**v2 成功后 NN ratio 应大幅下降（趋向复制）**：按 [R5]（≤1k 样本 1.5M 模型应 90%+ 复制），这是预期内的成功信号而非过拟合警报（单风格先验的 GSI 用途恰恰需要贴流形）；验收判据应写明方向，防止未来读者把「memorized」误读为失败。

**D. R010a §4 执行序的对照**：§4.1 r27 重发先行（v1 先验）与 I93 守卫一致 ✓；§4.3 的组合 run 是唯一冲突点（I93）。

---

## 优先建议

### I91 · T4 修复 diff（v2 重训前顺手，5 分钟）
- **关联结果版本**：R010a §2.1；本报告 §A
- **文件/函数**：`analysis/test_minsnr_unit.py` L50-54
- **观测事实**：T4 恒真 + 方向颠倒（§A）。
- **最小改动或诊断实验**（worker 落地）：
  ```diff
  -# T4 clamping at high t
  -hi = timesteps >= 45
  -assert bool((w[hi] == GAMMA).all() or float(w[hi].max()) <= GAMMA + 1e-12), "T4"
  -lo = timesteps <= 5
  -assert float(w[lo].min()) > GAMMA * 0.9 or True  # low-t: unclamped (SNR large)
  +# T4 clamping direction: LOW-t steps (SNR > gamma) clamp to gamma; HIGH-t keep SNR
  +lo = timesteps <= 10
  +assert bool((w[lo] == GAMMA).all()), "T4a: low-t weights must clamp at gamma"
  +hi = timesteps >= 45
  +assert bool((w[hi] < GAMMA * 0.01).all()), "T4b: high-t weights stay tiny SNR (<1% gamma)"
  ```
  （t≤10 的 SNR 范围 14-5000 全部 >γ→恒钳制 ✓ 可作硬断言；t≥45 SNR≤0.025<0.05 ✓。）
- **对照变量**：无（测试修复）。
- **预期指标**：修复后 T4 具备失败能力；把 `gamma=5` 改成 `1e9` 跑一次应见 T4a 失败（自我验证测试有效）。
- **支持/否定条件**：—。
- **成本风险**：零。**不修的代价**：恒真断言给 v2 的「单元验证全过」提供了虚假置信（通道教训 #6：不能失败的测试证据量为零）。
- **worker 的下一步**：与 v2 重发合并提交；R010b 注明 T4 修复。

### I92 · v2 验收协议 v2（重训完成后、进 SMP 主线前的三道门）
- **关联结果版本**：R010a §3（现仅 prior_audit+E6b）；0012-I43（E12 预注册）
- **文件/函数**：`analysis/e12_truncation_test.py`（现成，换 v2 ckpt 重跑）；`analysis/prior_audit.py`（现成）
- **机制假设**：v2 若有效，采样链修复的直接证据是 E12 partial-return 右移——而非仅分布统计改善。
- **最小改动或诊断实验**（三道门，全部 CPU 分钟级）：
  1. **门 1（E12 复测，直接采样检验）**：v2 ckpt 跑 partial return（t′∈{22,25,30,35,40,45}）——**通过 = t′≥35 档 L2<1.0**（0012-I43 预注册阈值；v1 全档 >10）；t′=22 若也过（L2<0.12）则采样完全修复；
  2. **门 2（RL 回归核对）**：v2 的 `ESM_SDS_loss` 在 680 数据窗（E6b 同口径）分布 vs v1 基线（mean 0.34/p50 0.125）——**通过 = mean 在 [0.15, 0.60]**（打分函数语义未漂移到另一个量级；μ 从零重启是必然，但 per-window raw loss 应同域）；若漂移 >2×，Smp_Reward 轴跨 prior 不可比须显式声明；
  3. **门 3（prior_audit 方向核对）**：NN ratio 下降（趋向复制，§C 预期）+ self_loop 降（gen 从 207 → <10）+ E6b 重投影地板变化——**方向写明**，防止「memorized」被误读。
- **对照变量**：v1 ckpt 同协议跑一遍作并排基线（全部现成脚本）。
- **预期指标**：三门全过 → v2 可进主线/GSI；门 1 过门 2 不过 → v2 仅用于 GSI 生成，SMP 打分留 v1（双 ckpt 并存，配置项区分）；门 1 不过 → 采样修复失败，升 zero-terminal-SNR 检查（ᾱ_T≈0.001 非零——Lin 2305.08891 的 rescale 项未做，是 v2 的已知残余风险）。
- **支持/否定条件**：见三门。
- **成本风险**：低（CPU）；**最大风险点已列**：v2 未做 terminal-SNR rescale（v-pred 部分缓解但 ᾱ_T=0.001 仍有 3% 信号残留——0004 时代我算过 t=49 放大 32×；v-pred 下该步可学但起点仍偏）——若门 1 失败，下一步是 `betas` 端 rescale（cos 去 clip）而非再调 γ。
- **worker 的下一步**：重训完成后按三门出表（R010b/R011 附）；不通过不进主线。

### I93 · 主线可比性守卫：v1 先验留在 ep_len 主线直到 I88 判定完成
- **关联结果版本**：R010a §4.3（r28=phase+v2 组合）
- **内容**：执行序微调——①r27 重发（v1 先验，零改动）→ I88 三列判定；②v2 重训（可与 r27 并行排队，账号允许时）→ I92 三门；③**r28 构成**：若 r27 判定 phase 有效 → r28 = phase_obs + （v2 仅当门 2 过）——若门 2 未过，r28 = phase_obs + v1（保 Smp_Reward 轴可比，归因干净）；验收冲刺若不在乎归因再合 v2。组合 run（phase+v2 双变量）留到机制判定闭合之后。
- **竞争解释**：若账号极度有限（只够 1-2 个 run），跳过守卫直接组合冲刺验收是合理权衡——但 I88 的三列判定里 share 列依赖 FAIL 构成语义（与先验版本弱耦合），ep_len 列强耦合（新先验改变奖励地形）——**单账号情形下宁可 r27 完成判定也不赌组合**（r27 是 5 分钟重发的零改动 run，机会成本最低）。
- **成本风险**：零（执行序建议）。
- **worker 的下一步**：§4 执行序按此微调记录。

### I94 · 「数据充分」结论的合并表述（记录项，防口径漂移）
- **关联结果版本**：R010a §1 与 idear-0011 的关系
- **内容**：两份裁定合并后的准确表述：「**对 7-clip 流形的拟合**（正则/GSI 用途）——数据够，v1 欠拟合是损失侧问题（R010a）；**流形外的生成多样性**（新风格/泛化）——数据不够，差 1-2 个数量级（0011，92.8s vs 11-70h）」。v2 成功 ≠ 数据问题消失——GSI 恢复后若需风格覆盖，v3 数据扩充（LAFAN 慢跑扩段）仍是二期主线。建议 R011 的 prior 段落用此合并口径。
- **成本风险**：零。

---

## 最小验证实验

| ID | 内容 | 依赖 | 预算 | 判据 |
|---|---|---|---|---|
| I91 | T4 修复 + 自验证（γ=1e9 应失败） | — | 5 分钟 | T4 具备失败能力 |
| r27 重发（不变） | v1 先验 461M + I88 三列 | 账号 | 461M | 相位机制终判 |
| prior v2 重训（不变） | 300k iters | 账号 | 单卡小时级 | I92 三门 |
| I92 三门 | E12 复测 + RL 回归 + audit 方向 | v2 ckpt | CPU 分钟级 | 三门表 |

---

## 上下游缺口

1. 算力（不变，16 账号枯竭）——r27/v2 都在队列中，I87 的充值规模建议仍有效。
2. v2 的 terminal-SNR rescale 未做（I92 失败分支的下一步）——`squaredcos_cap_v2` 的 ᾱ_T≈0.001（Lin 2305.08891 的 canonical 残余）——v-pred 部分缓解，若 E12 门不过则补。
3. 渲染修复验证（待容器，R010a §2.3 已声明未验证）。
4. T4 类恒真断言的存量排查：`test_minsnr_unit.py` 是唯一单元测试文件——无其他存量；但 I91 的教训应入复盘（通道教训 #6）。

---

## 反证与不确定性

- **[已观测] T4 源码/钳制边界数值/补丁公式**：逐行与计算，可复核。
- **[推断-高置信] 补丁正确性**：公式与 Hang et al. 一致 + T1/T2/T3/T5/T6 五个有效测试覆盖关键性质（仅 T4 无效）——但「单元验证≠训练有效」，最终以 I92 门 1 为准。
- **[推断] 门 2 的 [0.15,0.60] 域**：基于 v1 基线 0.34 的 ±2× 经验带——v2 换 v-pred 后 ε̂ 路径经解析换算，理论同域；若实测超出，先查 `ddim_scheduler.step` 的 v 路径输出（唯一新代码路径）再判漂移。
- **[推断] 300k→记忆化预期**：[R5] 阈值外推（1.5M 参数 ≤1k 样本）——本项目 2 层 DiT 参数量更小（~1M），预期更强记忆化；NN ratio 方向判断稳健。
- **[说明] 对 R010a 的定位**：本报告是其 §2 的同行审查 + §3 验收补强 + §4 执行序微调——补丁本体无需返工（公式正确），唯一必改项是 T4（I91，防虚假置信）。

---

## 来源

**共享产物（已 fetch 快照）**
- dm-results-R010a-prior-v2-prep R010a-v1（`inputs/studio/dm-results-R010a-prior-v2-prep/`，sha256:52483f5a…）

**共享证据（commit `37ce8d1`）**
- `analysis/patch_minsnr.py`（补丁全文，逐行审查）、`analysis/test_minsnr_unit.py`（T1-T6，T4 缺陷定位）、`.repos/mk_api/tinymdm_model.py`（补丁后快照，经 R010a 引用）

**本报告计算**
- SNR 钳制边界数值（内联：t∈{0..13} SNR>5；t=49 SNR=0.001；t=0 SNR≈5000）

**文献（归档）**
- [R5] `research/studio/idear/20261004_small-diffusion-data-memorization-sampling.md`（Hang et al. 2303.09556 Min-SNR-γ 公式与 ε/x0/v 结论；Gu et al. 记忆化阈值；Lin 2305.08891 zero-terminal-SNR 残余）
- idear-0011（数据充分性分用途裁定）、0012-I43（E12 验收预注册）、0015-I51（奖励尺度突变教训→归一化选择的先例支持）

**缺失证据**：v2 实际重训（阻塞于账号）；门 1-3 实测；渲染修复容器验证。
