# idear 报告 0004：SMP 冻结根因再分析——0.003 奖励是设计工作点，先诊断后调参

- 报告 ID：idear-ideas-0004
- 日期：2026-10-04
- 作者：idear（研究与建议角色）
- 关联结果版本：**dm-results-R001-baseline v1**（artifactId `dm-results-R001-baseline`，digest sha256:e38a135f…，worker 报告 R001-dm-baseline-20261004_0010）
- 新增证据：① x1_mimicKit 源码级核实（commit `88dbd89e`，4 个关键文件已归档 `research/studio/idear/mimickit_ref/`）；② SMP 原论文（arXiv 2512.03028, SIGGRAPH 2026）配方对照（[R4] 归档）；③ 帧格式实测复核
- 状态：建议（非验收）。**本报告结论与 R001 §5 的根因推断相左，请 worker 在采纳任一修复前先跑 E6 诊断（零训练成本部分当天可出）**

---

## 摘要与关联结果

R001 判定 SMP 策略冻结（r3: 853 iter / 111M samples，Smp_Reward_Mean≈0.003，Return 0，Imp_Ratio 1）根因为「sds_loss_scale=6 的 exp 核饱和 + SGD 1e-4 过慢」，计划 sds_loss_scale 6→≤2 + SGD→Adam(3e-4) 后重启训练。本报告经源码与原论文对照，给出三点修正：

1. **`Smp_Reward_Mean≈0.003` 是 SMP 论文配方的设计内工作点，不是饱和异常**。`mimickit/learning/diff_normalizer.py` L54–58 的 `DiffNormalizer.normalize(x)` 是 `x / mean|x|`（running 绝对值均值，**无中心化**，与 z-score 不同）；策略 SDS loss 恰处历史均值时 `mean_sds_loss_norm≈1` → 奖励 `exp(-1×6)=0.0025≈0.003`。论文 100STYLE 基线同样运行在这个量级。**以 0.003 判定饱和是误读**；判定冻结是否由奖励无信号造成，应看 R001 未引用的 `smp_reward_std` / `sds_loss_std`（`smp_agent.py` L172–179、L214–216 已记录到日志）。
2. **当前 x1 配置逐项等于 SMP 论文「单 clip 模仿」官方模板**（scale=6、SGD 1e-4、task_reward_weight 0.0 / smp 1.0、GSI False、K=[22,15,8]、norm_adv_clip 4.0——[R4] §A 逐项核实）。论文在此模板下单 clip 策略约 30 min @ RTX 4090 / 600M samples 训成。x1 r3 跑了 111M samples（论文预算 18%）未成且 Imp_Ratio=1——**在改被论文验证过的配方之前，应先排查 x1 特有的两个失败源**（下方 H-A/H-B）。
3. **两个更强竞争根因假设**（均源自 x1 与论文设定的数据差异，均可低成本检验）：
   - **H-A 先验判别力缺失**：x1_prior.pt 在 7 段/约 92 s/约 2.6k 窗口的数据上训练（论文 100STYLE 先验：20 h/100 风格），先验可能退化/过拟合/模式坍缩 → 对物理仿真状态的 SDS loss 高且**平坦**（无判别方差）→ advantage 为噪声 → 策略冻结。
   - **H-B 滑移流形不可达**：R001 自测的支撑相滑移占前进量 25–35% 意味着数据流形包含**物理不可达状态**（支撑相脚底速度≠0 被摩擦禁止）。若 disc_obs 含速度分量，策略可达域与先验流形系统性不相交 → SDS loss 恒高且梯度指向「学会滑移」这一不可能方向 → 冻结。

另含一处对 idear-ideas-0001 的**勘误**（见「反证与不确定性」末尾）。

---

## 论文与实现证据

**A. 源码核实（x1_mimicKit @ `88dbd89e`，文件已归档 `research/studio/idear/mimickit_ref/`）**
- `data/agents/smp_x1_agent.yaml`（全文 50 行）：SGD 1e-4（L12–17）、actor_std FIXED 0.05（L6–7）、steps_per_iter 32 / actor_batch 4 / critic_batch 2（L20/27–29）、norm_adv_clip 4.0（L33）、**sds_loss_scale 6、diffusion_steps [22,15,8]（L40–41）、task_reward_weight 0.0 / smp_reward_weight 1.0（L45–46）、enable_gsi False（L43）**。
- `mimickit/learning/smp_agent.py`：L185–218 `_calc_smp_rewards`：`ESM_SDS_loss(norm_x_obs, t_lst=K)` → `sds_normalizer.normalize`（按 timestep 维）→ mean over K → `exp(-mean_norm × 6)`；L172–179 记录 `smp_reward_mean/std`，L214–216 记录 `sds_loss_mean/std`；L28 `sds_normalizer_samples` 默认 inf（x1 yaml 未设 → 与论文 1e8 行为等效：111M < 1e8，全程更新）。
- `mimickit/learning/diff_normalizer.py` L17–25（record 用 `sum(|x|)`）、L54–58（normalize = `x / clamp_min(mean_abs, 1e-4)`，clip=inf 无裁剪）——**scale-only 归一化，输出恒 ≥0，无中心化**。
- `mimickit/envs/smp_env.py` L34–38：无 GSI buffer 时回落 `amp_env._reset_ref_motion`（= 从 7 段数据做 RSI 初始化）——r3 有 RSI，初始化在流形上，排除「初始化远流形」这一失败源。
- `mimickit/learning/tinymdm/arch.py` L9–93：TinyStableMotionDiT 默认 4 层 × 4 头 × head_dim 64（inner_dim 256），1×1 conv 前后残差 + AdaLN-Single——容量约 1–2M 参数级（x1 实际 config 在 `tools/diffusion_model/config/tinymdm_x1_multi_clip.yaml`，未拉取，参数量以该文件为准）。

**B. SMP 原论文配方（[R4] `research/studio/idear/20261004_sds-as-reward-smp-tadpole-smiling.md` §A，源：arXiv 2512.03028 + 官方配置）**
- 奖励 Eq.8：`r = exp(−(w_s/|K|)·Σ_{i∈K}‖ε̂_i−ε_i‖²)`，K={0.44N,0.30N,0.16N}（N=50 → [22,15,8]，与 x1 yaml 精确一致）；「按 DeepMimic 惯例取 exp 将奖励归一到 [0,1]，实证更好」。
- 自适应归一化：按 timestep 的 SDS error **running mean** 归一——与 diff_normalizer 实现一致。
- 官方调参优先级（repo 文档）：**smp_reward_weight > sds_loss_scale ≥ diffusion_steps**；单 clip：task 0.0/smp 1.0、GSI False、SGD 1e-4；任务：0.5/0.5、GSI True。
- 训练量级：单 clip ≈ 30 min/RTX 4090；任务策略 ≈ 600M samples；4096 并行 envs。
- 论文自述失败模式：随机 t 导致奖励方差 → 固定 K 修复；高噪声-only 引导 → 行为平均化；**低密度区 raw score 不可靠**（Fig.3 机制：前向扩散把状态拉回高密度区使 score 有效）。
- AMP-Frozen 对照（Table 1）：冻结判别器被策略剥削（style accuracy 0.00–0.43），SMP 的 score 奖励不可剥削——SMP 相对 AMP 的核心卖点。

**C. 与 R001 观测的交叉验证**
- 111M samples ÷ 853 iter ÷ 32 steps ≈ 4069 envs ≈ 论文 4096 并行（r3 环境规模与论文一致）。
- `Smp_Reward_Mean≈0.003`：`exp(-6×0.97)`，即当前策略平均 SDS loss ≈ running 均值的 0.97 倍——**策略状态的 loss 与归一化统计几乎同步**，这本身说明 normalizer 一直在追平策略分布，奖励绝对值不携带「离流形多远」的信息；判别力只能来自 batch 内方差（`smp_reward_std`）。

---

## 优先建议

### I13 · 重解读 r3 冻结：先提取现成日志曲线，再决定修什么（P0，零训练成本）
- **关联结果版本**：dm-results-R001-baseline v1
- **文件/函数**：`mimickit/learning/smp_agent.py` L172–179/214–216（已记录 `smp_reward_mean/std`、`sds_loss_mean/std`）；gradmotion 任务 TASK_20261002_060 日志
- **观测事实**：R001 引用了 Smp_Reward_Mean≈0.003 与 Imp_Ratio=1，未引用 std 曲线；上述 4 条曲线已在 r3 运行中持续记录。
- **机制假设**：冻结的三种指纹可在日志中区分——①`smp_reward_std` 持续 <0.3×mean → 奖励平坦（判别力缺失，H-A/H-B）→ 调 scale/优化器无用；②std 正常但 `Imp_Ratio` 从第 1 个 iter 起=1 → actor 更新链路问题（如 lr×梯度消失、norm_adv_clip 交互 bug）→ 代码级排查；③sds_loss_mean 快速上升后平台 → 策略主动漂离（探索崩溃）→ 检查 early termination 与 obs 归一化漂移。
- **竞争解释**：若 std 正常且 Imp_Ratio 缓慢偏离后回落——则真是「奖励信号弱」（H-C），worker 的 scale↓+Adam 有理。
- **最小改动或诊断实验**：从 TASK_20261002_060 日志提取 4 条曲线（iter 轴），判读指纹；无需任何训练。
- **对照变量**：无需（观测性实验）。
- **预期指标**：确定冻结指纹类型，指向 H-A/H-B/H-C 之一。
- **支持/否定条件**：见上①②③。
- **成本风险**：近零。
- **worker 的下一步**：日志提取 + 指纹判读（半天内），结果记入下一个 dm-results。

### I14 · 先验判别力三层测试（E6 主体；区分 H-A 与 H-B，指导修先验还是修数据）
- **关联结果版本**：dm-results-R001-baseline v1（滑移发现）
- **文件/函数**：`x1_prior.pt` + `tools/diffusion_model/config/tinymdm_x1_multi_clip.yaml`（`args/smp_x1_prior_args_README.md` 归档件 L21 给出训练入口）
- **观测事实**：先验在 7 段/92 s 数据上训练；R001 实测数据支撑相滑移 25–35%；TinyMDM 默认容量 4L×4H×64d。
- **机制假设**：H-A：先验对输入扰动过敏（过拟合）→ 加噪后 loss 飙升；H-B：先验把「滑移速度」编码进流形 → 消滑移重投影后 loss 大幅上升且梯度方向指向恢复滑移。
- **竞争解释**：两者可同时存在；也可能先验健康（噪声平滑、消滑移后 loss 变化温和）→ 冻结另有原因（回到 I13 指纹②③）。
- **最小改动或诊断实验**（离线脚本，本地或远端单卡分钟级）：
  1. **基线**：7 段数据滑窗（H=10，同 disc_obs 构造）→ ESM_SDS_loss(K=[22,15,8]) 分布（应最低）；
  2. **扰动泛化**：窗口加 σ∈{0.01,0.05,0.1}×各维 std 的独立高斯噪声 → loss 增长曲线；健康先验应平滑单调上升，过敏先验在 σ=0.01 即量级跳变；
  3. **消滑移重投影**：对支撑相帧（复用 R001 触地判定）把 root 前向速度按脚速重投影（即强制无滑移的速度场），重算 loss——若 loss 大涨（>2×），流形确实锚定在滑移上，H-B 成立。
  - 附：镜像/时间反转等不变性检查可加但不关键。
- **对照变量**：σ 档位；有/无滑移重投影；K 的 3 个 timestep 分别看（低 t 对抖动敏感，[R4] §A）。
- **预期指标**：得到「先验对物理可达状态的判别力」量化表；判定 H-A/H-B。
- **支持/否定条件**：H-A 支持=σ=0.01 即 loss 跳变 ≥10×；H-B 支持=消滑移后 loss ≥2×；两者皆否=先验健康，转 I13 指纹③或 H-C。
- **成本风险**：低（离线推理）；需要 worker 环境有 torch + pkl 数据（都有）。
- **worker 的下一步**：写 ~100 行诊断脚本（建议入 `analysis/`），输出分布表进 dm-results。

### I15 · 暂缓「scale 6→≤2 + Adam」为默认修复，改为受控消融（若 I13/I14 指向 H-C 才升级为主修复）
- **关联结果版本**：dm-results-R001-baseline v1 §6 计划 1–3
- **文件/函数**：`data/agents/smp_x1_agent.yaml` L12–17/L40
- **观测事实**：该修复改动的是与 SMP 论文单 clip 官方模板一致的项；论文调参优先级把 `smp_reward_weight` 排在 `sds_loss_scale` 之前。
- **机制假设**：若 H-A/H-B 成立，scale↓ 只把 e^-6 抬到 e^-2≈0.135，但奖励**方差**（信息量）不变——advantage 仍是噪声，Adam 只放大噪声跟随；白烧一轮远端训练。
- **竞争解释**：若 I13 指纹③或 H-C 成立（奖励信号真实存在但弱），scale↓+Adam 合理且应按论文优先级同时扫 smp_reward_weight。
- **最小改动或诊断实验**：把 R001 计划 3 的重启改为 2×2 消融：{scale 6, scale 2}×{SGD 1e-4, Adam 3e-4}，每格 50 iter（约 6.5M samples，远小于 853 iter 的 13%）看 `smp_reward_std` 与 Imp_Ratio 是否离开 1——作为指纹验证的补充。
- **对照变量**：scale×optimizer 四格；观测 smp_reward_std / imp_ratio / return 三指标的前 50 iter 斜率。
- **预期指标**：四格中若无任何一格 Imp_Ratio 离开 1 → 强证伪 H-C，回到 H-A/H-B。
- **支持/否定条件**：支持 H-C=scale 2 或 Adam 格出现 return 上升；否定=四格全冻结。
- **成本风险**：中（四格各 50 iter 的远端算力）；比直接全量重启 853 iter 便宜 8×。
- **worker 的下一步**：若坚持先重启，至少把四格消融作为并行小任务先跑。

### I16 · 数据侧根治滑移（若 H-B 成立）：先验训练数据的「物理可达化」增广
- **关联结果版本**：dm-results-R001-baseline v1 §4.1（滑移观测）；idear-ideas-0003（数据现状）
- **文件/函数**：`tools/x1_pipeline/`（正本，重投影脚本挂此）；`retarget_v3.py` 的支撑相判定可复用（`validate_retarget_v3.py` L31 STANCE_Z 口径）
- **观测事实**：v3 门修了平底/闭合/抖动，未做世界系锚定（R001 推断 v3 IK 遗留）；滑移占前进量 25–35%。
- **机制假设**：SDS 流形只需「速度场自洽的近似流形」而非精确 mocap——支撑相速度重投影（root/dof 速度按触地脚速度平移）得到的变体物理可达，先验在其上训练后策略可达域与流形交叠恢复。
- **竞争解释**：重投影会改变速度分布（root 净前进量下降）→ 需同步微调数据整体速度（或接受更慢的先验速度带）；另一路线是不修数据、给 disc_obs 去掉速度分量（改先验输入域）——工程上更侵入，需改 `_check_prior_env_config` 相关项，不推荐先走。
- **最小改动或诊断实验**：对 7 段做支撑相速度重投影生成 v3.1 变体（不覆盖原数据），重训先验（同 config），用 I14 的测试 3 复测 loss。
- **对照变量**：v3 原数据 vs v3.1 重投影数据训出的两个先验；判别力表对比。
- **预期指标**：v3.1 先验对物理 rollout 状态（或消滑移窗口）的 loss 显著更低、方差更大（判别力恢复）。
- **支持/否定条件**：支持=v3.1 先验下策略训练 return 离开 0；否定=仍冻结 → 主因非滑移，回到 H-A（先验容量/训练不足）。
- **成本风险**：中（数据脚本 + 一次先验重训，远端单卡小时级）；不动原数据、不动正本门。
- **worker 的下一步**：待 E6 结果后决定是否启动。

---

## 最小验证实验

| ID | 内容 | 依赖 | 预算 | 判据 |
|---|---|---|---|---|
| E6a | r3 日志四曲线提取与指纹判读（I13） | gradmotion 日志（已有） | 分钟级 | 指纹①/②/③ 判定 |
| E6b | 先验判别力三层测试：基线/加噪/消滑移重投影（I14） | torch 环境 + x1_prior.pt | 单卡分钟级 | H-A/H-B 判定 |
| E7 | （条件启动）v3.1 重投影数据 + 先验重训 + E6b 复测（I16） | E6b 指向 H-B | 单卡小时级 | 判别力恢复 |
| E8 | （条件启动）2×2 消融 {scale}×{optimizer}×50 iter（I15） | E6a 无法排除 H-C | 远端 4×50 iter | 任一格解冻与否 |

执行顺序：E6a → E6b →（E7 ∥ E8 按判定结果条件启动）。**全部完成前不建议 853 iter 级的重启训练。**

---

## 上下游缺口

1. **`tinymdm_x1_multi_clip.yaml` 未获取**（x1 先验实际容量/训练超参/window/归一化域）——网络通道已打通（raw 单文件可拉），worker 侧可直接读正本；这是 E6b 解读的前提之一。
2. **disc_obs 的确切构造未核实**（`amp_env.get_disc_obs` 的分量与顺序、是否含 root_vel/dof_vel）——H-B 的机制链成立与否取决于此；源码在 `mimickit/envs/amp_env.py`（未拉取，慢网络下建议 worker 本地读）。
3. **先验训练日志缺失**：x1_prior.pt 的训练 loss 曲线（是否收敛/过拟合）不在 R001 证据内；H-A 判定需要。
4. **r2 的 CUDA assert 未定位**：环境构建期 physx 崩溃与冻结无关但会复现于重启——建议在重启前先在 r3 已验证可跑的配置上做增量改动，避免同时引入两个变量。

---

## 反证与不确定性

- **[已观测] 配置=论文单 clip 模板**：x1 yaml 与 [R4] §A 官方配置逐项一致（本报告核实）。这**削弱但未证伪** H-C——论文模板在其数据上成立，不代表在 x1 数据域上成立。
- **[推断] 0.003=设计工作点**：由 diff_normalizer 语义（x/mean|x|）+ exp(-6×1)=0.0025 数值吻合推出；若 r3 的 `sds_normalizer` 统计因初始化（count=0→mean_abs=1）在前若干 iter 异常，早期奖励会偏高后跌落——日志前 20 iter 曲线可辨。
- **[推断] H-A/H-B 机制链**：均未经实验；E6b 是直接检验。H-B 依赖 disc_obs 含速度分量的未核实前提（见缺口 2）。
- **[推断] TinyMDM 容量 1–2M**：默认 arch 参数推算；x1 实际 config 未读，若 num_layers 调大过拟合风险变化。
- **[未验证] 论文 GSI/归一化消融数字**：[R4] 标注 §10 消融表因页面截断未取得——GSI 对单 clip 的影响幅度无数字支撑（但 x1 单 clip 设置 GSI=False 与论文一致，不构成 x1 特有差异）。
- **[勘误] idear-ideas-0001 的帧格式错误**：0001 写「36 维/帧 = pos(3)+quat(4)+29dof」；实测帧长 35、[3:6] 为 **expmap**（范数 1.29 无单位四元数约束），与 R001 §2 一致。0001 的其余结论（速度带宽、奖励、sim2sim）不受影响；0003 的实测基于前 3 列 pos，不受影响。
- **[未验证] gradmotion 平台侧**：日志提取命令、任务参数覆盖机制均以 R001 的描述为准，我无平台一手访问。

---

## 来源

**共享产物（已 fetch 快照）**
- dm-results-R001-baseline v1（`inputs/studio/dm-results-R001-baseline/20261004_001020_R001_baseline_inventory.md`，sha256:e38a135f…）

**x1_mimicKit 源码（commit `88dbd89e`，归档 `research/studio/idear/mimickit_ref/`）**
- `data/agents/smp_x1_agent.yaml`（全文）
- `mimickit/learning/smp_agent.py` L24–46/118–127/162–218
- `mimickit/learning/diff_normalizer.py` L17–58
- `mimickit/envs/smp_env.py` L34–74
- `mimickit/learning/tinymdm/arch.py` L9–93
- `args/smp_x1_prior_args_README.md`（先验训练入口）

**文献（归档 `research/studio/idear/`）**
- [R4] `20261004_sds-as-reward-smp-tadpole-smiling.md`：SMP（arXiv 2512.03028）、TADPoLe（arXiv 2407.01903）、SMILING（arXiv 2410.13855）、Diffusion Reward（arXiv 2312.14134），5 源

**本地（commit `fc0b9be`）**
- `x1_retargeted_motion/*.pkl` 帧格式复核（35 列 expmap）

**缺失证据**：`tinymdm_x1_multi_clip.yaml`；`amp_env.py` disc_obs 构造；先验训练曲线；r3 日志的 std 曲线。
