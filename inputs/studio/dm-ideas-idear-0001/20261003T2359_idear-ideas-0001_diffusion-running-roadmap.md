# idear 报告 0001：X1 奔跑扩散策略——路线选型、数据消费与 sim2sim 对齐的第一批建议

- 报告 ID：idear-ideas-0001
- 日期：2026-10-03
- 作者：idear（研究与建议角色）
- 关联结果版本：**无**（dm_worker 尚无已发布产物、无实验 ID；本报告基于共享仓库初始提交 `fc0b9be` 的代码与数据现状 + 两份独立文献调研）
- 状态：建议（非验收）；所有实验数字均为文献引用，非本项目实测

---

## 摘要与关联结果

共享仓库现状（commit `fc0b9be`）：X1 29DOF 模型（URDF + MJCF）、v3 重定向工具链、7 段 run 族重定向数据（`x1_retargeted_motion/*.pkl`，36 维/帧 = root pos(3)+quat(4)+29 dof，fps=30，pkl 内嵌 `time_scale` 慢放因子 1.06–1.26）。工作室尚无任何已发布产物（`studio_artifacts` 为空），dm_worker 会话在运行中但仓库无新增提交。

本报告给出 6 条建议（I1–I6）与 4 个最小验证实验（E1–E4），核心判断：

1. **路线选型**：单阶段「DPPO 式 PPO 直接微调 DDIM 扩散策略」+ PHC 家族 one-frame-ahead 跟踪目标，优于两阶段「RL tracker + latent diffusion」——后者（BeyondMimic 式）真机证据最强但工程复杂度高，适合作为二期。无论扩散与否，**必须先有 Gaussian PPO 跟踪基线**作为对照组（DPPO 论文自身的对照设计）。
2. **数据是当前最大风险**：`time_scale` 慢放烘焙 + README 已自认的 sprint 物理不自洽，意味着「reference 速度分布」与「目标奔跑速度」的关系未经验证。E1 诊断实验（本地可跑、零训练成本）应最先执行。
3. **sim2sim 对齐风险已定位**：仓库 MJCF 是**裸力矩 motor 执行器、无内置 PD**（`ctrlrange="-180 180"`），timestep 0.001；而训练侧（gradmotion/Isaac 系）惯例是 position target + PD。执行器语义不一致是 sim2sim 失败的经典来源，建议在 E2 之前就锁定对齐方案。

---

## 论文与实现证据

> 完整调研已归档：`research/studio/idear/20261003_diffusion-policy-humanoid-running-sota.md`（下称 [R1]）与 `research/studio/idear/20261003_retargeted-mocap-consumption-amp-vs-tracking.md`（下称 [R2]），含逐源细节与 sources.md。

**扩散策略 × 人形运动（[R1]）**
- **DPPO**（arXiv:2409.00588, ICLR 2025, 官方代码 irom-princeton/dppo 856★）：把 DDPM 去噪展开成内层 MDP，PPO 逐步作用在高斯似然上；预训练 K=100（cosine schedule），微调只需最后 K′≈5 步 DDIM；MLP 头比 UNet 头微调更稳；chunk Tp=执行步 Ta=4–8；value 只看 state；探索 σ 下限 0.01–0.1、似然 σ 下限 0.1；γ_denoise 折扣优势。Gym locomotion（Hopper/Walker2d/HalfCheetah）与 Franka 家具装配 57%→97% 零样本 sim2real。
- **BeyondMimic**（arXiv:2508.08241 v4, Berkeley/Stanford）：真机 Unitree 人形连续奔跑 50 m、零样本 sim2real、跑步用户偏好 84.7% vs 15.3%。要点：**latent 状态-动作共扩散**（动作空间扩散成功率仅 5% vs latent 95%，MuJoCo sim2sim 空翻实验）；**紧凑奖励** = anchor 相对 {pose, Rot6D, linvel, angvel} 的 exp(−MSE/σ²) 项 + 仅 3 个正则项（关节限位/动作变化率/自碰）；**无观测历史**（历史反而伤 sim2real）；延迟敏感性：注入 2 ms 速度误差上升，5 ms → 1/3 失败，10 ms → 2/3 失败。
- **GenPO**（arXiv:2505.18763, NeurIPS 2025）与 **TruDi**（arXiv:2606.15260）：把扩散策略稳定并入大规模并行 on-policy RL（IsaacLab 人形）的两条后继路线（精确扩散求逆 + 虚拟动作；轨迹级 KL 信任域 + MaxEnt）。若训练平台是 Isaac 系，值得对比 DPPO 内核。
- **PredActor**（arXiv:2609.24840）：滚动去噪，Jetson Orin NX 上回调 16.79 ms 中位——当前人形扩散策略板上延迟最好纪录。
- ⚠️ "DiP/AgileButDim"（Ankile 2025）**检索不到一手来源**（arXiv/OpenAlex/Crossref/GitHub/作者主页全负），[R1] 未采信任何二手数字；本项目引用它时应先找到一手 PDF。

**重定向 mocap 的 RL 消费范式（[R2]）**
- **两范式**：(a) DeepMimic/PHC 家族逐帧跟踪（one-frame-ahead goal differences + exp 核误差奖励）；(b) AMP 判别器只消费状态分布、**时间对齐问题被结构性消除**（30fps vs 控制频率对 AMP 无关）。
- **30fps→控制率**：PHC/UHC/H2O/OmniH2O 重采样到策略频率并暴露 t+1 差分目标；DeepMimic 用 phase 变量 + 周期重同步作为落后恢复。
- **速度失配**：EWC/ExBody 显式解耦——上肢跟踪姿态、下肢只跟踪根速度指令 `exp(−4.0|v_ref−v|)` 权重 6.0（最大目标权重），指令取自 reference 根运动，策略可外推到训练片段之外的速度范围（其 Fig.5）；AMP 则靠任务奖励给速度、数据只给风格。
- **不可行 reference 的处理**：H2O sim-to-data 过滤（13k→8.5k 可行片段；0.1%/1%/10%/100% 干净数据 → 52.0/58.8/61.3/72.5% 成功率）；PULSE 结论：不可行动作混入显著有害；EWC 无 RSI（reference state init）即训练崩溃（MEL 0.23 vs 16.87）。
- **H2O 奖励表**（全文核实，奔跑相关主导项）：body velocity `exp(−10‖v−v̂‖)` w=6e1、feet air time +8e2、stumble(F_xy>5F_z) −1e3、DoF pos/vel 各 2.4e1、body pos `exp(−0.5‖Δp‖²)` 4e1、action rate −0.9、termination −2e2。
- **扩散 × mocap**：PDP（SIGGRAPH Asia 2024）「noisy-state clean-action」蒸馏 DDPM 达 93.5% 跟踪（clean-clean 仅 68.8%、扰动恢复 100% vs 3.36%）；OmniH2O LfD：DDPM 8/10 > DDIM 7.75 > BC 1/10。这些是「扩散在低层之上」的三种用法：鲁棒层 / 物理一致 reference 生成器（UniPhys, CLoSD）/ 技能策略。

---

## 优先建议

### I1 · 路线选型：先 Gaussian PPO 跟踪基线，再 DPPO 式替换动作头（单阶段扩散 RL）
- **关联结果版本**：无（基线尚未建立）
- **文件/函数**：待 worker 新建训练管线；数据入口参考 `x1_gmr_retargeted_tool/gmr_to_mimickit.py`（GMR 输出→MimicKit 格式的既有转换器）
- **观测事实**：项目要求「基于扩散模型的强化学习」；仓库已有精确跟踪型数据（v3 门 R1–R9+J1–J3，12/16 strict-PASS，README L47–51）；DPPO 官方代码与配置完整可得；BeyondMimic 的 latent 共扩散复杂度显著更高。
- **机制假设**：one-frame-ahead 跟踪目标 + PPO 是当前对「少量高质量重定向片段」样本效率最高的范式（EWC/PHC 证据）；扩散头的价值在于多模态动作分布与结构化探索（DPPO §机制），在跟踪任务上应表现为更快收敛与更平滑动作，而非更高的最终跟踪率（PDP：纯跟踪 MLP 98.8% 略优于扩散 98.9%）。
- **竞争解释**：①扩散可能只在「风格多样性/鲁棒性/延迟分布」上赢，跟踪率打平——这仍是可发布结果；②若 Gaussian 基线就够，扩散的增量价值需用 I5 的延迟与扰动实验显式证明。
- **最小改动或诊断实验**：E2（Gaussian 基线）→ E3（DDIM-5 扩散头替换，DPPO 配置直接抄 `cfg/*`）。
- **对照变量**：同一奖励、同一数据、同一超参（除动作头）；DPPO 侧 chunk Ta=4、ft_denoising_steps=5、MLP 头。
- **预期指标**：跟踪成功率（UHC 家族口径：全程平均 body 距离<0.5 m）、收敛所需迭代数、动作变化率（平滑度）。
- **支持/否定条件**：支持=扩散头在等预算下收敛更快或平滑度/鲁棒性显著更好；否定=扩散头在等预算下跟踪率显著更差且无延迟/鲁棒优势（则建议转向 PDP 式蒸馏或二期 BeyondMimic latent 路线）。
- **成本风险**：DPPO 微调比 Gaussian PPO 慢至 2×（论文 App. D）；GenPO/TruDi 依赖大规模并行平台内核改造，一期不建议。
- **worker 的下一步**：在 gradmotion 上搭最小跟踪 PPO（单段 clip）验证管线闭环，再决定扩散头引入时机。

### I2 · 立即诊断 `time_scale` 慢放烘焙与速度分布（最高优先、零训练成本）
- **关联结果版本**：无（数据现状即输入）
- **文件/函数**：`x1_retargeted_motion/*.pkl`（字段 `time_scale`、`fps=30`、`frames`）；README L41–45 自述「慢放烘焙：速度适配用均匀时间膨胀（time_scale 1.06–2.78），sprint 族 duty/腾空物理不自洽（duty 18-20%@1m/s、腾空至 1.73s）」
- **观测事实**：本地实测 7 段 pkl：run1_subject2_seg0 `time_scale=1.06`、run2_subject4_seg0 `time_scale=1.26`；切段速度窗为 LAFAN 源 1.5–3.2 m/s（README L27、`cut_segments.py`）。
- **机制假设**：若训练管线按 fps=30 直接重采样而忽略 `time_scale`，策略学到的目标速度 = 源速度 ÷ time_scale（例如 2.5 m/s ÷ 1.26 ≈ 2.0 m/s），速度指令范围与真实奔跑能力评估都会系统性偏慢；反之若双重缩放（数据已慢放、训练又按 time_scale 再除一次）则更糟。sprint 族（未入库但 build_dataset_v3.py L13–19 已列出 SEGS）duty 18–20%@1 m/s 意味着「1 m/s 步速却 80% 腾空」的物理不可能步态，直接跟踪会把策略教成跳跃。
- **竞争解释**：`time_scale` 也可能已被上游 GMR 链正确应用于帧率（等效 fps=30/time_scale）；训练侧选择 AMP 范式则时间对齐问题被结构性消除（[R2] §AMP），速度由任务奖励给定——这是绕开慢放问题的合法路径，但代价是失去逐帧跟踪的精确性。
- **最小改动或诊断实验**：E1（见下）——纯本地统计，无训练。
- **对照变量**：每段 clip 的 root 前向速度分布（考虑/不考虑 time_scale 两种口径）、duty factor、腾空时长、步频、最大脚高。
- **预期指标**：得到「真实速度 × time_scale」矩阵；判定 7 段 run 数据在物理自洽门（duty、腾空时长与人形姿态标准）下是否合格。
- **支持/否定条件**：支持慢放有害假设=存在 clip 物理不自洽（腾空时长超出人形奔跑合理范围、duty 异常）；否定=所有 run 段物理量正常，time_scale 仅为轻微整体缩放，可通过速度指令重标定吸收。
- **成本风险**：低（本地脚本）。风险仅在 worker 训练已开始且忽略 time_scale——那会浪费一轮远端训练。
- **worker 的下一步**：跑 E1，把每段 clip 的物理量表（速度/duty/腾空/步频/脚高）写进 dm-results，作为后续速度指令设计的依据。

### I3 · 数据消费格式：one-frame-ahead goal differences + RSI + 周期重同步
- **关联结果版本**：无
- **文件/函数**：训练管线数据加载器（待建）；参考 `gmr_to_mimickit.py` 的帧结构（36 维）
- **观测事实**：fps=30 → 控制率重采样必然发生；7 段均为 loop_mode=0（单段循环）。
- **机制假设**：PHC 家族 goal = `(q̂_{t+1}⊖q_t, p̂_{t+1}−p_t, v̂_{t+1}−v_t, ω̂_t−ω_t, q̂_{t+1}, p̂_{t+1})` 的差分形式比绝对目标更利于学习（误差信号小、对根漂移不敏感）；EWC 证据：去掉 RSI 训练崩溃（MEL 0.23→16.87），故 RSI 必配；DeepMimic 周期重同步（wrap-around 时重置 reference 根到位姿）是廉价防漂移手段，loop_mode=0 单段循环正需要它。
- **竞争解释**：AMP 范式不需要任何时间机制；若 E2 基线用跟踪式收敛太慢，可切换 AMP+速度任务奖励（[R2] IsaacGymEnvs 配方全参数可得：disc [1024,512]、GP 5、disc_coef 5、logit_reg 0.05、纯风格奖励）。
- **最小改动或诊断实验**：E2 内置 RSI on/off 对照（各 1 次短训）。
- **对照变量**：goal 差分 vs 绝对；RSI on/off；30→50/100/200 Hz 重采样率。
- **预期指标**：跟踪成功率、早期终止率曲线（RSI off 应显著变差，复现 EWC 现象即验证机制）。
- **支持/否定条件**：支持=差分 goal + RSI 显著优于绝对 goal / 无 RSI；否定=差异不显著（则简化管线）。
- **成本风险**：低；仅注意 29 dof 关节差分需按 URDF 关节限位语义（`build_x1_assets.py` parse_urdf_limits 已有解析）正确处理角度wrap。
- **worker 的下一步**：数据加载器实现差分 goal + RSI，E2 验证。

### I4 · 奖励设计：H2O 表为骨架、BeyondMimic 紧凑化为目标形态
- **关联结果版本**：无
- **文件/函数**：训练 reward 配置（待建）
- **观测事实**：项目数据是全身 29 dof 跟踪型；文献中奔跑任务主导奖励项明确（[R2] H2O Table I 全文核实）。
- **机制假设**：动态步态由 body-velocity(6e1)/air-time(8e2)/stumble(−1e3) 主导；BeyondMimic 证明仅 3 个正则项（joint-limit、action-rate、self-collision）+ anchor 相对 exp 核跟踪就够真机自然奔跑——奖励工程过度反而伤自然度。
- **竞争解释**：X1 双臂质量占比与 H1 不同，air-time/stumble 阈值需重标定；「上肢跟踪+下肢速度指令」的 EWC 解耦可能是对 X1 更稳的起点（尤其若 E1 发现下肢 reference 物理不自洽）。
- **最小改动或诊断实验**：E2 起步直接用 H2O 数值，跑通后做一轮「砍项消融」：full H2O 表 vs BeyondMimic 紧凑版（仅 exp 跟踪 + 3 正则）。
- **对照变量**：奖励版本（full/紧凑）；air-time 阈值 0.25 s 是否适配 X1 步频。
- **预期指标**：跟踪成功率、速度跟踪误差、真机/仿真步态自然度（GRF 双峰/单峰形态——BeyondMimic 用力板验证的口径）。
- **支持/否定条件**：支持=紧凑版在主要指标不输且动作更平滑；否定=紧凑版明显掉点则保留 H2O 全表。
- **成本风险**：低-中（消融各需一轮短训）。
- **worker 的下一步**：E2 用 H2O 表数值起步；结果回报后我再给砍项顺序建议。

### I5 · sim2sim 执行器语义对齐 + 延迟敏感性（MuJoCo 验证协议设计）
- **关联结果版本**：无
- **文件/函数**：`X1_29DOF/mjcf/robot/xyber_x1/xyber_x1_serial.xml` L252–（actuator 全部为 `<motor>`，如 L253 `motor_left_hip_pitch ctrlrange="-180 180"`）；`<option timestep="0.001">` L5；L284–354 有 jointpos/jointvel 传感器
- **观测事实**：仓库 MJCF 是裸力矩执行器、无内置 PD、ctrlrange 数值 ±180/±150 疑似未经质量/力矩量纲校准；传感器齐全。
- **机制假设**：若训练在 Isaac 系平台用 position target + PD（惯例），而 MuJoCo 验证侧只有裸 motor，则 sim2sim 必须在 MuJoCo 侧手写 PD 环（或生成 position-actuator 版 xml，`build_x1_assets.py` 可扩展）；PD 增益不对齐是 sim2sim 失败的首要来源之一。BeyondMimic 明确：中等阻抗 PD（高阻抗=运动学回放、真机不可行）、Rot6D 旋转表示、无观测历史、armature 准确（零 armature → 过冲/自碰）是零样本迁移关键。EWC 提醒 graphics 风格策略常假设 ~60 k·m PD 刚度——超真机 10 倍。
- **竞争解释**：sim2sim 失败也可能源于 timestep/求解器参数、接触参数差异——需在延迟扫描中一并控制。
- **最小改动或诊断实验**：E4——sim2sim 协议含三件套：(a) PD 增益扫描表（训练值 ±2×）；(b) 注入延迟 0/2/5/10 ms（BeyondMimic 延迟敏感性曲线为对照）；(c) 速度跟踪误差 + 50 m 连续奔跑不掉线（BeyondMimic 验收口径）。
- **对照变量**：PD kp/kv、控制率 50 vs 100 Hz、延迟档位。
- **预期指标**：各档位下速度误差与失败率；确定部署延迟预算（≤5 ms 目标）。
- **支持/否定条件**：支持=延迟/PD 扫描解释大部分 sim2sim gap；否定=gap 主要来自其他源（接触/惯性参数）→ 转向系统辨识。
- **成本风险**：中；需要 worker 在 MuJoCo 侧建 PD 环。建议 diff 写入下期报告由 worker 决定落地。
- **worker 的下一步**：确认 gradmotion 训练侧动作语义（position target+PD 还是 torque）；决定 MuJoCo 验证 xml 用 motor+外环 PD 还是生成 position actuator 版本。

### I6 · 上游数据缺口：补物理自洽门与慢跑源（二期数据工作）
- **关联结果版本**：无
- **文件/函数**：`x1_gmr_retargeted_tool/validate_retarget_v3.py`（现有门 R7/R8/R9 定义于 L1–48）；`build_dataset_v3.py` L13–19（SEGS 含 9 段 sprint，实际仅 7 段 run 入库）
- **观测事实**：现有 9 个门全是**几何/运动学**门（平底、穿模、髋抖动），无任何**动力学/步态物理**门；README 自述 sprint 物理不自洽且建议「重训换 1.2–2.2 m/s 慢跑源并补 duty/腾空物理门」（L44–45）；H2O/PULSE 证据：不可行片段混入训练显著有害。
- **机制假设**：duty factor 与腾空时长是检测「慢放烘焙」与「物理不自洽」最灵敏的指标（人形奔跑 duty 典型 20–35%，腾空期 ~0.1–0.3 s；sprint 族腾空至 1.73×（README 数据）远超物理可能）。
- **竞争解释**：若采用 AMP 范式（I2 竞争路径），风格分布对个别不自洽片段更宽容，数据门可放宽——但 run 族 7 段体量下 AMP 判别器易过拟合，跟踪式 + 严格门仍是首选。
- **最小改动或诊断实验**：在 `validate_retarget_v3.py` 增设 R10(duty)/R11(腾空时长) 两门的文本 diff（我下期报告提供，由 worker 决定是否落地）；数据侧二期引入 LAFAN 慢跑/走跑过渡主题补 1.2–2.2 m/s 带宽。
- **对照变量**：加/不加物理门的数据集各训一次（等预算）。
- **预期指标**：训练集含/不含不自恰片段的策略跟踪成功率与步态物理量（duty、腾空、GRF 形态）。
- **支持/否定条件**：支持=过滤后策略物理步态量显著接近人形标准且不掉跟踪率；否定=差异不显著（则数据门优先级降低，专注 I1/I5）。
- **成本风险**：数据重跑需服务器端 G1 csv（README L19 注明未入库），依赖外部资源；一期先用现有 7 段。
- **worker 的下一步**：先不阻塞于数据；E1 结果出来再决定是否申请补数据。

---

## 最小验证实验

| ID | 实验 | 依赖 | 预算 | 判据 |
|---|---|---|---|---|
| E1 | 7 段 pkl 物理量统计：root 速度分布（两种 time_scale 口径）、duty、腾空时长、步频、脚高；对照 humanoid_pose_standard skill 的指标与人形奔跑常模 | 本地脚本（无 numpy 环境，需 worker 在有环境机器跑或我用纯 python 重写） | 分钟级 | 得到每段物理量表；判定 run 族是否全部物理自洽 |
| E2 | 最小 Gaussian PPO 跟踪基线：单段 clip（建议 run1_subject2_seg0，time_scale 最低 1.06）+ H2O 奖励表 + one-frame-ahead goal + RSI on/off 对照 | gradmotion 平台 | 1–2 轮短训 | 跟踪成功率>0 即管线闭环；RSI 效应复现 |
| E3 | DPPO 式扩散头替换：DDIM ft=5 步、chunk Ta=4、MLP 头，其余同 E2 | E2 通过 | 1–2 轮短训 | 等预算收敛速度/平滑度/跟踪率对比 Gaussian |
| E4 | MuJoCo sim2sim 协议：PD 增益扫描 × 延迟 0/2/5/10 ms × 控制率 50/100 Hz；验收=速度跟踪误差与 50 m 连续奔跑 | E2/E3 策略导出；MuJoCo xml 执行器方案（I5） | 每策略数百次 rollout | 延迟敏感性曲线；确定部署预算 |

顺序建议：E1 →（E2 ∥ I5 执行器方案决策）→ E3 → E4。E1 与 I5 无相互依赖可并行。

---

## 上下游缺口

1. **训练管线缺位**：仓库无任何 RL 训练代码（仅数据工具链）；「基于扩散模型的强化学习」的具体平台实现（gradmotion 侧框架、动作头、obs 设计）尚未定型——E2 是一切建议的落地前提。
2. **执行器语义断层**（I5）：URDF/MJCF 是裸 motor，训练平台若为 PD position 惯例，缺一份 position-actuator MJCF 或 MuJoCo 侧 PD 环实现。
3. **物理自洽数据门缺失**（I6）：9 个几何门 → 0 个动力学门；且 G1 源 csv 未入库，数据重做依赖服务器端资源。
4. **sprint 数据未入库**：SEGS 列 9 段 sprint，实际 7 段 run 入库；高速带宽（>2.2 m/s）目前无自洽数据，「奔跑」目标的可达速度上限受数据约束。
5. **诊断工具链断裂**：本地环境无 numpy（我实测 `import numpy` 失败），意味着本地连 E1 都跑不了，快速诊断只能靠 worker 环境或纯 python 重写——建议 worker 在 dm-results 中固定输出此类统计表。

---

## 反证与不确定性

- **[未验证] "DiP/AgileButDim"**：一手来源检索为负（[R1] §B 有完整负证据链）。任何引用该工作数字的说法在拿到一手 PDF 前应视为传闻。
- **[推断] 慢放有害程度**：README 的 sprint 自不洽数字是仓库自述，我未复算；run 族（time_scale 1.06–1.26）是否在安全范围待 E1 判定。若 E1 全绿，I2 优先级下调。
- **[推断] DPPO 在 29 dof 全身跟踪上的收益**：DPPO 论文的 locomotion 实验是 3–6 dof Gym 任务 + 操作任务，无人形全身跟踪实验；其收益外推到本项目依赖 PDP（人形，但为 BC 蒸馏）与 GenPO/TruDi（人形 locomotion，但方法变体）的间接证据。E3 可能给出「无显著差异」的否定结果——这本身有价值（收敛到 PDP 式或 latent 路线）。
- **[未验证] AMP 论文原始超参与 PHC 精确权重**（[R2] 标注）：若走 AMP 路线需从 IsaacGymEnvs 配置（已核实）而非二手转述取参。
- **[未验证] gradmotion 平台能力边界**：我未访问该平台；训练侧动作语义、并行规模、是否支持自定义扩散头均待 worker 确认——I1/E3 的可行性最终受此约束。
- **[推断] ctrlrange ±180 的单位问题**：裸 motor ctrlrange=-180~180 若为 Nm 则远超人形关节力矩合理值（峰值 ~100–200 N·m 仅髋/膝，腕部远小），疑为占位值；若 sim2sim 直接用此文件，力矩限幅语义需先澄清。

---

## 来源

**本地（commit `fc0b9be`）**
- `x1_gmr_retargeted_tool/README.md` L1–51（链路、遗留问题、v1→v3 演进）
- `x1_gmr_retargeted_tool/validate_retarget_v3.py` L1–48（R7/R8/R9 门定义与阈值）
- `x1_gmr_retargeted_tool/build_dataset_v3.py` L13–19（SEGS 清单与权重规则）
- `x1_retargeted_motion/*.pkl`（7 段；实测字段 loop_mode/fps/frames/time_scale/s_leg/s_arm/version）
- `X1_29DOF/mjcf/robot/xyber_x1/xyber_x1_serial.xml` L5（timestep 0.001）、L252–256（motor 执行器）、L284–354（传感器）

**文献调研（归档于 `research/studio/idear/`）**
- [R1] `20261003_diffusion-policy-humanoid-running-sota.md`：DPPO（arXiv:2409.00588）、BeyondMimic（arXiv:2508.08241）、GenPO（arXiv:2505.18763）、TruDi（arXiv:2606.15260）、PredActor（arXiv:2609.24840）、FastStair（arXiv:2601.10365）等 18 源
- [R2] `20261003_retargeted-mocap-consumption-amp-vs-tracking.md`：DeepMimic/AMP/PHC/PULSE/UHC/H2O/OmniH2O/HumanPlus/EWC/PDP/UniPhys/CLoSD/SONIC 等 14 源，含 H2O 奖励表与 IsaacGymEnvs AMP 配置的核实数字

**缺失证据**：无本项目实验数据；无 gradmotion 平台一手信息；"DiP/AgileButDim" 一手来源未获得。
