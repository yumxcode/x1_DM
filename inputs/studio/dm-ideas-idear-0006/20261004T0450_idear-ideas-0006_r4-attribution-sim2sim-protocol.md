# idear 报告 0006：r4 解冻的归因分析、sim2sim 增补与 I6 数据门 diff 交付

- 报告 ID：idear-ideas-0006
- 日期：2026-10-04
- 作者：idear（研究与建议角色）
- 关联结果版本：**dm-results-R002-r4-sim2sim-align v1**（artifactId `dm-results-R002-r4-sim2sim-align`，digest sha256:343a0eb9…，实验 ID TASK_20261004_005，分支 dm/smp-reward-fix @ 26c8326）；承接 idear-ideas-0004/0005
- 新增证据：r4 中期日志数字的数学解读（本报告§论文与实现证据）；R002 的 I5 落地细节（30Hz/120Hz、kp/kd 全表、tan_norm 列序、ctrl 量纲修复）
- 状态：建议（非验收）。**含对 idear-ideas-0004 的诚实修正（I18 节）与 R002 邀请的 I6 diff 交付（I22 节）**

---

## 摘要与关联结果

R002 报告 r4 解冻（Smp_Reward 0.18→0.37 单调升、Sds_Loss 0.94→0.45、Clip_Frac 0.40–0.60 vs r3 冻结态 0.017），处置表确认 I5 已落地（执行器/obs 契约对齐质量高，含两个真 bug 修复）。本报告四项内容：

1. **归因数学**：r3 与 r4 起始的 normalized SDS loss 几乎相同（0.968 vs 0.857，由奖励值反解）——证实 0004「0.003 是工作点数值」；但同一工作点上 scale=2 的奖励灵敏度 |dr/dL| 是 scale=6 的 **20×**（0.36 vs 0.018）——**worker 的「饱和」判断在梯度灵敏度意义上有实质内容，0004 在此点需修正**。r4 解冻 = 三变量联合作用（scale 灵敏度 + AdamW 3e-4 + task_reward 0.5），单变量归因未闭合；给出零成本判别法（I18）与停滞决策树（I19）。
2. **观测缺口**：`smp_agent.py` L162–183 混合奖励后 `info` 只记录 `smp_reward_mean/std`，**task 分量无独立日志**——r4 的归因判别因此受限，建议下次 run 补一行记录（I20，改动一行）。
3. **sim2sim 增补**：rollout 初始状态协议必须用 motion 帧（RSI 风格），home pose 静态不可平衡（R002 观测 pitch→94° 前倾）会污染评估；E4 参数按 R002 实测锁定；friction 差异是 R002 已知未对齐项，列入 E4 扫描（I21）。
4. **I6 diff 交付**：R002 处置表明确邀请「idear 若提供 diff，二期 review」——R10(duty)/R11(腾空) 两门文本 diff 见 I22，由 worker 决定落地时机。

---

## 论文与实现证据

**A. r3/r4 工作点的数学解读（已观测数字的反解，本报告计算）**

| 量 | r3（冻结） | r4 起点 | r4@iter572 |
|---|---|---|---|
| Smp_Reward_Mean（R001/R002） | 0.003 | 0.18 | 0.37 |
| sds_loss_scale | 6 | 2 | 2 |
| 反解 normalized loss L=−ln(r)/scale | **0.968** | **0.857** | **0.497** |
| 奖励灵敏度 \|dr/dL\|=scale·exp(−scale·L) | 0.018 | **0.360（20×）** | 0.740 |

三条推论：
- **(i)** r3/r4 起始 L≈1：两次 run 的初始策略离流形距离一致，r3 的 0.003 是 exp(−6×0.97) 的设计工作点数值（0004 论点在数值正确性上成立）。
- **(ii)** 但工作点处梯度灵敏度差 20×：scale=6 在 L≈1 处的奖励对 loss 变化几乎不响应（0.018），advantage 中的 SMP 分量信噪比极低——**worker 的修复方向获得机制支撑，0004「调 scale 无用」的表述过强，特此修正**。
- **(iii)** r4 中 L 0.857→0.497：策略状态确实逼近先验流形（running mean 为累计加权，主导项是 loss 下降）。**但注意归因混淆**：task_reward 0.5（DeepMimic 跟踪）驱动策略模仿数据 → 状态入流形 → SMP 奖励被动上升，与「scale/AdamW 恢复了 SMP 奖励引导力」两种机制在当前曲线上不可区分——需要 I18 判别。

**B. R002 的 I5 落地证据（已观测，worker 报告§3）**
- 训练侧：`isaac_lab_engine.yaml` control_mode=pos、**控制 30Hz / sim 120Hz**（非 0002 假设的 50/100Hz）；`isaac_lab_engine.py:908-945` pos→ImplicitActuatorCfg（stiffness/damping 从 x1.xml joint 解析 :871-905，effort_limit=motor gear）。
- kp/kd 全表（x1.xml）：腰 375/37.5、髋 pitch/roll 450/45、膝 450/45、踝 200/20、臂 50/5、腕 25/2.5；gear 力矩限幅：腰 150/180、髋 180/150、膝 180、踝 80、臂 20、腕 10；armature 0.01–0.02。
- obs 228 维 = [root_h(1), root_rot tan6, root_vel, root_ang_vel, joint_rot tan6×29, dof_vel×29, key_pos 相对×4]（char_env.py:412-441）；**tan_norm=[R·x, R·z]（第 1、3 列，非 Zhou 第 1、2 列）**——worker 已在 sim2sim 脚本修正（0002-I8 的契约清单命中实例）。
- MuJoCo 侧：x1_train.xml 去 joint stiffness、保 damping 走隐式积分（等价 IsaacLab implicit damping）；外环 tau=kp(q*−q)，ctrl=clip(tau/gear,±1)。两个真 bug 被修复：ctrl 量纲（±1×gear）、显式 kd·qdot 对腕/肘低 armature 关节发散（kd·dt/armature≈4.2>2）。
- 参考动作 PD 回放（3s）：跌倒前腿误差 0.12–0.19 rad、臂 0.05–0.14 rad，无单关节错位 → 执行器/关节序/gear 语义 PASS。

**C. 与文献的交叉印证（[R4] SMP 配方）**
- SMP 官方任务配置用 task 0.5 / smp 0.5 + **GSI True**；r4 是 task 0.5 / smp **1.0** + GSI False——smp 权重高于论文任务模板，且无 GSI（依赖 RSI 从 7 段数据初始化）。若 r4 后期停滞，GSI 是论文内生的候选修复（见 I19）。
- 论文「高噪声-only 引导 → 行为平均化」：r4 的 K=[22,15,8] 固定不变，无新风险。

---

## 优先建议

### I18 · r4 归因判别（零成本日志法 + 诚实修正 0004）
- **关联结果版本**：dm-results-R002-r4-sim2sim-align v1 §2
- **文件/函数**：`mimickit/learning/smp_agent.py` L162–183；r4 训练日志（/tmp/r4log3.json 同源）
- **观测事实**：见§论文与实现证据表——r3/r4 起始 L 一致（0.968/0.857），灵敏度差 20×，L 已降至 0.497。
- **机制假设**：解冻来自三变量之一或组合：①scale 灵敏度（20×）；②AdamW 3e-4 自适应放大；③task_reward 0.5 提供独立优势信号（策略模仿数据→入流形→SMP 奖励搭便车）。
- **竞争解释**：若③主导，则「SMP 奖励单独不可引导」仍成立（0004 H-A'/H-B 弱化版），r4 本质是「DeepMimic 跟踪训练 + SMP 风格正则」；若①②主导，SMP 奖励自身在工作，H-C 全量成立。
- **最小改动或诊断实验**（无需新训练）：
  1. **曲线相关性**：把 r4 的 Smp_Reward_Mean 与 Task 侧代理曲线（若有：ep_len、tracking 相关日志）对时间轴做相关——同步上升支持③；Smp 独立上升且 ep_len 不变支持①②。
  2. **斜率分段**：若 Smp_Reward 的上升斜率在 task 分量已饱和后（假设 ep_len 达到 clip 长度上限）仍维持，说明 SMP 奖励仍在提供梯度（①②有效）。
- **对照变量**：iter 轴分段（0–100 vs 300–572）。
- **预期指标**：相关系数 >0.8 或 <0.3 的清晰判别；模糊区间（0.3–0.8）则等 checkpoint 后用仿真跟踪误差对照。
- **支持/否定条件**：③支持=Smp_Reward 与跟踪代理强相关且起始延迟（task 先动）；①②支持=起始即同步上升（exp 核灵敏度恢复即生效）。
- **成本风险**：零（日志分析）。
- **worker 的下一步**：下载中间 checkpoint 前先做此 10 分钟分析；结果记入 R003。

### I19 · r4 后期停滞决策树（预防性）
- **关联结果版本**：R002 §4 计划 1
- **文件/函数**：`data/agents/smp_x1_agent.yaml`（dm/smp-reward-fix @ 26c8326）
- **观测事实**：r4 目标 Smp_Reward>0.5（R002 §4.1）；当前 0.37@iter572；论文单 clip 30min@4090 量级、r4 预算 ~6h/461M samples。
- **机制假设**：三种停滞形态对应不同修复，预先定义避免临场乱调：
  - **停滞 A**（Smp_Reward 卡在 0.4–0.5，ep_len 正常增长）：流形内剩余 loss 来自滑移分量（H-B 弱化版）——SMP 奖励的理论上限不是 1 而是物理可达流形的覆盖度。修复：接受该平台，靠 task 分量完成跟踪，步态质量交给 R003 九类指标判定；若滑移指标差，启动 I16 数据增广（v3.1 重投影）。
  - **停滞 B**（Smp_Reward 平台且 ep_len 也不增长）：梯度仍不足或探索枯竭。修复优先序（按侵入度）：GSI True（论文任务模板内生的初始化增广，先验已有 sample 接口，smp_agent.py L230–257 现成）→ K 扩高 t（如 [22,15,8]→[30,22,15,8]，高噪声更平滑）→ 回到 0004-E8 四格消融（含 scale 6 复格，验证 20× 灵敏度分析）。
  - **停滞 C**（发散/震荡，Clip_Frac>0.7）：AdamW 3e-4 过大或 reward 尺度漂移。修复：lr 退火或 norm_adv_clip 4.0→2.0。
- **竞争解释**：无停滞则本条作废——决策树只在触发时使用。
- **最小改动或诊断实验**：无（决策规则）。
- **对照变量**：停滞形态 A/B/C 三分支。
- **预期指标**：任一分支触发时修复动作 ≤2 个变量。
- **支持/否定条件**：见各分支内置判据。
- **成本风险**：低（预防性规则，避免临场三变量同改重蹈归因困难）。
- **worker 的下一步**：timer park 回访时按形态对号。

### I20 · 补 task 分量日志（一行改动，下次 run 生效）
- **关联结果版本**：R002 §2「Train_Return 仍显示 0（口径未明）」
- **文件/函数**：`mimickit/learning/smp_agent.py` L162–183 `_compute_rewards`：`info` 仅含 `smp_reward_mean/std`；task_r 在 L163 取出后于 L174 混合，无独立记录。
- **观测事实**：R002 无法报告 task 分量曲线 → I18 判别受限；Train_Return=0 的口径困惑（test episode 全程未成功记 0？）也因缺中间量。
- **机制假设**：补 `info["task_reward_mean"]=torch.mean(task_r)`（一行）后，r5+ 可直接做 I18 相关性分析，且 Train_Return 口径问题可通过 task_reward 曲线交叉定位。
- **竞争解释**：无。
- **最小改动或诊断实验**：文本 diff（worker 决定落地）：
  ```diff
  --- a/mimickit/learning/smp_agent.py
  +++ b/mimickit/learning/smp_agent.py
  @@ def _compute_rewards(self):
           task_r = self._exp_buffer.get_data_flat("reward")
           disc_obs = self._exp_buffer.get_data_flat("disc_obs")
  @@        info = {
  +           "task_reward_mean": torch.mean(task_r),
               "smp_reward_mean": smp_reward_mean,
               "smp_reward_std": smp_reward_std
           }
  ```
- **对照变量**：无。
- **预期指标**：r5+ 日志出现 Task_Reward_Mean 曲线。
- **支持/否定条件**：—（工程项）。
- **成本风险**：零风险（只增观测不改行为）。
- **worker 的下一步**：与任何下次配置改动合并提交。

### I21 · sim2sim rollout 初始状态协议 + E4 参数锁定（对 0002-I7/I8 的实例化更新）
- **关联结果版本**：R002 §3（home pose PD hold 失败：pitch→94° 前倾翻倒）
- **文件/函数**：`analysis/sim2sim_validate.py`（X1Sim）；`data/assets/x1/x1.xml`
- **观测事实**：X1 home pose 静态不可平衡（重心偏前）；训练侧从不静站（RSI 从 motion 帧初始化）；R002 未验证 friction（我方 1.0/0.05/0.05 vs IsaacLab USD 默认可能 1.0/0.005/0.0001）。
- **机制假设**：sim2sim 若用 home pose 初始化，前 0.5 s 的「扶正瞬态」会污染九类指标（尤其步频/duty/腾空），且失败可能被误判为策略缺陷；策略 obs 含 root_h 与速度，home pose 下速度=0 偏离训练分布（30Hz 控制域内 RSI 帧速度非零）。
- **竞争解释**：若一期验收包含「从静止启动」，则 home pose 瞬态是真实需求而非污染——但训练分布不支持，应显式列为已知限制而非让指标背锅。
- **最小改动或诊断实验**：E4 协议锁定如下——
  1. 初始化：从 7 段数据随机帧（RSI 语义，含帧速度）；
  2. 控制率 30Hz / 物理 120Hz（=训练值，R002 实测）；PD 用 x1.xml kp/kd 表；ctrl=clip(tau/gear,±1)；
  3. 扫描：--pd-scale 0.5/1/2 × --latency-ms 0/2/5/10 × **friction 0.6/0.8/1.0/1.2**（新增 friction 轴，R002 已知未对齐项）；
  4. 指标：九类 + 首触地滑移（数据滑移 25–35% 的对照点）；
  5. 判据：速度跟踪误差、50m 连续性（0001-E4 口径）。
- **对照变量**：初始化（RSI 帧 vs home pose）各跑一次对照，量化瞬态污染幅度（也是可发布证据）。
- **预期指标**：RSI 帧初始化下指标显著稳定；home pose 对照量化瞬态影响。
- **支持/否定条件**：若 RSI 帧初始化下 sim2sim 仍系统性失败 → 执行器之外的对齐问题（friction/接触）→ 按 0002-I9 开环重放定位。
- **成本风险**：低（协议项，无新代码）。
- **worker 的下一步**：R003 按此协议执行；friction 轴优先于延迟轴（已知未对齐项）。

### I22 · I6 数据门 diff 交付（R10 duty / R11 腾空，R002 邀请项）
- **关联结果版本**：R002 处置表 I6「idear 若提供 diff，二期 review」；R001 §3 已有逐段 duty/腾空实测
- **文件/函数**：`tools/x1_pipeline/validate_retarget_v3.py`（正本；本工作区归集副本 `x1_gmr_retargeted_tool/validate_retarget_v3.py`，R7-R9 定义于 L1–48，STANCE_Z=0.008 L31）
- **观测事实**：R001 实测 7 段 duty 0.29–0.45、腾空比 13–34%（全部自洽），但 sprint 族（未入库）README 自述 duty 18–20%@1m/s 不自洽——门在二期数据扩充时是第一道防线。
- **机制假设**：duty 与腾空比是「慢放烘焙/物理不自洽」最灵敏的检测器（skill ⑤ 一票否决项）；以 R001 已验证的实测分布设阈值，误杀率低。
- **竞争解释**：阈值过紧会误杀风格变化段；用 R001 实测分布 + 人形常模放宽带双向锚定。
- **最小改动或诊断实验**：文本 diff（worker 决定落地时机）：
  ```diff
  --- a/tools/x1_pipeline/validate_retarget_v3.py
  +++ b/tools/x1_pipeline/validate_retarget_v3.py
  @@ R7/R8/R9 之后新增（复用 R7 的 sole zmin 触地判定）：
  +  R10 duty-band ("占空比区制"): 触地帧占比（按 R001 口径：sole zmin < STANCE_Z）
  +     每脚 duty ∈ [0.25, 0.55] 判 PASS（R001 实测 0.29-0.45 + 余量；
  +     人类行走 duty>0.55（无双支撑消失）、奔跑 duty<0.25（腾空主导），
  +     X1 腿长换算后慢跑带宽此带）
  +  R11 flight-ratio ("腾空占比"): 全帧中双脚均离地的占比 ∈ [0.08, 0.40]
  +     判 PASS（R001 实测 0.13-0.34；>0.40 疑似慢放烘焙或腾空物理不自洽，
  +     <0.08 疑似行走混入）
  +  两门均要求与 Froude 联合判读：Fr=v^2/(g*L_leg)，L_leg 用 X1 实际腿长
  +  （hip 至 sole，URDF 提取常量化），Fr>0.5 且 duty<0.30 时允许腾空比上探 0.45
  ```
- **对照变量**：门开/关各跑一次数据集构建（二期）。
- **预期指标**：sprint 族复入库时 R10/R11 拦截不自洽段（README 自述的 duty 18-20%@1m/s 将被 R10 下界拦截）。
- **支持/否定条件**：支持=二期数据用门过滤后训练稳定性提升；否定=无差异则门降级为报告项。
- **成本风险**：低（纯数据侧，阈值可调）。
- **worker 的下一步**：二期数据扩充前落地；一期不阻塞。

---

## 最小验证实验

| ID | 内容 | 依赖 | 预算 | 判据 |
|---|---|---|---|---|
| E9（新） | I18 归因判别：r4 日志曲线相关性/斜率分段分析 | r4 日志（已有） | 分钟级 | ③ vs ①② 判别 |
| E4（更新） | sim2sim 协议按 I21 锁定：RSI 帧初始化 + 30/120Hz + pd-scale×latency×**friction** 三轴扫描 + 初始化对照 | r4 checkpoint | 数百 rollout | 九类指标 + 速度跟踪 + 50m |
| — | I20 task 分量日志一行 diff / I22 R10-R11 门 diff | worker 落地 | — | — |

执行顺序：E9（趁 checkpoint 未到）→ E4（checkpoint 到手后）。

---

## 上下游缺口

1. **task 分量无日志**（I20）：当前无法观测混合奖励中跟踪项的贡献——r4 归因与 R003 步态质量解读的共同前提。
2. **friction 未对齐**（I21）：IsaacLab USD 转换后的确切 friction 值待确认（R002 已列未验证）；E4 加 friction 轴对冲。
3. **Train_Return=0 口径未明**（R002 §2 未验证项）：若 test episode 需全程成功才计值，则 r4 后期应出现非零；持续为 0 需查 logger 实现。
4. **GSI 通道未用**：smp_agent.py L230–257 现成 GSI 接口（I19 停滞 B 的首选修复），但 enable_gsi=False——若启用需先验证 x1_prior 的 sample 质量已在 E6b 范围（worker 未跑 E6b；若停滞 B 触发，GSI 启用前先做 0004-E6b 的基线层）。

---

## 反证与不确定性

- **[已观测+计算] r3/r4 起始 L 一致（0.968/0.857）与灵敏度差 20×**：本报告由 R001/R002 发布数字反解；依赖 DiffNormalizer 语义（x/mean|x|，0004 已核实）与「running mean 未被早期样本大幅拉偏」的假设——若 r4 前几 iter normalizer 统计剧烈变化，反解值有小误差，但量级结论稳。
- **[对 0004 的修正]**：0004 称「调 scale 无用」表述过强——正确表述是「0.003 数值本身是设计工作点、不是异常；但 scale=6 在该工作点的奖励灵敏度仅为 scale=2 的 1/20，作为修复手段有机制支撑」。0004 的 H-C 完全否定修改为部分否定：scale/优化器可能是解冻的实质贡献者（与 task_reward 并列的候选）。
- **[推断] 停滞 A 的 SMP 奖励理论上限<1**：滑移分量物理不可达 → normalized loss 有下限 → 奖励平台。此推断若成立，r4 停在 0.4–0.5 不是失败而是结构特性——需 I18 曲线 + R003 步态滑移指标共同判定。
- **[推断] home pose 重心偏前**：R002 的 PD hold 前倾翻倒可能源于 PD 增益不足而非质心几何——两种解释都支持「sim2sim 用 RSI 帧初始化」，但若真机部署需要静止启动，需另行训练站立技能（超出本期）。
- **[未验证] r4 的 smp 权重 1.0 vs 论文任务模板 0.5**：smp 分量占混合奖励的 2/3（1.0 vs 0.5），若 SMP 奖励平坦（停滞 A），混合奖励退化为 task 单独驱动——权重敏感度未测，E8 消融可覆盖。
- **[未验证] IsaacLab ImplicitActuator 与 MuJoCo 隐式 damping 的数值等效性**：R002 的回放 PASS（3s/0.12–0.19 rad）是行为级证据；更长时程/接触丰富段的等效性由 E4 开环重放（0002-I9）补。

---

## 来源

**共享产物（已 fetch 快照）**
- dm-results-R002-r4-sim2sim-align v1（`inputs/studio/dm-results-R002-r4-sim2sim-align/`，sha256:343a0eb9…）
- dm-results-R001-baseline v1（前轮）

**x1_mimicKit（dm/smp-reward-fix @ 26c8326，经 R002 引用；主干核实 @ 88dbd89e）**
- `mimickit/learning/smp_agent.py` L162–183（info 字段）、L230–257（GSI 接口）
- `data/agents/smp_x1_agent.yaml`（0004 归档件；r4 改动经 R002 §2 描述）
- R002 新披露：`isaac_lab_engine.py:871-945`、`char_env.py:412-441`、`x1.xml`（kp/kd/gear/armature）、`analysis/sim2sim_validate.py`（均 worker 侧证据文件，未在我方归档范围）

**文献（归档）**
- [R4] `research/studio/idear/20261004_sds-as-reward-smp-tadpole-smiling.md` §A（SMP 任务模板 0.5/0.5+GSI、调参优先级）

**计算**
- 本报告 §论文与实现证据 A 表（python 反解，可复现：`-ln(r)/scale`）

**缺失证据**：r4 的 task 分量曲线（I20）；IsaacLab friction 确切值；r4 checkpoint 实测；E6b 先验判别力测试（worker 未执行，I19 停滞 B 触发时前置）。
