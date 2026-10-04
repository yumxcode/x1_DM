# idear 报告 0002：Isaac→MuJoCo sim2sim 对齐协议与 E1 指标表标准化（idear-ideas-0001 的 I5/E1/E4 深化）

- 报告 ID：idear-ideas-0002
- 日期：2026-10-04
- 作者：idear（研究与建议角色）
- 关联结果版本：承接 idear-ideas-0001（artifactId `dm-ideas-idear-0001`，released）的 I5（执行器语义断层）、E1（数据物理量诊断）、E4（sim2sim 协议）；dm_worker 仍无实验产物
- 新增证据：12 源 sim2sim 对齐调研（含 3 个读源代码级核实的开源管线）+ 项目内 `humanoid_pose_standard` 姿态标准指标框架对齐
- 状态：建议（非验收）

---

## 摘要与关联结果

本报告回答两个问题：(1) X1 的 MuJoCo 验证侧到底怎么搭（执行器/接触/观测契约的对齐协议）；(2) E1 的数据物理量统计表用什么指标体系。核心结论：

1. **X1 仓库 MJCF（裸 motor、无 PD）必须改造才能做验证**，社区两条成熟路线：(a) MuJoCo 力矩模式 + 外环 1 kHz PD（Humanoid-Gym 原型，G1/H1 社区事实标准）；(b) 原生 position actuator + joint damping 充当 kv（TienKung 半马冠军方案）。给出两路线的完整参考参数与选择判据。
2. **sim2sim 不是走过场**：文献定量证据显示单引擎（Isaac 系）训练的策略零样本进 MuJoCo 成功率仅 3.6–50%（PolySim，G1 全身跟踪）。训练侧应把 MuJoCo 差异当真实的鲁棒性过滤器来设计（接触 DR、延迟 DR、多引擎可后置）。
3. **E1 指标表应对齐项目 `humanoid_pose_standard`**：其中⑤「对地时间结构（duty×Froude 区制、腾空、双支撑期）」为**一票否决级**——正好覆盖 time_scale 慢放诊断（I2）与数据门缺口（I6）。
4. 新增建议 I7（sim2sim 验证侧三件套落地方案）、I8（部署前校准检查：正弦跟踪 + 相图）、I9（开环重放发散作为 gap 定位工具）；修订 E4 为带摩擦扫描与失败阈值的完整协议；E1 指标表细化为可直接实现的 14 项。

---

## 论文与实现证据

> 完整调研归档：`research/studio/idear/20261004_isaac-to-mujoco-sim2sim-alignment.md`（下称 [R3]，12 源）

**执行器对齐的两种已验证模式（读源码级核实）**
- **模式 A：外环 PD（Humanoid-Gym，arXiv 2404.05695，~2.1k★，XBot-L 1.65 m）**：MuJoCo 力矩模式，每个物理步（1 ms）计算 `tau = kp*(q*-q) + kd*(0-dq)`，力矩钳位 ±tau_limit 后写 `data.ctrl`；策略 100 Hz（decimation=10）。增益参考（XBot-L 12 关节）：kps=[200,200,350,350,15,15]×2（hip pitch/roll 200、knee 350、ankle 15），kds=10，tau_limit=200 N·m。G1-walk 管线同构（1 ms 物理 / 20 ms 策略 / PD 每物理步重算）。
- **模式 B：原生位置执行器（TienKung-Lab，Open-X-Humanoid，~900★，人形半马冠军）**：MJCF 内 per-joint kp（hip 700、hip-yaw 500、knee 700、ankle-pitch 30、ankle-roll 16.8、shoulder 60/20/10、elbow 10）+ joint damping（腿 10/10/5/10/2.5/1.4）充当 kv；策略 50 Hz（dt=0.005×decimation=4），action_scale=0.25，`data.ctrl = action*0.25 + default_qpos`；solver PGS + integrator implicitfast；地板 friction='1 0.005 0.0001'。
- **Isaac 隐式执行器与两者都不逐位等价**：ASAP（arXiv 2502.01143）显式 SysID **per-joint PD 增益比 (k_p^i, k_d^i)** + CoM 偏移 + 质量比来吸收差异。

**接触求解器差异（定量）**
- UCL B1+Z1（arXiv 2512.18938）：Isaac/PhysX 刚性校正冲量 → yaw 振荡（v_yaw RMSE 至 0.30 rad/s）；MuJoCo 默认 `impratio=1` → 平滑但站立**脚滑**；`impratio=100` 几乎消除滑动但作者警告这是「人为压高切向阻抗而非真实鲁棒性」。
- 跨引擎 gap（PolySim，arXiv 2510.01708，G1 全身跟踪、14 个 ASAP 动作）：单引擎训练零样本进 MuJoCo 成功率 IsaacSim 3.6% / Genesis 12.1% / IsaacGym 50.0%；三引擎并行训练 56.4%；**参数级 DR（ASAP 配方）在单引擎内只有 10%**——引擎级差异不是参数 DR 能覆盖的。

**观测契约静默失败清单（社区管线最常见的 bug 源）**
- 关节顺序 permutation 表（TienKung 明确导出 `mujoco_to_isaac_idx`/`isaac_to_mujoco_idx`）；四元数约定（MuJoCo `framequat` wxyz vs 常用 xyzw）；Euler wrap 到 [−π,π]；obs 缩放与 ±100 钳位；历史长度；步态相位信号（周期、air ratio）必须逐位复刻（TienKung walk 0.85 s 周期/air 0.38，run 0.5 s/air 0.6）。

**部署前校准与 gap 定位**
- Humanoid-Gym 校准法：MuJoCo vs 真机的**关节正弦跟踪曲线 + 5 s 相图**（膝/踝 pitch @0.5 m/s）——策略 rollout 之前先验证执行器对齐。
- ASAP 开环重放诊断：把训练引擎的动作序列在目标引擎重放，全局 MPJPE 随 0.25/0.5/1.0 s 发散 19.5→33.3→80.8 mm 即为动力学 gap 的直接度量；其 delta-action 修正把 1.0 s 误差压到 37.9 mm。
- 训练侧可存活性 DR（Humanoid-Gym Table III）：关节位置噪声 N(0,0.05 rad)、关节速度 ±0.5 rad/s、角速度 ±0.1、Euler ±0.03、系统延迟 U[0,10] ms、摩擦 U[0.1,2.0]、电机强度 95–105%、负载 ±5 kg。

**X1 现状对照（commit `fc0b9be`）**
- `X1_29DOF/mjcf/robot/xyber_x1/xyber_x1_serial.xml`：L5 `timestep=0.001`；L252– `<motor>` 全裸力矩（如 L253 `motor_left_hip_pitch ctrlrange="-180 180"`），无 kp/kv、无 damping 语义；L284–354 jointpos/jointvel 传感器齐全。
- 工具链已有 `X1_DOF_ORDER` 常量（`retarget_g1_x1.py` 定义、`retarget_v3.py` 复用）——观测契约 permutation 表有现成锚点。

---

## 优先建议

### I7 · sim2sim 验证侧落地：优先模式 A（外环 1 kHz PD），次选模式 B（生成 position-actuator MJCF）
- **关联结果版本**：idear-ideas-0001 / I5（执行器断层观测）
- **文件/函数**：`X1_29DOF/mjcf/robot/xyber_x1/xyber_x1_serial.xml` L252–283（motor 块）；验证脚本待 worker 新建（建议 `tools/sim2sim/` 或对应正本路径）
- **观测事实**：MJCF 为裸 motor + ctrlrange ±180/±150（单位可疑）；文献两模式参数齐备（[R3] §1–2）；Humanoid-Gym/G1-walk 的模式 A 是 G1/H1 社区事实标准且与「训练侧 PD position target」语义最近。
- **机制假设**：gradmotion（Isaac 系）训练大概率用 PD position target；模式 A 在验证侧显式复刻「目标位置 → kp/kd 力矩 → 钳位」全链路，与训练语义一一对应，gap 最小且可逐项排查（kp/kd/tau_limit/频率都有独立旋钮）；模式 B 把 kv 藏进 joint damping，与 Isaac 隐式执行器的等效性更难审计。
- **竞争解释**：若训练侧直接输出力矩（少见但存在），模式 A 退化为直写 `data.ctrl`，反而更简单；模式 B 的优势是 MJCF 自洽、不依赖 Python 外环速率稳定性。
- **最小改动或诊断实验**：写一个 ~80 行的 MuJoCo 力矩模式 runner：`tau = clip(kp*(a*scale+q0−q) − kd*dq, ±tau_lim)` 每 1 ms 步、策略每 decimation 步更新一次；kp/kd 初值用训练侧导出值（若有），否则参考 TienKung 比例（hip/knee 500–700、ankle 17–30）。
- **对照变量**：模式 A vs 模式 B（同一策略、同一指标）；PD ±2× 扫描。
- **预期指标**：同一 Isaac 导出策略在两种模式下的速度跟踪 RMSE 与成功率差异 <5%（若差异大，说明模式实现有 bug 而非物理差异）。
- **支持/否定条件**：支持=模式 A 复刻训练指标趋势；否定=两模式都无法维持步态 → 转向 I8 校准检查定位（大概率观测契约或接触参数问题）。
- **成本风险**：低（纯验证侧代码，不动训练）；风险是 ctrlrange 单位（±180 若为 Nm 则异常大，钳位形同虚设——先澄清）。
- **worker 的下一步**：确认 gradmotion 训练动作语义与 PD 参数导出方式；按模式 A 搭 runner 并与训练侧同名指标对表。

### I8 · 部署前校准检查：正弦跟踪 + 相图 + 观测契约逐位核对（策略 rollout 之前）
- **关联结果版本**：idear-ideas-0001 / E4（sim2sim 协议）
- **文件/函数**：验证 runner 的自检脚本（待建）；关节顺序锚点 `X1_DOF_ORDER`
- **观测事实**：[R3] §观测契约清单与 Humanoid-Gym 校准法；社区管线最常见的静默失败是契约错位（关节顺序/四元数约定/相位信号）。
- **机制假设**：X1 29 dof 的 URDF/MJCF/训练平台三套关节顺序几乎必然不同序（工具链已有 `X1_DOF_ORDER` 佐证命名差异）；契约错位导致策略在 MuJoCo 表现为「立刻摔倒」，极易被误诊为动力学 gap。
- **竞争解释**：摔倒也可能真是执行器/接触问题——校准检查的目的正是把这两类失败分开。
- **最小改动或诊断实验**：策略验证前先跑两检：(a) 正弦关节跟踪：对每关节给 0.5–1 Hz 正弦位置目标，MuJoCo 外环 PD vs 训练引擎同目标对比相位滞后与幅值；(b) 契约单测：固定随机动作序列回放，逐位核对 obs 向量（顺序/约定/缩放/钳位）。
- **对照变量**：随机 29 维动作序列在两引擎的 qpos/dq 一致性（开环，1 s 窗口）。
- **预期指标**：正弦跟踪幅值差 <5%、相位滞后差 <10 ms；开环 1 s qpos 累计误差 <机器可分辨阈值（若发散见 I9 定量分级）。
- **支持/否定条件**：支持=校准通过后策略行为与训练侧一致（则剩余 gap 是物理的）；否定=校准即失败 → 契约 bug，修完再谈物理对齐。
- **成本风险**：低；半天工作量，避免整轮 sim2sim 白跑。
- **worker 的下一步**：把 (a)(b) 做成 sim2sim runner 的 `--selftest` 入口（项目工具链已有 `--selftest` 惯例，见 `validate_retarget_v3.py` L16）。

### I9 · 开环重放发散作为动力学 gap 的定量定位工具（ASAP 协议精简版）
- **关联结果版本**：idear-ideas-0001 / E4
- **文件/函数**：验证 runner 增加回放模式（输入：训练侧记录的 state-action 轨迹）
- **观测事实**：[R3] ASAP：跨引擎开环重放全局 MPJPE 0.25/0.5/1.0 s → 19.5/33.3/80.8 mm 的发散尺度，是 gap 分级的现成标尺；PolySim 证明引擎间残差不可用参数 DR 消除。
- **机制假设**：闭环测试失败时无法区分「策略不鲁棒」vs「动力学 gap 大」；开环重放把策略摘除，发散率直接度量引擎差异，指导修哪里（发散快 → 执行器/接触参数错；发散慢而闭环失败 → 策略鲁棒性不足）。
- **竞争解释**：初始状态导出误差（浮点、单位）也会造成发散——需先做 1 步重放（应逐位一致）排除。
- **最小改动或诊断实验**：E4 加入回放档：训练侧 rollout 保存 (qpos,qvel,action) @每步，MuJoCo 侧重放同序列，输出 0.25/0.5/1.0 s 三个窗口的 root 位置误差。
- **对照变量**：回放窗口（0.25/0.5/1.0 s）；修复前后的发散曲线（如调 PD、摩擦）。
- **预期指标**：得到本项目自己的发散标尺；修复应单调压低曲线；1.0 s 误差 <50 mm 大致对应「闭环可救」，>80 mm（ASAP 未修水平）说明有实质性未对齐项。
- **支持/否定条件**：支持=某项修复（如 PD 对齐）显著压低发散且闭环同步改善；否定=发散低但闭环失败 → 策略侧问题（回训练加 DR/延迟）。
- **成本风险**：低；只需训练侧多存一条轨迹。
- **worker 的下一步**：训练 export 时固定保存 1 条完整 (s,a) 轨迹用于回放。

### I10 · E1 指标表对齐 humanoid_pose_standard：14 项统计 + duty×Froude 一票否决
- **关联结果版本**：idear-ideas-0001 / E1、I6
- **文件/函数**：数据侧新脚本（建议挂 `x1_gmr_retargeted_tool/` 同级或正本 `tools/`）；标准来源：项目 skill `humanoid_pose_standard`（九大类指标框架）
- **观测事实**：skill 定义九大类参考指标，标注 R（重定向层可测）/S（仅 sim2sim 策略层）；其中⑤「对地时间结构：duty×Froude 区制、腾空、双支撑期」为**一票否决**级；④足地接触含攻角 10–25°、heel-toe 滚动、滑步、穿模（角点 FK 测量法——与 `validate_retarget_v3.py` 的 8 角点 sole FK 同构，L52–76）。
- **机制假设**：README 自述的 sprint 不自洽（duty 18–20%@1 m/s、腾空至 1.73×）正落在⑤的否决区；run 族 7 段是否越界是 I2 假设的直接检验。Froude 区制（v²/(g·L)）可把「速度-步态」关系无量纲化，天然免疫 time_scale 争议——同一 clip 两种口径的 Froude 不同但 duty-腾空时长关系不变，能区分「慢放」与「物理不自洽」两类问题。
- **竞争解释**：X1 腿长与人类不同，人类 duty/Froude 常模需按腿长换算后才可比；若不换算会误判。
- **最小改动或诊断实验**：E1 实现以下 14 项（R 层，全部本地可算，纯 numpy/纯 python）：
  1–3. root 速度分布（p50/p90/max，含与不含 time_scale 两口径）；4. 步频；5. duty factor（左右分别）；6. 腾空时长分布；7. 双支撑期占比；8. Froude 数（用 X1 实际腿长）；9–10. 攻角分布（触地时脚速度方向与地面夹角，标准 10–25°）；11. 最大脚高；12. 触地滑移（脚触地期间水平位移）；13. 穿模深度（复用 v3 门 R8 口径）；14. 髋抖动 p99（复用 R9 口径）。
- **对照变量**：7 段 run × 2 种 time_scale 口径；输出对照表附每段判决（⑤否决/通过）。
- **预期指标**：全部 7 段通过⑤ → I2 降级为「速度指令重标定」；任一段否决 → 该段降权或剔除，并触发 I6 数据补充。
- **支持/否定条件**：支持=出现⑤否决段且剔除后训练改善（需 E2 证据）；否定=全过（则 time_scale 无害假设成立）。
- **成本风险**：低；注意腾空/duty 判定需先定义触地检测（建议用 v3 已有的 sole 8 角点 zmin < 8 mm 口径，与 STANCE_Z=0.008 一致，`validate_retarget_v3.py` L31）。
- **worker 的下一步**：实现 E1 脚本（约 150 行），输出 markdown 对照表进 dm-results。

---

## 最小验证实验（对 0001 的 E 序列修订与增补）

| ID | 内容 | 相对 0001 的变化 |
|---|---|---|
| E1 | 14 项物理量统计表（I10 细化），含 duty×Froude 一票否决判决 | 指标从 5 项扩到 14 项并锚定项目姿态标准 |
| E2 | Gaussian PPO 跟踪基线（不变） | — |
| E3 | DPPO 式扩散头对比（不变） | — |
| E4 | sim2sim 协议升级为四层：(0) I8 自检（正弦+契约）→ (1) 基准 rollout（速度 RMSE + 0.5 m 阈值成功率，摩擦扫描 0.4/0.6/0.8/1.0）→ (2) I9 开环重放发散（0.25/0.5/1.0 s）→ (3) I5 延迟扫描 0/2/5/10 ms | 新增自检前置层与回放诊断层；摩擦扫描与失败阈值有文献锚点 |
| E5（新） | 模式 A vs 模式 B 执行器对齐对比（同一策略） | 验证 I7 选择 |

顺序建议：E1 →（E2 ∥ E5 的 runner 搭建+I8 自检）→ E3 → E4。

---

## 上下游缺口

1. **验证侧 runner 缺位**（I7/I8/E5）：仓库没有任何 sim2sim 代码；模式 A runner + selftest 是 worker 出第一个策略前就应备好的（否则策略导出后无法验收）。
2. **训练侧轨迹导出契约未定**（I9 依赖）：gradmotion 平台能否按步保存 (qpos,qvel,action) 待确认。
3. **ctrlrange 单位与 tau_limit 语义未澄清**（I7 风险项）：±180 对 motor 若为 Nm 明显超物理（人形膝/髋峰值 ~100–200 N·m，腕 <<10）；需要 worker 对照 URDF（`f1.urdf` effort 字段）或厂家规格澄清，否则力矩钳位在 sim2sim 中不起保护作用。
4. **腿长/质量参数表**（I10 依赖 Froude 计算）：X1 实际腿长（hip 到 sole）与总质量需从 URDF/MJCF 提取固化成常量，避免口径漂移。
5. **训练侧 PD 参数导出**（I7 依赖）：若无导出，则需从 gradmotion 配置读取并写入 policy metadata（TienKung 的做法：TorchScript + 元数据一起导出）。

---

## 反证与不确定性

- **[推断] 模式 A 优于模式 B**：基于「语义对应度」论证与社区采用面，无 X1 实测；E5 就是为证伪此推荐而设。
- **[推断] gradmotion 为 Isaac 系 PD position target**：平台一手信息缺失；若输出语义不同，I7 的前提变化（好在模式 A 框架对力矩输出同样成立）。
- **[未验证] MuJoCo solref/solimp 与 Isaac 默认的等效参数表**：[R3] 明确指出文献缺口（B1 论文只调了 impratio；各仓库数值未文档化）。X1 的 flat.xml 若用默认值，摩擦扫描（E4）是唯一经验性对冲。
- **[推断] duty×Froude 对 X1 的适用区制**：人类常模按 X1 腿长换算后仍需标定；建议先用 7 段 run 的实测分布自身聚类，再对照人类常模边界，避免拿错标尺误判。
- **[未验证] PolySim 3.6–50% 的迁移率对本项目的代表性**：那是 14 个 ASAP 全身动作（含空翻等高动态）；本项目 7 段慢跑的跟踪任务难度低得多，实际 gap 可能显著更小。
- **[推断] humanoid_pose_standard 的 Froude 区制边界值**：skill 文本未给数值边界（仅给指标名）；具体阈值需 worker 结合标准文档或实测分布确定——我未在仓库内找到该 skill 的完整数值表，如存在于别处请 worker 提供引用。

---

## 来源

**本地（commit `fc0b9be`）**
- `X1_29DOF/mjcf/robot/xyber_x1/xyber_x1_serial.xml` L5/L252–283/L284–354
- `x1_gmr_retargeted_tool/validate_retarget_v3.py` L16（selftest 惯例）、L31（STANCE_Z=0.008）、L52–76（sole 8 角点 FK）
- `x1_gmr_retargeted_tool/retarget_g1_x1.py`（X1_DOF_ORDER 定义处）
- 项目 skill `humanoid_pose_standard`（九大类 R/S 指标框架，⑤一票否决）
- `reports/studio/idear/20261003T2359_idear-ideas-0001_diffusion-running-roadmap.md`（承接的 I5/E1/E4）

**文献调研（归档于 `research/studio/idear/`）**
- [R3] `20261004_isaac-to-mujoco-sim2sim-alignment.md`：Humanoid-Gym（arXiv 2404.05695）、TienKung-Lab（Open-X-Humanoid）、UCL B1+Z1（arXiv 2512.18938）、PolySim（arXiv 2510.01708）、ASAP（arXiv 2502.01143）、g1_walk_isaaclab_mujoco 等 12 源 + 15+ 开源管线清单

**缺失证据**：gradmotion 平台一手信息（动作语义、导出格式、轨迹保存能力）；X1 厂家执行器规格（力矩限制、PD 推荐值）；humanoid_pose_standard 的完整数值边界表。
