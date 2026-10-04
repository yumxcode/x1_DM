# idear 报告 0007：sim2sim_validate.py 预审——触地判定阈值 bug 会作废 R003 对地时间结构指标

- 报告 ID：idear-ideas-0007
- 日期：2026-10-04
- 作者：idear（研究与建议角色）
- 关联结果版本：**dm-results-R002-r4-sim2sim-align v1** 引用的未提交证据文件 `analysis/sim2sim_validate.py`（353 行，2026-10-04 01:16 版本，R003 将基于它产出九类指标）；`analysis/x1_train_sim.xml`（26KB，01:13 版本）
- 证据等级：已观测（对 worker 未提交文件的只读审查 + XML 几何手算；本地无 mujoco/numpy，FK 数值未经仿真验证——见反证节）
- 状态：**R003 前置预警**（若不修，R003 的⑤类一票否决指标无效）

---

## 摘要与关联结果

R002 计划 checkpoint 到手后用 `analysis/sim2sim_validate.py` 跑 R003（九类指标）。只读预审发现 3 个影响 R003 有效性的问题，按严重度排序：

1. **[P0-阻断] 触地判定阈值错误**：L299–300 `contact = feet[:,2] < 0.02` 用 ankle_roll_link **body 原点** z 判触地；但由 `x1_train_sim.xml` 几何推算，踝原点正常站立时离地约 **5.5 cm**（sole box 半厚映射到世界 z，见§证据 A），阈值 0.02 意味着踝需下沉 3.5 cm 才判触地——**正常触地时 contact 恒 False** → duty≈0、flight_ratio≈1、double_support≈0、step_SI 失真 → R003 的⑤对地时间结构（一票否决项）、④穿模（penet 用同一 z，量纲偏 5.5cm 但符号逻辑仍对——穿模深度会低估为负值被 max(0,·) 吞掉）、①SI 全部无效。
2. **[P1-协议缺口] `--latency-ms` 参数已解析但未实现**：L246 解析、L262–271 rollout 循环无延迟缓冲（`a = policy(obs)` 直连 `apply_action`）。R002 §1 处置表称「E4 协议参数已内置」——**pd_scale 已实现、latency 未实现**。E4 延迟轴将空转。
3. **[P2-覆盖缺口] 九类指标子集缺攻角与滑移**：L287–323 已覆盖③⑤部分①⑥部分；缺④攻角（10–25° 标准）、④滑移（触地期水平位移）——**滑移恰是 H-B（数据滑移 25–35% 是否被策略继承）的判读关键**，R003 无此指标则 0006-I19 停滞 A 分支无法定论。

---

## 论文与实现证据

**A. 踝原点离地 ~5.5 cm 的几何推算（已观测 XML + 手算）**

`analysis/x1_train_sim.xml`：
- `left_ankle_roll_link` body：`pos="0 0 0" quat="0.70710678 0 0.70710678 0"`（wxyz，绕 y 轴 90°）→ 旋转矩阵 R 把**局部 z → 世界 x**、局部 x → 世界 −z。
- `left_ankle_roll_link_sole` geom：`pos="0 -0.0303 0.005" size="0.055 0.012 0.098"`——局部 y 半宽 0.012（脚宽 2.4cm→世界 y ✓）、局部 z 半长 0.098（脚长 19.6cm→世界 x ✓）、**局部 x 半厚 0.055（→世界 z，即踝原点在 sole 盒中心平面上下各 5.5cm）**。
- 交叉验证：`retarget_v3.py` 的 `FOOT_Z_FLOOR = 0.046`（v3 IK 脚地板高度）与 0.055−余量吻合——踝离地 4.6–5.5 cm 是本项目几何共识。
- 结论：正常触地（sole 下表面 z≈0，渗透 <3mm）时 ankle 原点 z≈0.052–0.055 ≫ 0.02 → L299 判定恒 False。

**B. R001 数据侧口径（对照基准）**
R001 用 MuJoCo FK 的 **sole 8 角点 zmin < STANCE_Z=0.008** 判触地（`validate_retarget_v3.py` L31 口径）——sim2sim 脚本应同口径，否则 R003 的 duty/flight 与 R001 数据侧（0.29–0.45/0.13–0.34）不可对照。

**C. latency 参数（已观测代码）**
L246 `ap.add_argument("--latency-ms", ...)`；L262–271 主循环无队列/缓冲；`grep latency` 全文仅此一处实质引用。

---

## 优先建议

### I23 · 触地判定改 sole 几何口径（P0，R003 前必修）
- **关联结果版本**：dm-results-R002-r4-sim2sim-align v1 §3/§4
- **文件/函数**：`analysis/sim2sim_validate.py` L275（frames 记录）、L299–300（判定）
- **观测事实**：见§证据 A/B。
- **机制假设**：—（确定性 bug）。
- **竞争解释**：无。
- **最小改动或诊断实验**：两方案任选（diff 供 worker 落地）：
  ```diff
  # 方案1（快，与 R001 可比）：frames 记录 sole geom 8 角点 zmin
  - feet=[sim.data.xpos[i].copy() for i in sim.key_body_ids[:2]],
  + # per-step: sole box corners via geom_xpos/geom_xmat (R001 口径)
  + feet=[_sole_zmin(sim, f"{s}_ankle_roll_link_sole") for s in ("left","right")],
  ...
  - contact_L = feetL[:, 2] < 0.02
  + contact_L = feetL_sole_zmin < 0.008   # STANCE_Z, validate_retarget_v3 L31
  ```
  ```diff
  # 方案2（更快但粗）：保留 body 原点，阈值改踝高
  - contact_L = feetL[:, 2] < 0.02
  + contact_L = feetL[:, 2] < 0.055 + 0.008  # ankle origin height + margin
  ```
  方案 1 精确且与 R001/数据门同口径（推荐）；方案 2 五分钟改完（脚姿态变化时误差 ~1cm）。
- **对照变量**：两种口径各跑同一 rollout，duty 差应 <0.05（若大说明脚姿态变化大，必须方案 1）。
- **预期指标**：duty 恢复到 0.25–0.5 量级（数据侧 0.29–0.45 对照）；flight_ratio 0.1–0.35。
- **支持/否定条件**：修后 duty≈0/flight≈1 仍出现 → 判定另有问题（如 rollout 全程真无触地=策略未学步态，那也是 R003 的真发现）。
- **成本风险**：低；注意 `_sole_zmin` 需 geom_xmat 8 角点展开（R001 baseline_metrics.py 已有同构实现可抄）。
- **worker 的下一步**：R003 前落地方案 1；若时间紧先用方案 2 出初版、方案 1 复核。

### I24 · latency 实现补齐（P1，E4 延迟轴前置）
- **关联结果版本**：R002 §1 E4 处置「--latency-ms 参数已内置」
- **文件/函数**：`analysis/sim2sim_validate.py` L262–271
- **观测事实**：参数未接线（§证据 C）。
- **机制假设/竞争解释**：—（工程缺口）。
- **最小改动或诊断实验**：
  ```diff
  + import collections
  ...
  + n_delay = max(0, int(round(args.latency_ms / 1000.0 * 30)))  # control steps
  + abuf = collections.deque([np.zeros(29)] * n_delay, maxlen=n_delay + 1) if n_delay else None
    while t < args.duration and not done:
        a = policy(obs)
  +     if abuf is not None:
  +         abuf.append(a.copy()); a = abuf.popleft().copy()
        for _ in range(n_phys):
  ```
  （0ms 时行为不变；延迟量化到控制步 33ms 粒度——对 2/5/10ms 需物理步级缓冲，若 E4 要扫 2ms 档建议改为物理步计数缓冲，粒度 8.3ms。）
- **对照变量**：0/33/66/100ms（控制步粒度）或 0/8.3/16.7/25ms（物理步粒度）。
- **预期指标**：延迟敏感性曲线（BeyondMimic 口径：2ms 起速度误差上升，5ms→1/3 失败）。
- **支持/否定条件**：—（工具项）。
- **成本风险**：低。
- **worker 的下一步**：与 I23 一并提交；E4 扫描前自测 0ms 与实现前行为一致。

### I25 · 补攻角与滑移两指标（P2，H-B 判读前置）
- **关联结果版本**：0006-I19 停滞 A 分支；R001 §4.1（数据滑移 25–35%）
- **文件/函数**：`analysis/sim2sim_validate.py` L287–323 `analyze_episode`
- **观测事实**：现有 keys 无 attack angle / slip；frames 已存 feet 位置与 qvel，可后处理补齐。
- **机制假设**：策略继承数据滑移 → 触地期脚水平速度 ≈ 前进速度×(0.25–0.35)；若物理摩擦已抑制 → ≈0 且步长分布改变（策略用真粘附步态走）。两者对 0006-I19 的分支选择相反。
- **竞争解释**：滑移低也可能因为策略速度整体低——需与 fwd_vel_mean 联判（滑移率=触地期脚水平位移/身体前进量，R001 同口径）。
- **最小改动或诊断实验**：`analyze_episode` 增两键：
  ```diff
  + # 触地期（I23 修正后口径）脚水平速度均值 / 身体速度均值 = slip_ratio
  + # 攻角：触地瞬间脚速度方向与地面夹角，理想 10–25°
  + r["slip_ratio_L"] = ...; r["attack_angle_deg_p50"] = ...
  ```
- **对照变量**：与 R001 数据侧滑移率（0.25–0.35）直接对照。
- **预期指标**：slip_ratio <0.05（物理粘附）或 ≈0.3（继承数据滑移）——二值即可判。
- **支持/否定条件**：slip≈0.3 且 duty 正常 → H-B 强支持（SMP 流形滑移分量被策略复现）；slip<0.05 → 策略已物理修正滑移，停滞 A 的「SMP 奖励上限<1」推断弱化。
- **成本风险**：低。
- **worker 的下一步**：R003 指标表加这两项；结果直接写入 R003 的 H-B 判读段。

---

## 最小验证实验

无新增编号——I23/I24/I25 全部为 R003 前置修复/增补，修完即并入 E4/R003 执行。

---

## 上下游缺口

1. `load_policy` 仅支持 jit（L326–349），state_dict+normalizer 分支 TODO——R002 已声明，checkpoint 到手后先解决（否则 R003 无法启动）。
2. 九类中②腰/躯干 pitch 分区、⑥力矩/vGRF/摩擦锥、长时程漂移未实现（L287 注释自认 subset）——一期可接受，二期补。
3. friction 1.0/0.05/0.05（xml 实测）vs IsaacLab USD 默认（R002 未验证项）——E4 friction 轴覆盖（0006-I21 已列）。

---

## 反证与不确定性

- **[推断-需仿真复核] 踝离地 5.5cm**：由 XML 旋转链手算（body quat 绕 y 90° + sole geom 尺寸），无本地 mujoco 验证。交叉证据：FOOT_Z_FLOOR=0.046（retarget_v3）量级吻合。**若 worker 在 MuJoCo 里实测踝原点站立高度 <0.02m，则本 P0 误报**——一行脚本可证伪：reset 后打印 `data.xpos[ankle_body_id][2]`（HOME_QPOS 站立应 ≈0.055 或按 FOOT_Z_FLOOR ≈0.046，两者均 >0.02）。
- **[已观测] latency 未实现 / 指标缺项**：代码直读，无歧义。
- **[推断] 方案 2 阈值误差**：脚 pitch/roll 变化 ±15° 时踝高变化 ~±2.5cm（脚长 19.6cm 的半长投影）——攻角大的跑姿下方案 2 会漏检触地末段，故推荐方案 1。
- **[提醒] 该脚本为 worker 未提交文件**：本报告基于 01:16 版本（353 行）；若 worker 已在修改，以最新版为准复核三处行号。

---

## 来源

- `analysis/sim2sim_validate.py`（未提交，2026-10-04 01:16，353 行）：L34–55（DOF/HOME）、L143–151（strip springs）、L212–220（servo）、L246/262–271（latency 未接线）、L287–323（指标）
- `analysis/x1_train_sim.xml`（未提交，01:13）：`left_ankle_roll_link` body/quat、`left_ankle_roll_link_sole` geom pos/size、ankle_pitch body 链
- `x1_gmr_retargeted_tool/retarget_v3.py`（commit fc0b9be）：FOOT_Z_FLOOR=0.046
- `x1_gmr_retargeted_tool/validate_retarget_v3.py` L31：STANCE_Z=0.008（R001 同口径）
- dm-results-R001-baseline v1 §3（数据侧 duty 0.29–0.45）、dm-results-R002-r4-sim2sim-align v1 §3–4
- 承接：idear-ideas-0006（I19 停滞决策树、I21 E4 协议）
