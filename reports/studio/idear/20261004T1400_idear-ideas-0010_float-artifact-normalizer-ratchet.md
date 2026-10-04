# idear 报告 0010：E6b 否定后的再归因——「浮空」疑点中心/角点口径复核、奖励平台的三重机理、算力阻塞期的离线清单

- 报告 ID：idear-ideas-0010
- 日期：2026-10-04（14:00，基于 worker 13:34 发布的 R003 与共享证据）
- 作者：idear（研究与建议角色）
- 关联结果版本：**dm-results-R003-e6b-channel-chain v1**（digest sha256:7499055b…，实验 TASK_20261004_005/012/006/040/053/054，workspace commit `8e0d6ad`）；直接消费其 §1（E6b 否定）/§1.1（浮空新嫌疑）/§3（事故链与阻塞声明）
- 新增证据：①三套 xml 运动链逐字一致性比对（本报告§证据 A，本地只读）；②e6b 数值再解读：base 0.34 vs r4 平台 0.20 的**策略低于数据**悖论 + DiffNormalizer 累计棘轮算术（§证据 B）；③e6b 脚本 grounded 探针的口径缺陷定位（§证据 C）
- 状态：建议（非验收）。**含对 idear-0009 停滞 A 机理的第二次修正（I30 否定分支已触发，本报告给出替代机理）与对 R003 §1.1 新头号嫌疑的正面质疑（附 5 分钟判定实验）**

---

## 摘要与关联结果

R003 三项关键进展：E6b' 重投影 -2.1% 否定滑移机制（noVel≈base 双重否定）、新头号嫌疑「数据在训练几何浮空 1.3cm+」、算力耗尽（账号 19-24 全空，r4-r9 权重全丢）。本报告给出四点：

1. **对 0009 的诚实结算**：I30 预注册否定分支（<10%）被精确触发——滑移机制出局，这是判据设计在工作而非建议失败。但停滞 A 的「结构性上限」现象本身（峰值 0.49→平台 0.36）需要新机理，见下。
2. **「浮空 1.3cm」高度疑似测量口径伪影**：e6b 用 sole geom **中心** z（`geom_xpos`）测"zmin"，中心高度=踝高+脚俯仰偏移（±3cm 量级）；而 v3 数据平底是按 **8 角点** 校准的（validate_retarget_v3 口径）。我对三套 xml（校准源 `X1_29DOF/mjcf/.../xyber_x1_serial.xml`、训练真源 `.repos/mk_api/x1_train.xml`、sim2sim 派生 `analysis/x1_train_sim.xml`）做了 root→sole 全运动链逐字比对：**lumbar/hip/knee/ankle 全部 body pos+quat 与 sole geom 完全一致**——FK 输入相同则输出必相同，若角点 zmin 在 xyber 上≈0（R001 已证），在 x1_train 上必≈0。**5 分钟角点复测可判定**（I32）；判定前不应启动 v3.2 重标定。
3. **奖励平台的三重机理（替代滑移论）**：(i) ESM 去噪损失的新鲜噪声下限——数据窗口自身 raw loss 0.34，**r4 策略平台 0.20 反而低于数据**——策略已比数据更贴（EMA 平滑后的）流形，继续训练此分量的边际收益封死；(ii) DiffNormalizer 无限样本累计平均的**棘轮效应**——平台奖励 0.36 与 raw 0.20 反解出 μ≈0.39，恰为含早期高 loss（0.6-0.9）的终身平均，奖励数值被早期历史压低是仪表读数问题而非学习问题；(iii) 速度块不敏感（noVel 仅 +2.4%）——损失主要由位姿/关键点块驱动。
4. **阻塞期离线清单**（I35）：全部零算力，且其中三项能改变 r9 之后的方向决策。

---

## 论文与实现证据

**A. 三套 xml 运动链一致性（已观测，本地 grep 比对）**

| 链环节点 | xyber_x1_serial.xml | .repos/mk_api/x1_train.xml（训练真源/e6b 用） | analysis/x1_train_sim.xml |
|---|---|---|---|
| lumbar_yaw pos/quat | 0.00245 0 0.11553403 / 1 0 0 -2.27e-05 | 同左* | 同左 |
| left_hip_pitch | 0.00245 0.092277 -0.012143 / 0.653… 0.270… | 同左（实测 grep） | 同左（实测 grep） |
| left_knee_pitch | -0.0337 0 0.1422 / 0.5 -0.5 0.5 -0.5 | 同左 | 同左 |
| left_ankle_pitch | 0 -0.30494 0.0336 / 0 0 -1 0 | 同左 | 同左 |
| left_ankle_roll | 0 0 0 / 0.70710678 0 0.70710678 0 | 同左 | 同左 |
| sole geom | pos 0 -0.0303 0.005, size 0.055 0.012 0.098 | 同左（实测 grep） | 同左（实测 grep） |

（*lumbar 行仅在 xyber 与 x1_train_sim 间比过；hip/knee/ankle/sole 三套全比。差异仅在关节 range（如膝 0-2.7 vs 0-2.0）与执行器参数——不影响 FK。）
**推论**：同一 (root_pos, root_rot, dof) 在两套几何下 FK 输出逐位相同。R001 已证 8 角点 zmin≈0（穿模 0mm）→ x1_train 几何上角点 zmin 也必≈0 → 「浮空 12.6-64mm」是**中心口径**读数：sole 中心相对踝原点偏移 (0.005, -0.0303, 0)（经 body 90° 旋转），脚俯仰 ±30° 时中心 z 波动 ±1.5cm，触地帧中心高于接触角点 1.3-3cm 正是平底脚的几何预期。

**B. 奖励平台算术（已观测数值的再解读）**

| 量 | 值 | 来源 |
|---|---|---|
| 数据窗口 raw ESM loss（K=[22,15,8]） | **0.340**（p50 0.125） | e6b base_mean（e6b_reprojection_result.json） |
| r4 策略平台 raw loss | **0.20** | r4_curves Sds_Loss 1501-3000 窗口（0009 §A） |
| 平台 Smp_Reward | 0.36（峰值点 0.49） | 同上 |
| 反解 normalizer μ = 2L/(−ln r) | **0.391** | 本报告计算 |
| noVel（速度全零）loss | 0.348（+2.4%） | e6b |

三个推论：(i) **策略 0.20 < 数据 0.34**——训练后的策略比先验的训练数据本身更贴流形（EMA+ l1 使先验偏好平滑模态；策略的规律性步态比含噪多样的原始窗口更易去噪）。SMP 奖励对「更像数据」的激励已饱和，「上限」的实质是**新鲜噪声去噪误差下限**（t∈{22,15,8} 重采样 ε 的不可预测部分）。(ii) μ≈0.39 与「累计平均含早期 0.6-0.9 阶段」自洽（`DiffNormalizer.update` 为 w_new=new/total 的**累计**平均而非指数衰减，`sds_normalizer_samples` 未设→inf→永不冻结，smp_agent L28）。(iii) 综合判读：**平台 0.36 不是策略停滞的警报，而是仪表（奖励定义）+底噪（ESM floor）的合成读数**；策略质量的真实量尺应转向 task 跟踪与 sim2sim 九类指标。

**C. e6b 脚本 grounded 探针的口径缺陷（已观测代码）**

`e6b_reprojection_test.py` L304-318：`shift = zmin`（**中心** zmin，L308）→ root_z 与 key_z 同减 13-64mm。若角点本已贴地（§A 推论），该变换会把脚**埋入地下** 13-64mm，制造伪影而非修复。且 result json 中**无 grounded_mean 键**（脚本 L359 会写）——现存 json 生成于探针加入之前，grounded 数值从未落地。stance 阈值 0.02（L216/L286）与「70% 覆盖」同样基于中心口径。

**D. R003 其余要点（已观测，引用）**：r4 峰值点 0.49（修正 0009 的窗口均值 0.436 表述）；r5@582 0.449 被杀；r6 PASS 0.388/ep_len 67.2（task 分量驱动跟踪的直接佐证：ep_len 从跌倒级到 67 步）；渲染双链路就绪（用户指令）；通道事故链与账号耗尽（19-24）。

---

## 优先建议

### I32 · 5 分钟角点复测：判定「浮空」真伪（P0，先于一切 v3.2 工作）
- **关联结果版本**：dm-results-R003 v1 §1.1（新头号嫌疑）
- **文件/函数**：`analysis/e6b_reprojection_test.py` FK 类（L110-155，现成）；`.repos/mk_api/x1_train.xml`
- **观测事实**：§A 三套 xml 链逐字一致；R001 角点校准 0 穿模；e6b 用中心口径读到 12.6-64mm。
- **机制假设**：浮空=中心/角点口径差（平底脚触地时中心高于接触角点 1.3-3cm 属几何必然）。
- **竞争解释**：若角点复测在 x1_train.xml 上 zmin≠0（§A 比对有遗漏环节，如两侧不对称或 mesh 依赖差异），则浮空为真，v3.2 保留。
- **最小改动或诊断实验**：在 e6b 的 FK.frame 里加角点展开（复用 validate_retarget_v3.sole_geometry 的 8 角点逻辑：`geom_xpos + geom_xmat @ (±size)`），对 7 段全部帧输出角点 zmin 分布。diff（worker 落地）：
  ```diff
  + # in FK.frame: corners instead of centers
  + zmin_c = []
  + for gid in self.sole_ids:
  +     c = self.data.geom_xpos[gid]; R = self.data.geom_xmat[gid].reshape(3,3)
  +     h = self.model.geom_size[gid]
  +     cs = [c + R @ np.array([sx*h[0], sy*h[1], sz*h[2]])
  +           for sx in (-1,1) for sy in (-1,1) for sz in (-1,1)]
  +     zmin_c.append(min(cc[2] for cc in cs))
  ```
- **对照变量**：中心 zmin vs 角点 zmin 同帧并排输出。
- **预期指标**：角点 zmin 分布 ≈ R001 的 [0±3mm] → 浮空证伪；若 [13,64]mm → 证实。
- **支持/否定条件**：即上；判定后 v3.2 立项或撤销。
- **成本风险**：5 分钟。**不判定就投入 v3.2 的风险是数天数据工作白费**。
- **worker 的下一步**：立即跑（离线、零算力），结果一句话记入 R004 或状态更新。

### I33 · grounded 探针修正后再跑（P1，依赖 I32 结论）
- **关联结果版本**：R003 §1.1（"修法=数据重标定或训练资产对齐"）
- **文件/函数**：`analysis/e6b_reprojection_test.py` L304-318
- **观测事实**：§C——shift 用中心 zmin，会把已贴地的脚下埋；grounded 数值未落地。
- **机制假设**：—（口径缺陷，确定性）。
- **竞争解释**：若 I32 证实浮空为真，则中心口径的 shift 方向对、幅度错（应按角点差）——仍需修。
- **最小改动或诊断实验**：shift 改为**角点 zmin**（I32 的输出）；重跑 grounded 变体，得 grounded_drop。若 I32 证伪浮空，则 grounded 探针整体撤销（变换无意义）。
- **对照变量**：grounded 修正版 vs baseline。
- **预期指标**：若浮空为真且是残余 loss 主因 → 修正版 drop>15%；若浮空证伪 → 探针撤销。
- **支持/否定条件**：drop 大 → v3.2 升级为 r9 后第一优先；drop 小 → 残余 loss 归因收束到 ESM floor（§B-i），v3.2 撤销。
- **成本风险**：低。
- **worker 的下一步**：I32 之后顺手。

### I34 · 奖励仪表修正：记录 μ 与 L/μ，停止用 Smp_Reward 绝对值判读停滞（P1，一行改动）
- **关联结果版本**：R003 §3（r5/r6 的 0.449/0.388 判读）；0006-I19 停滞决策树更新
- **文件/函数**：`mimickit/learning/smp_agent.py` L214-216（sds_info）；`data/agents/smp_x1_agent.yaml`
- **观测事实**：§B——μ≈0.39 为终身累计平均；r5/r6/r4 的奖励峰值 0.449/0.388/0.49 差异可能大半来自 μ 漂移而非策略差异。
- **机制假设**：奖励绝对值横向不可比（不同 run 的 μ 历史不同）；raw loss 才是可比量尺。
- **竞争解释**：无——算术上必然（除非 μ 冻结，而 x1 配置为 inf）。
- **最小改动或诊断实验**：diff（worker 落地，与 0006-I20 合并提交）：
  ```diff
  +        for i_t in range(len(self._diffusion_steps)):
  +            sds_info[f"sds_norm_mean_t{self._diffusion_steps[i_t]}"] = \
  +                self._sds_normalizer.get_abs_mean()[i_t]
  ```
  （或最简：`info["sds_norm_mu"] = self._sds_normalizer.get_abs_mean().mean()`。）可选：`sds_normalizer_samples: 1e7` 让 μ 反映近期质量——注意会改变奖励定义，仅 r10+ 消融用，r9 保持冻结配方。
- **对照变量**：raw L、L/μ、exp(-2L/μ) 三条曲线并排。
- **预期指标**：r9 曲线上 raw L 应单调降且平台 <0.20 或持平——这才是「策略还学不学得动」的判据。
- **支持/否定条件**：raw L 平台而 ep_len/task 指标仍升 → 确认 SMP 分量饱和无害（§B 结论）；raw L 尚在降而奖励平 → 纯仪表效应。
- **成本风险**：零（观测性改动）。
- **worker 的下一步**：并入 r9 配置（frozen recipe 只加日志不改行为）。

### I35 · 算力阻塞期离线优先级清单（P2，按信息价值排序）
- **关联结果版本**：R003 §3 阻塞声明 / §5 下一步
- **内容**：(1) I32 角点复测（决定 v3.2 生死）；(2) I33 grounded 修正版（同上）；(3) I34 日志 diff（r9 就绪度+1）；(4) **数据侧攻角基线补齐**（0009 缺口 2：R001 未报数据攻角，R003 判据 4 需要三方对照——复用 e6b FK 的 sole 角点序列即可算，~20 行）；(5) E8 四格消融表设计定稿（账号恢复后与 r9 并行提交，验证 0006 灵敏度归因）；(6) 渲染链路的人形确认（R003 待办：video_inspect 服务恢复后复核 _smoke_render.mp4）。
- **成本风险**：全部零算力/本地 CPU。
- **worker 的下一步**：按序推进；账号恢复则 r9 一键启动（配方已冻结 @ f591eb0）。

---

## 最小验证实验

| ID | 内容 | 依赖 | 预算 | 判据 |
|---|---|---|---|---|
| E10（新） | 角点 vs 中心 zmin 复测（I32）+ 修正版 grounded（I33） | e6b FK（现成） | 10 分钟 | 浮空真伪判定 |
| E11（新） | 数据侧攻角分布（I35-4） | e6b FK + 角点速度 | 30 分钟 | 补齐 R003 三方对照第三点 |
| r9 前置 | I34+I20 日志 diff 合入冻结分支 | — | 一行 | — |

---

## 上下游缺口

1. **checkpoint 全丢**（r4-r9 事故链）：算法结论只能靠曲线与离线测试支撑，sim2sim 最终结论仍「未验证」（R003 §5 自认）——账号恢复后 r9 + E4 是唯一路径。
2. **r5/r6 的 raw loss 不可比**：r4_curves 只发布了 r4；r5（0.449@582）与 r6（0.388）若无 raw Sds_Loss 曲线，横向比较无效（I34 的动机）。
3. **EMA vs 非 EMA 先验**：e6b 用 `x1_prior.pt` 的 EMA 权重（load_state_dict from main）——若 r9 后换先验，§B-i 的「策略<数据」结论需复测。
4. 渲染人形确认（video_inspect 503 未复）——R003 已列。

---

## 反证与不确定性

- **[已观测] 三套 xml 链一致 / 中心 vs 角点口径 / json 无 grounded 键 / noVel≈base / base 0.34 vs 平台 0.20**：本地 grep 与文件直读，可复核。
- **[推断] 浮空=口径伪影**：链一致性 → FK 相同 → 角点必≈0 是演绎推理，但**未经角点复测实证**（I32 就是证伪通道；若 R001 的角点测量与 e6b 的中心测量之间存在第三种差异——如 R001 实际用的也是中心——则反转）。注意 R003 称"n=34 采样点"的浮空探针细节未见于共享文件，以 I32 实测为准。
- **[推断] μ≈0.39 为累计平均解释**：由平台奖励与 raw loss 反解，与累计平均机制自洽；但 r4 早期 μ 演化（init=1 起）未逐 iter 复算，量级可信、精确值待 I34 日志。
- **[推断] ESM 新鲜噪声下限主导残余**：由「策略<数据」+「noVel 不敏感」推出；竞争解释是关键点/位姿块的特定偏差（如根高分布差）——若 I33 修正版 grounded drop 大，则此推断被部分推翻。
- **[结算] 0009-I30 机制假设被否定**：滑移分量不是残余 loss 主因（-2.1%）；0009 的「结构性上限」现象描述仍成立但机理改写（本报告 §B）。0009-I29（R003 判读指南）保持有效。
- **[勘误] 0009 的峰值表述**：「峰值 0.44」为窗口均值；R003 口径峰值点 0.49。无实质影响。

---

## 来源

**共享产物（已 fetch 快照）**
- dm-results-R003-e6b-channel-chain v1（`inputs/studio/dm-results-R003-e6b-channel-chain/`，sha256:7499055b…）

**共享证据文件（commit `8e0d6ad` 或未提交，只读）**
- `analysis/e6b_reprojection_test.py`（L110-155 FK、L143-144 中心口径、L210-216 注释与阈值、L304-318 grounded、L359 json dump）；`analysis/e6b_reprojection_result.json`（无 grounded 键）
- `analysis/r4_curves/r4_iter_points.json`（0009 已分析）
- 三套 xml 链比对：`X1_29DOF/mjcf/robot/xyber_x1/xyber_x1_serial.xml`、`.repos/mk_api/x1_train.xml`、`analysis/x1_train_sim.xml`
- `research/studio/idear/mimickit_ref/mimickit_learning_diff_normalizer.py`（累计平均语义，0004 归档）

**计算**：μ=2×0.20/(−ln 0.36)=0.391；策略/数据 loss 对照见 §B 表。

**缺失证据**：角点 zmin 在 x1_train.xml 上的实测（I32 待跑）；r5/r6 raw Sds_Loss 曲线；r4 μ 逐 iter 演化。
