# idear 报告 0005：0004 缺口补全——disc_obs 含速度分量（H-B 前提成立）与先验配置实测

- 报告 ID：idear-ideas-0005
- 日期：2026-10-04
- 作者：idear（研究与建议角色）
- 关联结果版本：承接 idear-ideas-0004（dm-results-R001-baseline v1）；本报告仅补全 0004「上下游缺口 1/2」两处证据，不重复 0004 结论
- 新增证据：`mimickit/envs/amp_env.py` 与 `tools/diffusion_model/config/tinymdm_x1_multi_clip.yaml`（x1_mimicKit @ `88dbd89e`，已归档 `research/studio/idear/mimickit_ref/`）
- 状态：建议（非验收）

---

## 摘要与关联结果

拉取并核实 0004 遗留的两个源码缺口，结果都强化「先诊断后调参」（I13/I14）的必要性：

1. **H-B（滑移流形不可达）的机制前提成立**：`amp_env.py` 的 disc_obs 构造显式包含 `root_vel`、`root_ang_vel`、`dof_vel`（`_compute_disc_obs` L53–64，且 `disc_dof_vel_obs` 默认 True，L12）。R001 观测的支撑相滑移 25–35% 会以「支撑相 root 前向速度 + 非零脚速」的形式直接编码进先验流形；物理仿真中摩擦禁止该状态 → 策略可达域与流形在速度维度上系统性错开。
2. **先验实际容量/训练量确认**：`tinymdm_x1_multi_clip.yaml`——DiT 仅 **2 层**（arch.py 默认 4 层），4 头 × head_dim 64（inner_dim 256），EMA 0.995；**T=50**（确认 K=[22,15,8]={0.44,0.30,0.16}×N 公式）；loss_type l1、epsilon 预测、cosine schedule；batch 512 × **200k iterations ≈ 1.02×10⁸ 采样次**对约 2.6k 个 10 帧窗口——每窗口平均被重复采样数万次。**先验分布必然是 92 s 数据的精确窄流形**（扩散加噪提供一定平滑，但低 t 步的 ε̂ 网络对数据外状态的响应仍可能很陡）。
3. **对 E6b 解读的更新**：H-A 的「过拟合」表述应修正为「流形窄化」（2 层容量下经典过拟合难，但分布支撑窄是必然的）；H-B 与 H-A 实为同一机制的两侧：**窄流形 + 部分物理不可达 → 物理状态普遍落在流形外**。E6b 的三层测试中，「消滑移重投影后 loss 变化」的判读阈值不变（≥2× 判 H-B），但「加噪泛化」一层若显示 σ=0.01 即跳变，应视为流形窄化的直接证据而非网络缺陷——**修复方向是数据增广（I16）而非改网络**。

## 论文与实现证据

- `mimickit/envs/amp_env.py`（归档件）L11–12：`num_disc_obs_steps`、`disc_dof_vel_obs` 默认 True；L37–64：`_compute_disc_obs` 以 `ref_root_pos/root_pos/root_rot/root_vel/root_ang_vel/joint_rot/dof_vel/body_pos` 构造 obs（demo 侧与仿真侧共用同一构造，域一致性由 `_check_prior_env_config` 的 5 项 assert 保障——但 assert 只查 flag，不查分布）。
- `tools/diffusion_model/config/tinymdm_x1_multi_clip.yaml`（归档件，全文 30 行）：上述全部字段；`normalizer_std_clip: 0.2`；`control_freq: 30`（先验在 30Hz 域，与数据 fps=30 一致）；`env_config: data/envs/smp_x1_env.yaml`（仍未拉取，disc_obs 步数等在彼处）。
- `mimickit/learning/smp_model.py` 仅 165 字节（stub/import 转发，无增量信息）。

## 优先建议

### I17 · E6b 判读与修复方向的联动表（对 0004-E6b 的操作化增补）
- **关联结果版本**：idear-ideas-0004 / I14（E6b）、I16（E7）
- **文件/函数**：同 0004
- **观测事实**：本报告两项核实（disc_obs 含速度；先验 2 层/200k iter/窄流形必然性）。
- **机制假设 → 判读 → 修复的映射**（可检验决策表）：

| E6b 观测 | 判定 | 首选修复 | 次选 |
|---|---|---|---|
| 基线 loss 本身高（数据窗口 loss 不低） | 先验未收敛（训练问题） | 查先验训练曲线/延长训练 | — |
| σ=0.01 即 loss 跳变 ≥10×，消滑移后 <2× | 流形窄（H-A'） | I16 数据增广（镜像/时间伸缩/噪声窗口入训练集） | 扩 K 向高 t（如加 t=30/35） |
| 消滑移重投影后 loss ≥2× | 滑移锚定（H-B） | I16 支撑相速度重投影 v3.1 | disc_obs 关 dof_vel_obs 重训先验（侵入大，后置） |
| 三层都温和（loss 平滑、低值） | 先验健康 | 回 I13 指纹②③：查 actor 更新链路/日志 std | H-C 消融 E8 |

- **竞争解释**：H-A' 与 H-B 可叠加（窄流形且部分不可达）——若两层测试都显著，I16 的增广应同时覆盖两类（重投影 + 噪声/镜像）。
- **最小改动或诊断实验**：同 0004-E6b，无新增实验；仅判读规则。
- **对照变量/预期指标/支持否定条件/成本风险**：同 0004。
- **worker 的下一步**：E6b 输出后按表行动；若走 I16，增广数据建议落在 `data/datasets/` 新 yaml（如 `dataset_x1_run_v31.yaml`），不动原数据。

## 最小验证实验

无新增编号；E6a/E6b/E7/E8 定义见 0004。本报告仅更新 E6b 的判读表。

## 上下游缺口

1. `data/envs/smp_x1_env.yaml`（`num_disc_obs_steps` 窗口步数、key_bodies 等先验训练域的完整定义）仍未拉取——E6b 复现数据窗口构造时需要；worker 本地可读。
2. 先验训练日志（loss/EMA 曲线）仍缺——基线 loss 高时的判定依据。
3. `compute_disc_obs` 的具体拼接顺序与 ref_root_pos 的对齐方式（amp_env 上游函数，在 char_env/base 侧）未核实——E6b 脚本若直接构造 disc_obs 需复用 `fetch_disc_obs_demo`（amp_env L32–35 已有现成接口）而非手拼。

## 反证与不确定性

- **[已观测] disc_obs 含速度**（源码级）；但仿真侧 disc_obs 与 demo 侧的构造是否逐位一致仍依赖运行时（assert 只覆盖 flag 子集）——E6a 若走指纹②，应同步 diff 两侧 disc_obs 的统计（直方图对比）。
- **[推断] 「窄流形必然」**：由 200k×512 采样次 vs 2.6k 窗口的算术推出；EMA/l1/加噪的正则化强度未量化，实际流形宽度以 E6b σ 扫描为准。
- **[推断] 30Hz 控制域**：`control_freq: 30` 与 fps=30 一致推出；若 Isaac 侧实际控制频率不同（如 30.0 浮点舍入），`_check_prior_env_config` L107–110 的 assert 会拦——r3 已通过该 assert（训练跑起来了），故此项一致性大概率没问题。
- **[未验证] normalizer_std_clip=0.2 的作用域**：字段名提示是先验训练时的输入标准化裁剪；与 DiffNormalizer（reward 侧）无关。若 E6b 显示 loss 量纲异常，回头查此值。

## 来源

- x1_mimicKit @ `88dbd89e`：`mimickit/envs/amp_env.py`（L11–12/37–64/94–95）、`tools/diffusion_model/config/tinymdm_x1_multi_clip.yaml`（全文）、`mimickit/learning/smp_model.py`（stub 确认）——归档 `research/studio/idear/mimickit_ref/`
- 承接：idear-ideas-0004（I13–I16/E6–E8）、dm-results-R001-baseline v1
