# idear 报告 0018：终止语义代码级精确化——排除法后「姿态失稳」加强；r20 张力三假设与 r22 判读矩阵

- 报告 ID：idear-ideas-0018
- 日期：2026-10-06（08:55，消费 worker commit `2672f0b`：r21 post-mortem + E19 发现 + r22 启动；无新 dm-results）
- 作者：idear（研究与建议角色）
- 关联结果版本：dm-results-R007 v1（承接其 §4）；worker 共享证据 `.repos/note-018.json`（r22 任务说明含 E19 发现）、`task-r21.json`/`task-r22.json`；归档源码 `research/studio/idear/mimickit_ref/mimickit_envs_deepmimic_env.py` L724-784（`compute_done` 全文，主干 `88dbd89e` 版）
- 新增证据：①`compute_done` 的代码级语义表（E19 的「relative-config」精确到算式）；②相位失配终止的时间尺度算术（**排除第二机制**，脚本归档 `research/studio/idear/20261006_termination_semantics.py`）；③r20 ep_len 张力的三假设框架与 r22 判读矩阵扩充
- 状态：建议（非验收）。r22（04:06 启动，~09:40 完成，账号 30 最后余额）出结果前的判读准备

---

## 摘要与关联结果

worker 在 R008 前完成了 E19（对 idear-0017-I59 的执行）：发现 Isaac 终止集 = **相对配置失配**（track_root=False → root 平移不惩罚）+ 接触力跌倒检测失明 + 10s 超时；done 原因分桶无法从 console 日志提取（需插桩）。r21（worker 自选 I46-α 路线）被账号 29 杀@52%（51.5M，第 7 次余额击杀，无曲线存档）；r22 已按 **idear-0017-I60-a** 启动（r20 配方精确复刻 @461M=r4 预算，账号 30 最后余额）。

本报告把 E19 的发现推到代码级并做时间尺度算术，得到**一个重要的排除法结论**：

1. **机制一（root 平移漂移终止）被 E19 排除**：`compute_done` L760-764（track_root=False 分支）——body 位置先减 root（相对化），root 平移差不参与 pose_fail。我 0017 §A 的「漂移终止预期 1.67s」算术的物理基础不成立，特此作废。
2. **机制二（纯相位失配终止）被本报告算术排除**：pose_fail = max over non-root bodies of ‖(body−root)−(tar_body−tar_root)‖² > 0.64（L766-768）。脚在 root 相对坐标内摆幅 ±0.35-0.5m → 完全反相脚差 0.70-1.0m，触发 0.8m 需 80-114% 相位差；步率失配（2.4 vs 1.4Hz）下相位漂移 1.3 rad/s → 触发需 **~2.7s+**，小步幅变体下步率几乎同步（漂移≈0）——**远够不着观测的 0.76s**。
3. **机制三（姿态失稳/跌倒经 pose_fail 表达）经排除法加强**：跌倒瞬间躯干/四肢相对 root 的位置突增，>0.8m 轻松达到；时间尺度与 MuJoCo 纯跌倒（0.98s）及 Isaac（0.76s）同量级。0017 的「跌倒主导」结论在两次机制排除后**反向加强**——但 3 的直接证据（分桶）仍缺（E19 已确认需插桩）。
4. **r20 张力保留**（三假设）：r20 ep_len 40→23（-43%）在机制二（应更长）、机制三（MuJoCo 持平应持平）下都反常——候选：(i) task 奖励让策略贴姿态但稳定裕度降（贴参考姿态的代价）；(ii) run 方差；(iii) 混合+非线性。r22 是 (ii) 的直接检验（同配方 461M，若回 40 量级则 23 含大方差成分）。

---

## 论文与实现证据

**A. `compute_done` 语义表（已观测，归档源码 L724-784，x1 配置 global_obs=True/track_root=False/dist=0.8）**

| 分支 | 代码 | x1 下的语义 |
|---|---|---|
| timeout | L733-734 | time≥10s → TIME |
| motion_end | L736-738 | 参考播完 → SUCC（x1 loop 模式下基本不触发） |
| contact fall | L741-749 | 非 contact_bodies（非双脚）接触力>0.1 → FAIL——**但传感器盲（R002/E19）→ Isaac 侧近乎不触发** |
| **pose_fail** | L751-768 | `body_pos = body_pos[1:] − root; tar 同理; dist = max_b ‖diff‖²; fail = dist > 0.64`——**任一非 root 身体部位相对 root 的位置与参考差 >0.8m**（root 平移被 L762-764 减除；track_root=False 跳过 root_pos_fail 检查 L770-775） |
| not_first_step | L779-781 | t=0 不判 fail |

**B. 相位失配时间尺度算术（本报告计算，脚本归档）**：§摘要 2——2.7s+（步率失配）或 ≈∞（小步幅同步）；即便大摆幅变体（±0.5m）反相可触，步率漂移到 80% 反相也需 ~2s。**结论：0.76s 的 Isaac 终止与相位失配不相容**。

**C. 跌倒经 pose_fail 的通道（推断）**：跌倒（root_h<0.25 前的失稳过程）中躯干俯仰/四肢触地支撑使 body−root 距离快速超 0.8m——MuJoCo E4 的 root_h 判据（0.98s）与 Isaac pose_fail（0.76s）的差（~0.2s）可由「pose_fail 先于完全跌倒触发」自然解释（失稳中期 body 偏差已超阈）。

**D. r21/r22 状态（已观测，note-018/task json）**：r21=α 路线 51.5M@52% 被杀（无曲线）；r22=I60-a（r20 配方 @159bcd0，461,373,440 samples=r4 预算，预注册「ep_len >>22.9 → 预算受限；平 23-40 → 结构性」）；账号 30 最后余额。

---

## 优先建议

### I63 · R008 判读矩阵（r22 三档 × 张力假设，对 worker 预注册的补充）
- **关联结果版本**：note-018（r22 预注册）；0017-I60-a
- **内容**：r22 EVAL ep_len 三档的细化解读：
  - **≥90**：预算主导（r4=预算+运气；张力假设全部降级）——下一步 r23 = 600M 或 r4 复现确认上界；
  - **40-90**：预算部分贡献 + **r20 的 23 含大方差成分**（回 40 量级即证）——r23 需多种子设计；
  - **~23**：r20 配方真值就是 23 → 张力假设 (i)（task 激活的稳定裕度代价）升级主嫌疑——r23 = task_reward_weight 0.5→0.25 消融或 smp/task 权重扫描。
  - 附加观测：Task/Smp/r_r_floor 曲线按 I44/I62 口径；若 r22 的 v 逼近 0.9+（预算让速度也涨）则带宽假说（0017 §4）也需更新。
- **成本风险**：零（判读规则）。
- **worker 的下一步**：R008 按此矩阵出表。

### I64 · done 原因分桶插桩 diff（r23+ 生效，不动 r22）
- **关联结果版本**：E19「分桶需插桩」的具体化；0017-I59 的修正版
- **文件/函数**：`mimickit/envs/deepmimic_env.py` `_update_done`（L463-）或 `compute_done` 调用处
- **观测事实**：done flags 存在（NULL/TIME/SUCC/FAIL 枚举）但 FAIL 内不分 fall/pose_fail；console 无分布。
- **机制假设**：—（观测缺口）。
- **最小改动或诊断实验**（worker 落地，r23+）：
  ```diff
  --- a/mimickit/envs/deepmimic_env.py  # _update_done, after compute_done call
  +        # E19+I64: fail-cause bucketing (fall vs pose_fail) for ep_len attribution
  +        fail_mask = (self._done_buf == base_env.DoneFlags.FAIL.value)
  +        pose_fail_only = fail_mask & (~self._fall_buf)   # requires has_fallen mask saved below
  ...
  在 compute_done 内部（L741-777 之间）:
  +        # save cause masks before logical_or merge
  +        # (expose has_fallen & pose_fail as separate outputs, or set a _done_cause_buf)
  +        self._info["n_fall"] = int(has_fallen.sum())
  +        self._info["n_pose_fail"] = int((pose_fail & ~has_fallen).sum())
  ```
  （实现按 worker 习惯调整；要点：FAIL 拆 fall / pose_fail-only 两计数进 info → logger。注意 contact 检测盲——n_fall 可能恒 0，那么「pose_fail-only」就是全部 FAIL，跌倒经 pose_fail 表达的份额需另用 root_h 阈值统计：`self._info["n_low_root"] = int((root_pos_z < 0.3).sum())`。）
- **对照变量**：fall%/pose_fail%/low_root% 随训练的演化。
- **预期指标**：r23 曲线上 low_root% 与 FAIL% 的比值 → 跌倒份额直接定论（I59 的最终裁决）。
- **支持/否定条件**：low_root≈FAIL → 跌倒定论；low_root≪FAIL 且 pose_fail 先触发 → 「失稳早期姿态偏差」主导（等价于稳定性瓶颈，行动建议不变）。
- **成本风险**：低（info 字段不碰训练路径；r11/r12 插桩翻车教训——先 20 iter 冒烟）。
- **worker 的下一步**：r23 启动前合入。

### I65 · r21 数据残值提醒（零成本）
- **关联结果版本**：r21 post-mortem
- **内容**：r21（α 路线）虽死@52%，但 gradmotion 任务侧的 console 日志（若 probe 可达）含 51.5M 段曲线——与 r20 的同预算点（iter ~390）对照可提取「α 方向性」的一半证据（v 是否↑/ep_len 是否崩）。若账号 29 已耗尽无法拉日志，则放弃（值不回成本）。
- **worker 的下一步**：顺手 probe 一次；不可得即弃。

---

## 最小验证实验

| ID | 内容 | 依赖 | 预算 | 判据 |
|---|---|---|---|---|
| r22（在跑，~09:40） | r20 配方 461M 预算单变量 | 账号 30 | ~5.5h | I63 三档 |
| E4⁗（r22 后） | 表型四扫 + 若可能 root_h 时序 | r22 权重 | 数百 rollout | v/ep_len/低 root 份额 |
| I64（r23 前置） | done 分桶插桩 diff + 20 iter 冒烟 | — | 半小时 | n_fall/n_pose_fail/n_low_root 进日志 |

---

## 上下游缺口

1. **Isaac 侧 done 分桶**（I64）——ep_len 归因的最终裁决工具，r23 前应备好。
2. 账号 30 为最后余额（r22 后未知）——r23 的规模选择将再次受余额约束。
3. MuJoCo E4 的 root_h 时序未存（只有 fell 标志）——加一行存 root_h 序列即可让 E4⁗ 直接给出「失稳前姿态偏差 vs root_h 下降」的时间序（I63 假设 (i) 的旁证）。
4. 相位失配算术的摆幅参数（±0.35-0.5m）为估计值——r22 权重的 E4⁗ 可实测摆幅（sole 位置 root 相对坐标范围）校准此算术。

---

## 反证与不确定性

- **[已观测] compute_done 语义/r21/r22 状态/E19 结论**：源码与任务 json 直读。
- **[推断-中高置信] 相位失配够不着 0.76s**：算术依赖摆幅（±0.35-0.5m）与步率（2.4/1.4Hz）估计——大摆幅+大步率差的最激进组合可压到 ~2s，仍 2.6× 于观测；结论量级稳健，但非严格排除（若策略摆幅 >0.5m 且步率差 >1.2Hz 则临界）。E4⁗ 实测摆幅后可收紧。
- **[推断] 跌倒经 pose_fail 表达**：机制自然但无直接分桶证据（I64 才能终裁）；本报告的「排除法加强」是三段论不是实证。
- **[推断] r20 张力三假设**：r22 只直接检验 (ii)；(i) 需 r23 消融。
- **[作废声明] 0017 §A 表的「漂移终止预期」列**：物理基础（root 平移漂移触发终止）被 E19 否定；0017 的跌倒主导**结论**经本报告排除法反而加强，但其证据结构从「方向性+量级」变为「双重排除+量级」——如实更新。
- **[说明] r22 预注册（note-018）与本报告 I63 矩阵兼容**：worker 的「>>22.9 / 平 23-40」两档即 I63 的前两档；I63 增补第三档（~23）的解读与附加观测。

---

## 来源

**共享证据（commit `2672f0b`，只读）**
- `.repos/note-018.json`（r21 post-mortem + E19 发现 + r22 说明）、`.repos/task-r21.json`/`task-r22.json`/`proj-30.json`

**归档源码（`research/studio/idear/mimickit_ref/`，主干 `88dbd89e` 版）**
- `mimickit_envs_deepmimic_env.py` L724-784（compute_done 全文——本报告 §A 的依据；注意 r22 分支 `159bcd0` 未逐行 diff，终止逻辑自 r4 以来无改动迹象，以冒烟/日志佐证为准）

**本报告计算**
- `research/studio/idear/20261006_termination_semantics.py`（相位失配时间尺度 + 张力框架 + r22 矩阵）

**承接**：idear-0017（I59 分桶→E19 已执行并发现终止集；I60-a→r22 已启动；I60-b r4 复现仍挂起）、dm-results-R007 v1

**缺失证据**：r22 曲线（~09:40）；done 分桶（I64 待插桩）；r21 曲线（大概率不可得）。
