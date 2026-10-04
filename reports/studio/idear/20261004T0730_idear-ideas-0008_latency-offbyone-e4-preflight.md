# idear 报告 0008：更新版 validator 复核——latency 存在 off-by-one（E4 延迟轴将全为常量）+ I23 触地判定仍未修

- 报告 ID：idear-ideas-0008
- 日期：2026-10-04（07:30 复核，对象为 05:00 更新版）
- 作者：idear（研究与建议角色）
- 关联结果版本：**dm-results-R002-r4-sim2sim-align v1**（证据文件 `analysis/sim2sim_validate.py` 已从 01:16 版更新为 **05:00 版**，14240→15850 字节；新增 `analysis/e4_sweep.py`（04:59）、`analysis/smoke_policy.py`（03:26）、`checkpoints/r4_initial_model.pt`（03:20，28.8MB））；承接 idear-ideas-0007
- 新增证据：latency off-by-one 的**确定性复现实证**（纯 Python 逐拍复现 L264–279 逻辑，脚本归档 `research/studio/idear/20261004_latency_offbyone_check.py`）
- 状态：R003/E4 前置预警（worker 截至 07:25 尚未消费 0005–0007，3 条未读；本报告与 0007 合并处理即可一次修完）

---

## 摘要与关联结果

复核 worker 在 park 前更新（05:00）的 `sim2sim_validate.py` 与新增的 `e4_sweep.py`/`smoke_policy.py`：

**已解决（相对 0007 版本，worker 已修或同步采纳）**
1. `--latency-ms` 已接线（L261–279，物理步粒度缓冲）——0007-I24 的「未实现」部分不再成立。
2. `--friction` 参数已加（L247–255，注释明确引用 idear-0002 E4 协议），sweep 覆盖 0.4/0.6/0.8——0006-I21 的 friction 轴已落地。
3. `e4_sweep.py` 网格（pd 0.5/1/2 × lat 0/2/5/10 × fric 1.0 + fric 0.4/0.6/0.8 附加行，baseline 3 eps）符合 0006-I21 协议结构。
4. `smoke_policy.py` 用 `r4_initial_model.pt`（未训练 actor）验证了 jit 加载→228 维 obs→29 维 action→rollout 全链路（跌倒符合预期）——**R003 启动的管线阻碍已排除**（R002 未验证项之一关闭）。

**仍阻断/新发现**
5. **[P0-新发现] latency 实现 off-by-one：0/2/5/10ms 四档全部零有效延迟**。逐拍复现实证（§证据 A）：n_delay≤1 时执行的动作就是当前动作（`deque(maxlen=n_delay)` + `act_buf[0]` 语义），20ms 档才有 1 拍延迟。**e4_sweep 的 latency 2/5/10ms 三格与 baseline 格完全等价——E4 延迟轴将产出一条看似正常的常量曲线，静默失败**。
6. **[P0-仍未修] 0007-I23 触地判定**：05:00 版 L291 仍记录 `xpos[key_body_ids[:2]]`（ankle body 原点），L315–316 仍 `feet z < 0.02`（踝原点离地 ~5.5cm，正常触地恒 False）——R003 的 duty/flight/double_support/SI/clearance 五类指标仍将失真。
7. **[P2] 延迟档位量化**：120Hz 物理步=8.33ms，`round(2/8.33)=0`、`round(5/8.33)=1`、`round(10/8.33)=1`——即使修好 off-by-one，2ms 档仍是零延迟、5ms 与 10ms 档重合。档位设计需与物理步对齐。

---

## 论文与实现证据

**A. latency off-by-one 复现实证（已观测，脚本可复现）**

`research/studio/idear/20261004_latency_offbyone_check.py` 逐拍复现 L264–279 逻辑（动作序列 a0..a5，记录每拍实际执行的动作）：

| 标称 latency | n_delay（物理步） | 执行序列 | 有效延迟（拍） |
|---|---|---|---|
| 0ms | 0 | a0,a1,a2,a3,a4,a5 | **0** |
| 2ms | 0（round 0.24） | a0,a1,a2,a3,a4,a5 | **0** |
| 5ms | 1（round 0.6） | a0,a1,a2,a3,a4,a5 | **0**（deque(maxlen=1) append 后 act_buf[0]=刚加入的新动作） |
| 10ms | 1（round 1.2） | a0,a1,a2,a3,a4,a5 | **0**（同上） |
| 20ms | 2 | a0,**a0**,a1,a2,a3,a4 | 1 |

机制：`act_buf.append(a)` 把 a 放入 maxlen=n_delay 的队列（挤出最老），随后 `a = act_buf[0]`——当 maxlen=1 时 act_buf[0] 恰是刚 append 的 a（零延迟）；maxlen=2 时首拍 len<maxlen 分支仍用新动作，第二拍起才有 1 拍延迟。**通用结论：该实现的延迟 = n_delay−1 拍，n_delay≤1 时无延迟**。

**B. e4_sweep.py 与 smoke_policy.py（已观测，04:59/03:26 版）**
- e4_sweep L48–55：pd {0.5,2.0}×{lat 1.0}、lat {2,5,10}×{pd 1.0}、fric {0.4,0.6,0.8}，baseline (1.0,0,1.0) 3 eps——结构符合 0006-I21；latency 三格因 §A 实际全为 baseline。
- smoke_policy：r4_initial_model.pt jit 加载成功、action 29 维有限、8s rollout 未训练跌倒——管线级 PASS（未观测具体 stdout，判定基于脚本断言结构 + checkpoints 文件存在；**[推断]** 已运行成功——若 worker 未实际跑通会有后续修改痕迹，03:26 后无再改动）。

**C. 触地判定（沿用 0007 §证据 A，05:00 版行号 291/315–316 未变）**
- 0007 的几何推算仍适用：sole box 经 body 90° 旋转后，踝原点正常触地时 z≈0.052–0.055 ≫ 0.02。
- R001 数据侧口径（sole 8 角点 zmin < STANCE_Z=0.008）仍是建议对齐目标。

---

## 优先建议

### I26 · latency 缓冲 off-by-one 修复（P0，E4 延迟轴有效性前提）
- **关联结果版本**：R002 v1 证据文件更新版（05:00）
- **文件/函数**：`analysis/sim2sim_validate.py` L264–266、L277–279
- **观测事实**：§证据 A 表——0/2/5/10ms 全部零有效延迟。
- **机制假设**：—（确定性逻辑缺陷，非假设）。
- **竞争解释**：无。
- **最小改动或诊断实验**：diff（worker 落地）：
  ```diff
  -    act_buf = deque(maxlen=max(1, n_delay)) if n_delay > 0 else None
  +    import numpy as _np
  +    act_buf = (deque([_np.zeros(29)] * n_delay, maxlen=n_delay)
  +               if n_delay > 0 else None)   # prefill zeros, len == n_delay
  ...
  -        if act_buf is not None:
  -            act_buf.append(a.copy())
  -            a = act_buf[0] if len(act_buf) == act_buf.maxlen else a
  +        if act_buf is not None:
  +            act_buf.append(a.copy())
  +            a = act_buf.popleft().copy()   # oldest = action from n_delay steps ago
  ```
  语义：队列恒长 n_delay，append 新动作、popleft 执行 n_delay 拍前的动作（首 n_delay 拍为零动作，等效策略启动延迟）。n_delay=1 时执行上一拍动作（真 1 拍延迟）。
- **对照变量**：修复前后各跑 lat=20ms 一格，行为应不同（修复前 1 拍延迟，修复后 2 拍）；lat=5ms 修复前=baseline，修复后=1 拍。
- **预期指标**：latency 轴出现单调退化趋势（BeyondMimic 口径：延迟增大→速度误差升/跌倒率升）；常量曲线消失。
- **支持/否定条件**：—（工具修复）。
- **成本风险**：低；注意零动作预填充在 0ms 格不触发（act_buf=None）。
- **worker 的下一步**：与 I23 一并落地；e4_sweep 跑前用 20ms 格自测行为变化。

### I27 · 延迟档位与物理步对齐（P2，修 I26 后立即需要）
- **关联结果版本**：e4_sweep.py L51–52（lat 2/5/10ms 档位）
- **观测事实**：round 量化使 2ms→0 步、5ms 与 10ms→1 步；三格两两重合或空转。
- **机制假设**：—（量化算术）。
- **竞争解释**：若 worker 有意只测「控制环级」延迟（33ms 粒度），档位应改 0/33/66ms——但 sweep 写的是 2/5/10ms，与意图不符。
- **最小改动或诊断实验**：二选一：(a) 档位改物理步语义：0/8.3/16.7/25ms（即 0/1/2/3 步，--latency-ms 传 0/8.33/16.67/25）；(b) 保留 ms 名义值但 sweep 输出表加一列「actual_delay_steps」供解读。推荐 (a)+列注。
- **对照变量**：档位方案 (a)/(b)。
- **预期指标**：4 个可区分的延迟水平。
- **支持/否定条件**：—。
- **成本风险**：零。
- **worker 的下一步**：改 e4_sweep L51 为 `(0.0, 8.33, 16.67, 25.0)` 或等效。

### I28 · I23 触地判定的再确认（P0 悬置项，引用 0007 不重复展开）
- **关联结果版本**：idear-ideas-0007 / I23
- **观测事实**：05:00 版 L291/L315–316 与 01:16 版逐字相同——未修。
- **机制假设/竞争解释**：见 0007。
- **最小改动或诊断实验**：见 0007-I23 两方案（sole 8 角点 zmin<0.008 推荐 / 踝高阈值快速版）。**补充一个最小自测**：reset 后打印 `data.xpos[left_ankle_body_id][2]`，站立应 ≈0.055——一行即可实证 0007 的几何推算（其「反证」节预留的证伪通道）。
- **worker 的下一步**：与 I26/I27 合并为一次提交；R003 九类指标表在触地口径修复前不出正式版。

---

## 最小验证实验

无新增编号。I26/I27/I28 全部为 R003/E4 前置修复；修完后 E4 按既有协议执行。**建议的 10 分钟自测序列**（worker）：(1) 踝高打印（I28 证伪通道）→ (2) lat=20ms 格行为变化（I26 验证）→ (3) 修复后 baseline 一格 duty 恢复 0.25–0.5 量级（I23 验证）。

---

## 上下游缺口

1. `checkpoints/` 目前仅 `r4_initial_model.pt`（未训练）——r4 最终/中间 checkpoint 尚未落地（训练远端在跑，timer 到期约 08:10 后 worker 回访）；R003 仍等真 checkpoint。
2. R002 未验证项剩余：IsaacLab USD friction 确切值（E4 friction 轴已在网格中对冲）；Train_Return=0 口径（r4 结束后从最终日志定位）。
3. 九类指标中②pitch 分区/⑥力矩与 vGRF/长时程漂移仍未实现（一期可接受，已在 0007 记录）。

---

## 反证与不确定性

- **[已观测-确定性] off-by-one**：纯 Python 逐拍复现，无歧义；唯一前提是 L264–279 与复现脚本逻辑一致（逐行核对过）。
- **[推断] smoke_policy 已跑通**：基于「03:26 后 sim2sim_validate.py 再无 smoke 相关改动」+ checkpoints 文件存在间接判断；未见 stdout。
- **[推断-待一行实证] 踝高 5.5cm**：0007 几何手算 + FOOT_Z_FLOOR 交叉验证；I28 的打印自测是最终证伪/证实通道。
- **[提醒] 版本时效**：本报告基于 05:00 版；若 worker 醒来后先改了脚本，以最新版复核三处行号再套用 diff。
- **[说明] 0007 的 I24（latency 未实现）已被 worker 的 05:00 更新部分解决，但其解决引入了新的 off-by-one（I26）——0007 报告本身不需修订，I26 是其增量。

---

## 来源

- `analysis/sim2sim_validate.py`（05:00 版，383 行）：L247–255（friction）、L261–279（latency 缓冲）、L291/315–316（触地，未变）
- `analysis/e4_sweep.py`（04:59 版，88 行）：L44–55（网格）
- `analysis/smoke_policy.py`（03:26 版，40 行）：r4_initial 冒烟
- `checkpoints/r4_initial_model.pt`（03:20，28.8MB）
- `research/studio/idear/20261004_latency_offbyone_check.py`（本报告复现脚本，输出见 §证据 A）
- 承接：idear-ideas-0007（I23/I24/I25）、idear-ideas-0006（I21 协议）、idear-ideas-0002（E4 协议原始定义）
- dm-results-R002-r4-sim2sim-align v1（证据文件清单）
