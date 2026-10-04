# idear 报告 0012：R004 判读——ep_len 非单调的配置归因（iters_per_output 连带语义）、E12 截断假设结算、r15 配方

- 报告 ID：idear-ideas-0012
- 日期：2026-10-05（00:50，基于 worker 00:40 发布的 R004）
- 作者：idear（研究与建议角色）
- 关联结果版本：**dm-results-R004-r14-pipeline-closure v1**（digest sha256:4290a60f…，实验 TASK_20261004_089 r14 + r4-r13 链，workspace commit `831f51c`）；消费其 §1（管线闭环）/§3（E4）/§5（prior/E12）/§7（下一步）
- 新增证据：①r4 分支（`26c8326`）与 r14 分支（`ab36e74`）的 smp_x1_agent.yaml **逐项 diff**（本报告§证据 A）；②`base_agent.py` train_model 循环的 test/reset/保存耦合机制链（§证据 B，r14 分支源码 L52-96）；③E12 数值的复合误差算术（§证据 C）
- 状态：建议（非验收）。**含对 idear-0011 截断采样假设的否定结算（I43）与 R004 §7 ep_len 消融方向的具体化（I41）**

---

## 摘要与关联结果

R004 三大结论我方独立复核成立：first-slot 通道闭环（_obs_norm._count=100,270,080 证实训练后权重）；E4 十格平坦 → sim2sim 管线验证通过（worker 的执行器/obs 契约对齐 + 0007-I23/0008-I26 修复经受住真实策略检验）；μ=0.39 与 0010 反解 0.391 一致（棘轮机理直接证实）。剩余主要矛盾：**策略质量**（ep_len 40/300 步、速度 0.56-0.79 低于数据带 0.97-1.52、duty 单脚 0.73-0.92 拖行、slip 0.09-0.81 大方差）。

本报告三项内容：

1. **ep_len 非单调的归因锁定**：r4 与 r14 的训练配置**逐项对比后唯一实质差异是 `iters_per_output` 100→5000**（optimizer=Adam 3e-4、scale=2、task/smp=0.5/1.0、GSI=False 全同）。而该参数在 `base_agent.py` 中耦合三件事（test rollout、全局 envs 重置、model 保存）——**r14 全程从未发生「每 100 iter 的周期性全局 RSI 重置」**。三个 run 按 iters 排序 r4(3487)=95 > r6(1876)=67 > **r14(2288)=40 反常居低**，与该差异方向一致。但须诚实：单 run 无种子重复，跌倒-RSI 频繁（40 步均值）会削弱周期性 RSI 的边际作用，「run 间方差」仍是并列候选——r15 设计应同时消除两者（I41）。
2. **E12 对 0011 的否定结算**：ε̂-err 曲线**平坦**（0.115-0.218 随 t 缓降，无拐点）推翻「高 t 单步放大主导」的预期；partial return L2 随起步 t′（22→45）10.5→61.8 单调放大，复合率 ≈11.5%/步——**发散是多步复合误差，截断采样救援不成立**（我 0011 预注册的否定分支被触发）。GSI/生成路线修复必须走重训侧（Min-SNR-5 加权是最小改动，I43 给 diff）；RL 单步打分不受影响（与 r4-r14 六轮 PASS 一致）。
3. **r15 配方（I41）**：恢复 `iters_per_output=100` 的训练语义 + 把保存逻辑与 test/reset 解耦（保 first-slot 单 .pt）+ 预算 ≥461M 对齐 r4——一次消除「语义差异」与「样本量」两个混淆变量；ep_len 若回 ~95+ 则归因闭合。

---

## 论文与实现证据

**A. r4 vs r14 配置 diff（两分支 raw 拉取，逐项）**

| 配置项 | r4 (`26c8326`) | r14 (`ab36e74`) | 差异 |
|---|---|---|---|
| actor/critic optimizer | Adam 3e-4（两处） | Adam 3e-4 | 无 |
| sds_loss_scale / K | 2 / [22,15,8] | 2 / [22,15,8] | 无 |
| task/smp weight | 0.5 / 1.0 | 0.5 / 1.0 | 无 |
| enable_gsi | False | False | 无 |
| **iters_per_output** | **100** | **5000**（注释：>总 iters，唯一 .pt=最终权重占 SDK 首槽） | **唯一实质差异** |
| save_int_models | （r4 时代默认） | false | 伴生（r11 引入） |

**B. `iters_per_output` 的耦合语义（`base_agent.py` @ `ab36e74`，L52-96，worker 分支源码）**

`output_iter = (iter>0 and iter%ipo==0) or sample_count>=max`（L80）为 True 时依次：①`test_model(32)`——eval 模式 + **`_reset_envs()` 全局重置全部 envs**（L108）+ 32 episodes rollout；②`write_log`；③`_output_train_model` 写 model.pt（L91）；④`_train_return_tracker.reset()` + **再次 `_reset_envs()` 全局重置训练环境**（L94）。
- r4/r6（ipo=100）：每 100 iter（=3200 env-steps/每 env，32 steps×4096 envs/iter）发生一次同步全局 RSI——全部 envs 同时被打断（含未跌倒的）并撒回参考轨迹随机时刻。
- r14（ipo=5000>2288）：训练期间**从未发生**——episode 只按自然终止（跌倒）异步重启。
- 文献机制支持（[R2] EWC Table IV）：No-RSI 训练崩溃（MEL 0.23 vs 16.87）——RSI 是 DeepMimic 家族成立的必要组件；周期性全局 RSI 额外提供「打断病态长 episode + 起点时刻多样性」。
- **反方向的诚实注记**：r14 平均 ep_len 40 步 → 跌倒-RSI 本身频繁（每 env 每 ~40 步），周期性全局 RSI（每 3200 步/次）的边际作用可能有限——见§反证。

**C. E12 复合误差算术（e12_truncation.json 数值再解读）**

- ε̂-err vs t：8→0.218 … 47→0.116（**平坦缓降，无拐点**）——我 0011 表格的 1/√ᾱ 放大系数描述的是「给定 ε̂ 误差到 x0 的传递」，但 ε̂ 误差本身在高 t 并不更大；单步放大不是发散主因。
- partial return L2（起点 t′ 加噪真实窗 → DDIM 到 0，阈值 0.12）：t′=22: 10.5 / 25: 11.9 / 30: 15.3 / 35: 20.0 / 40: 29.2 / 45: 61.8——L2 ≈ 随**回程步数**指数增长，复合率 ≈ exp(ln(61.8/10.5)/(45-22))≈1.074/步（7.4%/步，t′≥25 段）~11.5%/步（22→25 段外推）。
- 结论：每步 ε̂ 固有误差 ~0.12-0.22，多步复合导致任意起点发散；**t′=22（RL 打分区上端）回程 22 步都发散** → 采样链不可用与打分可用（单步）的边界比我 0011 表述的更尖锐：不是「高 t 才坏」，是「多步就坏」。

**D. E4 表型细节（e4_r14_results.json，判读 R004 §3 一致）**
base 3 eps：duty_L 0.73-0.92 / duty_R 0.20-0.50（单脚拖行）、slip 0.185-0.44、attack 27.9-71°（ep3 无触地反弹 n_steps_TDB=0）、v 0.60-0.73。fric 0.6-1.2 与 pd 0.5/2.0 存活 0.73-1.1s 平坦；lat 档 0.82-1.08s 平坦——**执行器/接触参数不敏感，病态不是对齐引起**（R004 结论成立）。

---

## 优先建议

### I41 · r15 配方：恢复训练语义 + 解耦保存 + 预算对齐（一步消除两个混淆）
- **关联结果版本**：dm-results-R004 v1 §7（「ep_len 非单调需消融；r15 计划 iters_per_output 2000」）
- **文件/函数**：`data/agents/smp_x1_agent.yaml`（ipo）；`mimickit/learning/base_agent.py` L80-94
- **观测事实**：§A/B——唯一配置差异 ipo 100→5000 且其耦合 test/全局 reset/保存三件事。
- **机制假设**：周期性全局 RSI（ipo=100 副作用）维持状态分布贴参考轨迹；缺失后 r14 ep_len 反常低。**并列候选**：run 间方差 + r14 预算 300M < r4 461M。
- **竞争解释**：若 r15（语义恢复+461M）ep_len 仍 ~40 → 归因转向平台/镜像差异或算法上限，需 E8 四格消融定论。
- **最小改动或诊断实验**：r15 = ①ipo=100（恢复 r4 全部训练语义）；②保存解耦 diff（保 first-slot，worker 落地）：
  ```diff
  --- a/mimickit/learning/base_agent.py
  +++ b/mimickit/learning/base_agent.py
  @@ train_model:
  -            output_iter = (self._iter > 0 and self._iter % self._iters_per_output == 0) or (self._sample_count >= max_samples)
  +            test_iter = (self._iter > 0 and self._iter % self._iters_per_output == 0)
  +            final_iter = (self._sample_count >= max_samples)
  +            output_iter = test_iter or final_iter
  ...
  -                self._output_train_model(self._iter, out_model_file, int_out_dir)
  +                if final_iter or save_int_models:  # first-slot: only final .pt
  +                    self._output_train_model(self._iter, out_model_file, int_out_dir)
  ```
  （test+全局 reset 照常每 100 iter；model.pt 只在终局写一次。若 `_output_train_model` 内部含日志依赖，保底方案：临时把 out_model_file 指向 int_models/ 目录、final 再写顶层——worker 择一。）③预算 ≥461M（对齐 r4）。
- **对照变量**：r15 vs r14（唯一差异=训练语义+预算）；若想严格单变量，可先跑 r15a=ipo100+461M。
- **预期指标**：ep_len ≥90（r4 量级）→ 归因闭合；ep_len 40-60 → 方差/其他源，转 E8。
- **支持/否定条件**：如上；另看 Smp_Reward 峰值是否同时回到 ~0.45-0.49。
- **成本风险**：低（一次 461M 训练 ≈ r4 时长）；diff 需过 r12 式冒烟（test_info 占位键已在 L69 修复，无重复风险）。
- **worker 的下一步**：r15 启动前合 diff + 冒烟（gm play 一轮）；R002 的「跳过 iter-0 保存」注释保留。

### I42 · E4 病态表型的归因分诊：拖脚/低带速的三个候选与最小判别
- **关联结果版本**：R004 §3（duty 0.92/0.20、v 0.56-0.79、attack 大方差）
- **文件/函数**：`analysis/e4_r14_results.json`（已有）；task 奖励侧（deepmimic_env 的 reward 定义，未拉取）
- **观测事实**：单脚 duty≈0.9 拖行 + 速度低于数据带下限 35% + slip 大方差。
- **机制假设**（三候选）：①训练不充分（ep_len 40 的伴生表型——r15 后可能自愈，**首选等待 r15 再诊断**）；②task 奖励的 root-速度项被「慢速拖行」局部满足（速度低但 pose 项失守，权重失衡）；③先验速度块盲区（0011-I40：vel/ang 重构误差最差）叠加 ①——先验不惩罚慢，task 独撑速度。
- **竞争解释**：duty 不对称也可能源自数据 SI 偏置（R001：run1_s5_seg0 SI=0.119）——但 r14 策略 SI 0.22-0.64 远超数据上限，非直接继承。
- **最小改动或诊断实验**：r15 后复跑 E4：若 duty 对称化 → ①确认；若仍拖脚 → 从 r15 checkpoint 提取 reward 分量曲线（I20 的 task_reward_mean 已生效于 r13+）对照 duty 时间线。
- **对照变量**：r15 vs r14 同格 E4；task/smp 分量曲线。
- **预期指标**：r15 duty_L/R 差 <0.2、v 进入 0.9+。
- **支持/否定条件**：自愈→①；不自愈且 task_reward 曲线与 duty 无相关→③（调 K 或权重）。
- **成本风险**：低（复用管线）。
- **worker 的下一步**：r15 出权重后 E4 复扫，两表并列 R005。

### I43 · E12 结算与 GSI 重训侧修复：Min-SNR-5 一行加权（二期项，不阻塞）
- **关联结果版本**：idear-0011-I36（截断假设，**否定**）；R004 §5
- **文件/函数**：`tinymdm_model.py` forward L161-175（loss 计算）
- **观测事实**：§C——ε̂-err 平坦 + 复合 7.4-11.5%/步，截断救援在所有 t′ 失败。
- **机制假设**：均匀 ε-loss 对高 t 欠训练（[R5] Min-SNR：uniform ε ≡ SNR 加权 → 低噪声主导梯度）；Min-SNR-5 加权 w_t=min{SNR(t),5} 平衡各 t 的有效梯度，文献报 3.4× 收敛且修复 ε 预测的发散。
- **竞争解释**：也可能需要 v-prediction + zero-terminal-SNR 组合（Lin 2305.08891 三件套）——Min-SNR 是最小改动首选。
- **最小改动或诊断实验**：diff（二期，worker 决定）：
  ```diff
  --- a/mimickit/learning/tinymdm/tinymdm_model.py
  +++ b/mimickit/learning/tinymdm/tinymdm_model.py
  @@ forward():  # after computing loss, before return
  -        return loss
  +        snr = self.diffusion_scheduler.alphas_cumprod[timesteps] / (
  +            1 - self.diffusion_scheduler.alphas_cumprod[timesteps])          # (B,)
  +        w = torch.clamp(snr, max=5.0) / 5.0                                  # Min-SNR-5, normalized
  +        per_elem = torch.abs(pred - target.squeeze())                        # l1 per-element
  +        return (w.unsqueeze(-1) * per_elem).mean()
  ```
  （先验重训 200k iter 单卡小时级；验收=E12 partial return 曲线在 t′≤35 达 L2<1.0。）
- **对照变量**：均匀 vs Min-SNR-5（同数据同容量）；验收用 E12 协议复测。
- **预期指标**：partial return 至少右移（安全 t′ 从 <22 → ≥35）。
- **支持/否定条件**：支持=曲线右移；否定=仍全档发散 → 升级 v-prediction+zero-terminal-SNR。
- **成本风险**：中（重训+验收）；**不阻塞一期**（GSI=False）。
- **worker 的下一步**：二期立项时用；一期忽略。

### I44 · r/r_floor 仪表正式启用（I39 落地确认）
- **关联结果版本**：0011-I39；R004 §2（μ=0.39 实测=0010 反解）
- **内容**：r14 已带 I20/I34 日志（r13 起）——r15 报告建议直接以 `raw Sds_Loss`、`L/μ`、`r/r_floor`（floor=exp(−2×0.125/μ)）三线呈现，Smp_Reward 绝对值降为参考列；ep_len/task 曲线为策略质量主仪表（0010 结论的执行化）。
- **worker 的下一步**：R005 表格口径。

---

## 最小验证实验

| ID | 内容 | 依赖 | 预算 | 判据 |
|---|---|---|---|---|
| E14（新） | r15 = ipo100 语义 + 解耦保存 + ≥461M（I41） | 账号余量 | ~6h | ep_len ≥90？Smp_Reward ≥0.45？ |
| E4'（复跑） | r15 权重 E4 十格 + 表型对照（I42） | E14 | 数百 rollout | duty 对称/v≥0.9？ |
| E12'（二期） | Min-SNR-5 重训后 partial return 复测（I43） | 二期 | 重训+CPU | t′≥35 达标？ |

---

## 上下游缺口

1. **账号余量未知**（R004 §7：账号 26 为最后余额，r14 后余量不明）——r15 启动前先查。
2. Isaac 侧渲染 `_build_lights` bug（R004 §4）——Isaac 侧视频交付仍缺，修复需改引擎源码（用户指令的 Isaac 侧一半未完成）。
3. r14 渲染 mp4 的人形复核仍缺（video 服务持续 503；我 00:44 重试仍 503）——像素统计已过，恢复后补一帧级确认即可。
4. task 奖励分项（deepmimic_env 的 pose/vel/root 权重表）未入档——I42 若需 ③ 分诊要拉取。

---

## 反证与不确定性

- **[已观测] 配置唯一差异 / 机制链 / E12 数值 / E4 表型 / μ=0.39**：分支源码与共享 json 直读，可复核。
- **[推断-中置信] ep_len 归因于 ipo 连带语义**：方向由配置 diff + 机制链 + 次序异常（r14 按 iters 应居中却最低）三重支持；但**单 run 无种子重复**，且 r14 平均 ep_len 40 步意味着跌倒-RSI 每 env 每 ~40 步就发生，周期性全局 RSI（每 3200 步）的边际贡献机制上存疑——「run 方差 + 预算差」是同等先验的并列候选。E14 的设计正是让两者同时被消除/暴露。
- **[结算-0011] 截断采样假设否定**：ε̂-err 平坦推翻「单步放大主导」；复合误差 7.4-11.5%/步使任意多步链发散。0011 表格的 1/√ᾱ 系数本身数学正确，但其作为「发散主因」的解释被 E12 数据否定——正确表述：**多步复合是发散主因，单步打分（RL 用法）幸存是因为只走一步**。
- **[推断] E4 attack ep3=None**：n_steps_TDB=0（无触地反弹）→ 无攻角样本，非计算错误；判读时注意样本缺失。
- **[未验证] r15 diff 与 gradmotion 镜像的交互**（save 解耦是否影响 SDK 扫描）——冒烟一轮即可。

---

## 来源

**共享产物（已 fetch 快照）**
- dm-results-R004-r14-pipeline-closure v1（`inputs/studio/dm-results-R004-r14-pipeline-closure/`，sha256:4290a60f…）

**分支源码（raw 拉取，归档 `research/studio/idear/mimickit_ref/`）**
- r14 分支 `ab36e74`：`data/agents/smp_x1_agent.yaml`（全文）、`mimickit/learning/base_agent.py` L52-136（本次新增归档）、`mimickit/learning/ppo_agent.py`
- r4 分支 `26c8326`：`data/agents/smp_x1_agent.yaml`（grep 对比）

**共享数据（commit `831f51c`）**
- `analysis/e4_r14_results.json`（10 cells）、`analysis/e12_truncation.json`、`checkpoints/r14_final.pt`（元数据经 R004 引用）

**计算**：E12 复合率 exp(ln(61.8/10.5)/23)≈1.074/步；对照阈值 0.12。

**承接**：idear-0011（I36 否定结算）、0010（μ 反解被 R004 证实）、0006-I20/0011-I39（日志项已在 r13+ 生效）、dm-results-R001..R004

**缺失证据**：r15 训练曲线；账号 26 余量；video 服务恢复后的 r14 mp4 人形复核；deepmimic_env task 奖励分项表。
