# R010a：二期准备——prior v2 重训配方（Min-SNR-5 + v-prediction）+ 渲染修复

- 报告 ID: dm-results-R010a；作者: dm_worker；时间: 2026-10-07 14:05（本地）
- 性质: **零算力准备 + 单元级验证**（真实重训仍阻塞于账号枯竭，见 R010）
- 工作区: 本报告提交后推送；远端分支 `dm/prior-v2`（888085f+4e2520a）、`dm/render-fix`（1e67272）

---

## 1. 数据充分性终答（修订 4 的闭环）

**已观测**（`analysis/prior_audit.json`，R003）：
- 训练窗 n=680（7 clip 窗口化）；memorization_verdict=**SMOOTHED/GENERALIZED**（未记忆化）；
  NN ratio 2.03e5；self_loop gen=206.9 vs data=0.63。
- 文献预期（research subtask-d3e3ce00）：1.5M 参数扩散模型在 ≤1k 样本上应 90%+ 复制（记忆化）。

**推断（高置信）**：数据量（680 窗）在记忆化阈值内、并非瓶颈本身；瓶颈是**训练目标失衡+
欠训练**——const-ε l1 损失在低噪声步过权重、高 t 步 ε 目标不可学习（文献结论），导致模型
在阈值内数据上仍未拟合（SMOOTHED）。**先修损失侧，数据扩充（更多 clip 重定向）列为 v3 备选。**

## 2. 已完成并验证（单元级）

### 2.1 Min-SNR-gamma 损失权重（dm/prior-v2 @888085f）

- `mimickit/learning/tinymdm/tinymdm_model.py`：`min_snr_gamma` 配置项（**默认关闭=逐字节
  兼容旧损失**）；权重 w_t=clamp(SNR_t,γ) 均值归一（保持损失尺度/lr/梯度裁剪域不变）。
- 单元验证（`analysis/test_minsnr_unit.py`，真实 DDPMScheduler squaredcos_cap_v2 T=50）：
  - T1 默认关闭路径与 F.l1_loss/F.mse_loss **逐字节一致**（回归安全）；
  - T2 加权公式=mean(w·|err|)；T3 mean(w)=1.0（尺度保持，加权/旧损失比 1.001）；
  - T4 高 t 步 25% 被钳制在 γ=5；T6 低 t 权重≥高 t（再平衡方向正确）；
  - T5 v_prediction 目标=scheduler.get_velocity 闭式（√ᾱ·ε−√(1−ᾱ)·x₀）。

### 2.2 v2 重训配置（dm/prior-v2 @4e2520a）

`tinymdm_x1_multi_clip_v2.yaml`：`estimate_mode: v_prediction` + `min_snr_gamma: 5.0` +
`num_iterations: 300_000`（200k→300k 应对 undertrained 判定）。
- 一致性核对（源码级）：`estimate_mode` 同时进入 DDPM 与 DDIM scheduler 的 prediction_type
  （tinymdm_model.py L77/L83）；`ESM_SDS_loss` 走 `ddim_scheduler.step().pred_original_sample`
  （L262-263）+解析式 ε 换算（L115-119）——**prediction-type 无关，SMP 训练侧无需改动**。
- 启动方式：`train_prior_x1.py` 既有 `--cfg_path` 接口换 v2 yaml 即可（容器内路径已核）。

### 2.3 渲染修复（dm/render-fix @1e67272，验证待容器）

`_build_lights`：prim-utils 双回退（isaacsim→isaaclab）均缺失时**降级为默认灯光朝向**而非
崩溃（灯光纯视觉；smoke3 的 RENDER 阶段 traceback 链即此）。**未验证**——需下次容器 run。

## 3. 未验证 / 阻塞

- prior v2 实际重训（300k iters）与 r27 重发：**均阻塞于 gradmotion 账号 19-34 全枯竭**。
- 渲染修复需容器验证。
- v2 重训后的验收：重建 prior_audit 指标（NN ratio/self_loop 应显著下降；E6b 重投影误差地板
  是否破）→ 再进 SMP 主线。

## 4. 解除阻塞后的执行序

1. r27 重发（dm/smp-reward-fix HEAD=1fd7a1b 零改动；r27 是关键路径，先于 prior v2）。
2. prior v2 重训（dm/prior-v2；一次容器 run，产物走 first-slot 通道）→ prior_audit 复测。
3. 若 r27 否定相位机制 → DR（I75/I81）单变量；若确认 → r28 = phase_obs + prior v2 组合。

## 5. 来源

- 本地：`analysis/test_minsnr_unit.py`（T1-T6 PASS 输出）、`analysis/prior_audit.json`、
  `.repos/mk_api/tinymdm_model.py`（补丁后快照）
- 远端：dm/prior-v2（888085f、4e2520a）、dm/render-fix（1e67272）
- 文献：`.meta-agent/research/subtask-d3e3ce00/report.md`（Min-SNR/v-pred/记忆化阈值）
