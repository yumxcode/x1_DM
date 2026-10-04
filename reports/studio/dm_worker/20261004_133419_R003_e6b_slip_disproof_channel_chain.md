# R003 (interim): E6b 滑移机制否定 + sim2sim/渲染双链路就绪 + 产物通道事故链与算力阻塞

- 报告 ID: R003-dm-e6b-channel-chain-20261004_1334
- 作者: dm_worker
- 时间: 2026-10-04 13:34 (本地)
- 状态: 已观测（本地实验 + 远端日志）+ 推断（标注）/ 阻塞声明（算力）
- 关联产物: dm-results-R001/R002；dm-ideas-idear-0001..0009
- 实验 ID: TASK_20261004_005 (r4), 012 (r5), 006 (r6), 040 (r7), 053 (r8), 054 (r9 未启动)

## 1. E6b 消滑移重投影测试（idear-0004 I14 测试3 / 0009 I30）——已观测，机制否定

设置（analysis/e6b_reprojection_test.py，本地 CPU 推理）：
- 7 段 pkl → 680 个 H=10 窗口，disc_obs 构造逐位复刻 mimickit jit 函数（amp_env.compute_disc_vel_obs + deepmimic_env.compute_tar_obs，经本地 shim 加载，零复刻误差）
- prior=x1_prior.pt（main 下载，load_state_dict missing=0 unexpected=0）
- ESM_SDS_loss(K=[22,15,8])，对照=原始窗口 vs 支撑相消滑移重投影（root_vel -= 触地脚水平速度）vs noVel 消融

结果（analysis/e6b_reprojection_result.json）：

| 变体 | sds_loss mean | p50 |
|---|---|---|
| baseline | 0.340 | 0.125 |
| 消滑移重投影 | 0.347 (**-2.1%**) | 0.133 |
| noVel（速度全零） | 0.348 | — |
| 支撑相子集 n=424 | 0.351→0.359 (**-2.4%**) | — |

**判定：OTHER SOURCE——停滞 A 的「流形滑移分量→奖励上限」机制未获证实**（idear-0009 I30 否定分支 <10%）。noVel≈baseline 是更强的双重否定：速度特征整体对 SDS loss 贡献极小，滑移（速度域子分量）不可能是 0.20 平台的主因。

对 idear-0009 处置：I29 accepted（判读指南适用于将来 checkpoint）；**I30 accepted 已执行，结果否定其机制假设**（否定≠建议无效——正是其设计的可证伪判据在工作）；I31 路线 α（v3.1 消滑移数据重训）预期收益据此下调。

### 1.1 新头号嫌疑：数据浮空几何偏移（已观测，本报告新发现）

v3 数据平底在 xyber_x1 几何上标定（zmin≈0，R001 实测 0 穿模）；但在**训练几何 x1_train.xml**（IsaacLab USD 源）上 FK 复测：sole zmin ∈ [12.6, 64]mm（p10=13mm，n=34 采样点）——**参考动作在训练物理里整体浮空 ~1.3cm+**。含义（推断）：
- 训练中「触地」实际发生在 z≈1.3-3cm，接触-姿态分布与物理地面系统性错位，可能贡献残余 SDS loss 与 sim2sim 姿态 gap；
- E6b 触地阈值按 x1_train 几何口径取 0.02（70% 帧覆盖）而非 xyber 的 0.008——两套几何标定差是根因，修法=数据重标定（v3.2 地面闭合按 x1_train 几何）或训练资产 sole 对齐。
- 此项为 r5 后训练的第一优先修复候选（先于 idear-0009 路线 β）。

## 2. 渲染双链路就绪（用户指令 2026-10-04 修订：X1 mesh 渲染）——已观测

- **Isaac 侧**（远端，r8+ 脚本 scripts_remote/render_policy_x1.py @ 4399759）：训练结束后 headless viewport 相机录制（enable_cameras），cv2 落盘 logs/x1_smp_policy/x1_policy_render.mp4 → SDK videoUrl 通道上传。未执行（r8 失败于 pip，见 §3）。
- **MuJoCo 侧**（本地）：analysis/build_render_xml.py 生成 x1_render_sim.xml = 训练物理 + xyber 30 mesh 视觉层（body 树 1:1 实证同构，visual geom 无碰撞）+ 灯光 + offscreen 854x480。sim2sim_validate.py `--render` 选项端到端验证：29 帧 mp4 产出，非黑屏像素比 74%（analysis/_smoke_render.mp4）。待真实策略 checkpoint 后按用户指令出正式视频。
- 待办：视频画面最终人形确认（video_inspect 服务 503，像素统计已过；恢复后复核）。

## 3. 产物通道事故链（已观测）与 r4-r9 全记录

| run | 任务 | 结果 | 权重 |
|---|---|---|---|
| r4 | TASK_20261004_005 | PASS 3487iter/457M, Smp_Reward 峰 0.49→0.35 | 丢失（镜像335 只注册首个 .pt） |
| r5 | TASK_20261004_012 | iter 582 被静默杀（Smp_Reward 0.449） | 丢失 + 账号23 余额耗尽 |
| r6 | TASK_20261004_006 | PASS 1876iter/246M, 0.388, ep_len 67.2 | 丢失（同 r4 根因；23 个顶层文件 SDK 检测到但不注册） |
| r7 | TASK_20261004_040 | iter~1100 主动停止 | 丢失预定（git 中继失败：容器无凭证 rc=128） |
| r8 | TASK_20261004_053 | 早期失败：镜像1 容器 pypi SSL EOF（diffusers 装不上） | — |
| r9 | TASK_20261004_054 | **未启动：账号 24 余额耗尽（1002056）** | — |

根因链（跨任务实证，见 research.ruledOutRoutes）：镜像 BJX00000335 SDK 只注册任务首个 .pt（镜像1 老任务注册 20 个实证）；镜像1 无 pypi 网络（已写 CN-mirror fallback @ f591eb0 未及验证）；git push 容器无凭证。

**阻塞声明**：gradmotion 账号池 19-24 全部余额耗尽（最后确认 13:19），远端训练中断。解锁条件：充值或新账号。算法侧证据链完整（r4/r6 两轮 PASS、Smp_Reward 0.003→0.49 修复实证），重训配方已在分支 dm/smp-reward-fix 冻结（@ f591eb0），账号可用后 r9 一键启动。

## 4. 老线策略 sim2sim interim（已观测，参考证据）

x1_policy_10k5.pt（老线 main，228 维同构）三配置（rsi/pd033/老增益全对齐）：全部 0.72-2.14s 跌倒，slip_ratio 0.105-0.161（物理抑制滑移），攻角 65-77°（病态），duty 极端不对称。与老项目未解 sim2sim 一致 + key_bodies 契约差异（5 vs 4）→ 仅作管线验证与滑移对照（idear-0009 已正确解读），非策略结论。

## 5. 判据状态与下一步

- sim2sim 最终结论：**未验证**（新线权重未取回，阻塞于算力/通道）
- 算法修复结论：已验证（r4/r6 双 PASS，奖励 120×抬升，曲线固化 analysis/r4_curves/）
- E6b：已验证（滑移机制否定 + 浮空偏移发现）
- 下一步（按依赖序）：① 算力恢复 → r9（镜像1+CN-mirror 或镜像335+单权重策略）→ 下载 → E4 扫描 + 双侧渲染（用户指令）→ R004；② 阻塞期间：数据浮空重标定方案（v3.2）设计稿

## 6. 证据文件（本工作区提交见 git log）

- analysis/e6b_reprojection_test.py + e6b_reprojection_result.json（§1）
- analysis/build_render_xml.py + x1_render_sim.xml + _smoke_render.mp4（§2）
- analysis/_old_policy_*.json（§4）、r4_curves/r4_iter_points.json（§3 曲线）
- 分支 dm/smp-reward-fix：c012cef（gm_play 通道+渲染调用）、4399759（渲染脚本）、f591eb0（CN-mirror）
