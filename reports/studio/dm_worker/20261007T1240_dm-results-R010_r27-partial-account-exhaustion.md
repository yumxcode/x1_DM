# R010-partial：r27 判定实验中断（47%）+ 账号池全枯竭阻塞 + 四连杀复盘

- 报告 ID: dm-results-R010
- 作者: dm_worker；时间: 2026-10-07 12:40（本地）
- 状态: **BLOCKED on compute**（gradmotion 账号 19-34 全部耗尽，r27 无法完成）
- 上游: R009（r23 tier 裁决）；idear-0021（I74-I77）、0022（I78-I81）、0023（I82）、0024（I83-I86）处置见 §5
- 工作区: @ 01ebff7（本报告及 r27 曲线）；远端分支 dm/smp-reward-fix @ 1fd7a1b（r27 全量配置，随时可重发）

---

## 1. 结论摘要

1. **r27（phase_obs 单变量判定实验）于 47%（218M/461M）被账号 34 余额击杀**——第 12 次 balance kill，
   无 traceback、日志截断于正常训练行（`.repos/r27final.json`：iter 1662, test_ret 14.6, 分桶 live）。
2. **全部 16 个账号（19-34）耗尽**（balance-probe 34 返回 1002056；池刷新后无新账号）。
   关键路径上所有实验（r27 完成、DR 训练）均阻塞。
3. r27 部分证据（47% 窗口 191-218M）：分桶 share 平坦在 19.5-19.8%（对比 r25 非相位基线 17.3-17.5%、
   r26@232M 20.8%）——**未出现预注册的 40%+ 信号，相位机制假设保持 OPEN**（fresh-start 使早期轨迹
   更慢，47% 处 test_ret 14.6-17 vs r26 warm-start 同点 15.1，不可比性已在 §4 声明）。
4. **基础设施全部就绪且经冒烟验证**：229 维 phase_obs 训练路径（smoke3 四要素 PASS）+
   sim2sim 229 维 phase 通道（ckpt 维数自动检测、228 向后兼容验证）——新账号到账后可 5 分钟内重发 r27。
5. 项目最佳存档成果不变：r23（EVAL ep_len 74.88，Isaac tier A；MuJoCo 存活比 gap 2.23 未解）。

## 2. 已观测（证据）

### 2.1 四连杀链（r24-r27，全部因余额 SIGKILL）

| run | 任务 | 账号 | 起止 | 死点 | 预算完成度 | 关键损失 |
|---|---|---|---|---|---|---|
| r24 | TASK_20261006_093 | 31 | 10-06 | 25M | 4%（600M 计划） | 无实质（部署期） |
| r25 | TASK_20261006_105 | 32 | 17:13→22:21 | 440M | 73% | 600M 上界点+终值相变概率测量 |
| r26 | TASK_20261006_118 | 33 | 23:50→02:34 | 232M | 50% | 461M 第二种子（I76 裁决） |
| r27 | TASK_20261007_021 | 34 | 06:49→09:22 | 218M | 47% | phase_obs 判定（I61 终审） |

四杀共性：status=6、无 traceback、日志末行是正常训练表格行——SIGKILL 特征与
1002056 余额错误码（balance-probe 实证：31/32/33/34 四账号全返回）。

### 2.2 r27 部分曲线（`analysis/r27_buckets.json`，206 行，日志窗口 191-218M）

- 分桶 share（Fail_Low_Root/Fail_Count）：19.8%→19.5%（平坦，无趋势）
- smp 0.375-0.417 / task 0.342-0.368（双通道健康，Task≠Smp）
- test_ret 14.6-17.0（fresh-start 无 warm-start，预期低于 warm-start 线）
- 被杀时 iter 1662 / 1665-iter 满额约 3532（461M/29.7k/iter）

### 2.3 smoke 链验证记录（r27 中断前的完整验证）

- smoke2（TASK_20261007_017）：229 维 phase_obs 模型 vs 228 维 staged warm-start 权重
  size-mismatch 崩溃——**冒烟抓住了会烧掉 461M 的缺陷**（通道教训第 5 次命中）。
- smoke3（TASK_20261007_020）：@9700a00（warm-start 默认关闭）四要素 PASS：
  无训练崩溃 / Fail 列 live（948,491 fails@10M, share 16.4%）/ Task≠Smp（0/77 恒等）/
  RESULT PASS（EVAL 10.26 @10M fresh-start 预期低值）；traceback 仅 RENDER 阶段
  （`_build_lights`/isaacsim 缺模块，r14 起已知，不影响权重）。

### 2.4 sim2sim 229 维就绪（commit 5beacc0）

- `analysis/sim2sim_validate.py`：ckpt `_obs_norm._mean` 维数自动检测（229→phase 通道 ON）；
  phase=clip(t_ref/motion_len,0,1)（CLAMP 语义，x1 loop_mode=0）；拼接顺序按
  compute_deepmimic_obs 的 [char_obs, phase_obs] append 语义（`mk_api/mimickit_envs_deepmimic_env.py` 拼接段）。
- 向后兼容：r23 权重（228 维）复跑 dur/v 与 R009 一致（1.07s/0.65、1.02s/0.63）。

## 3. 推断（中置信）

1. r27 若跑完，share 大概率不会在末 27% 突然跳到 40%+（r25/r26 的 share 终生平坦）——
   但 fresh-start 的相位学习可能后置（phase 通道利用需要 critic 先收敛），47% 不足以排除。
2. 账号批次模式：用户按 3 个一批补充（25/26、29/30、31/32/33、34）；下一批到账后
   r27 重发（@1fd7a1b 零改动）是唯一关键路径动作。
3. r26/r25 双点 share（20.8%@232M warm vs 17.3%@440M warm）与 r27（19.5%@218M fresh）
   差异均在 warm/fresh 与进度混杂内，无机制信息。

## 4. 未验证 / 缺失证据

- 相位机制判定（r27 未完成）——预注册三档（share→40%+ 且 ep_len 升=确认 / share 平坦=否定 /
  ep_len≥90=验收档）均无法裁决。
- MuJoCo gap 趋势第五点（r27 权重不存在，E4 无法跑）。
- fresh-start vs warm-start 的严格对照（r23 线均 warm-start；r27 fresh——若重发，判定解释需带此注）。
- Isaac 侧渲染（`_build_lights` 容器缺 isaacsim 模块，遗留）。

## 5. idear 处置记录（0023/0024）

| 建议 | 处置 | 证据/理由 |
|---|---|---|
| I82(a) Isaac 构型偏差时序导出 | deferred | 需完成的 r 系 run；被算力阻塞 |
| I83 r26/r27 排序=phase_obs 首选 | accepted-已执行 | r27 即此路径（@1fd7a1b） |
| I84 sim2sim 229 维契约清单 | accepted-已执行 | §2.4 全四项（维数/编码/reset 同步/prior 断言） |
| I85 r25 终值三档判读 | moot（r25 被杀@73%） | 平台三现（25.2/25.7/25.4）已并入 R009/R010 叙述 |
| I86 相变机制备忘 | noted | r25 曲线已存（r25_buckets.json）；深挖等算力 |
| I73 commit 回显 | partially-verified | 列存在间接证明部署；回显行未在抓取窗口出现（非关键） |

## 6. 解除阻塞后的动作清单（优先级序）

1. **重发 r27**（零改动：账号配额+`gm task create/run`，配置 @ 1fd7a1b）→ 完成后按预注册三档判定。
2. 若 r27 ≥90：E4 全量+mesh 渲染（Isaac 用本地 mp4 链、MuJoCo 用 229 维通道）→ R011 验收评估。
3. 若 r27 否定相位机制：DR（I75/I81 配方已备）为下一单变量。
4. （二期）prior 重训侧：Min-SNR-5/v-prediction（E12 结论）。

## 7. 来源

- `.repos/r25final.json`/`r26final.json`/`r27final.json`/`smoke3final.json`（任务日志原始抓取）
- `analysis/r25_buckets.json`（@59904ad）、`analysis/r27_buckets.json`（@01ebff7）
- 远端分支 dm/smp-reward-fix：5748941（phase_obs True）→ 9700a00（warm-start off）→ 1fd7a1b（461M 全量）
- 本地 `analysis/sim2sim_validate.py` @ 5beacc0（229 维通道+兼容验证）
