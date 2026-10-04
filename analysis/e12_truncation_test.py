"""E12 (idear-0011 I36): partial-return test - from noised REAL windows at
t' in {45,40,35,30,25}, run full DDIM(50) down to 0 and measure terminal
L2-to-real. Success at max t' = safe truncation point for future GSI.
Also: eps-err vs t curve (I36-1) + gen-stats reshape fix (I38-1)."""
import glob
import json
import os
import pickle
import sys

import numpy as np
import torch

sys.path.insert(0, "/Users/yumx/code/x1_DM/.repos/mimickit_shim/mimickit")
os.chdir("/Users/yumx/code/x1_DM/.repos/mimickit_shim")
from learning.tinymdm.tinymdm_model import TinyMDMModel
from envs import disc_obs_fns

H10 = 201 * 10
DEVICE = "cpu"
torch.set_grad_enabled(False)

# ---- reuse the A/B script's data/model builders via exec of its prefix ----
src = open("/Users/yumx/code/x1_DM/analysis/prior_sampling_ab.py").read()
prefix = src.split("# ---- model ----")[0]
ns = {}
exec(prefix, ns)
FK, build_sequences, expmap_to_quat = ns["FK"], ns["build_sequences"], ns["expmap_to_quat"]
XML = ns["XML"]
REPO = ns["REPO"]

cfg = {"T": 50, "loss_type": "l1", "estimate_mode": "epsilon",
       "noise_schedule_mode": "squaredcos_cap_v2", "num_layers": 2,
       "num_attention_heads": 4, "model_ema": True, "model_ema_decay": 0.995,
       "model_ema_steps": 10, "model_ema_update_after": 5000,
       "normalizer_std_clip": 0.2, "arch_name": "DiT",
       "env_config": "data/envs/smp_x1_env.yaml", "input_dim": H10}
m = TinyMDMModel(cfg, DEVICE)
sd = torch.load("/Users/yumx/code/x1_DM/checkpoints/x1_prior.pt",
                map_location=DEVICE, weights_only=False)
m.load_state_dict(sd, strict=True)
m.eval()
m.obs_normalizer._mean.data = sd["obs_normalizer._mean"].clone()
m.obs_normalizer._std.data = sd["obs_normalizer._std"].clone()

# ---- build 16 real windows from clip 0 (frames 20..180 step 10) ----
fk = FK(XML)
fr = np.asarray(pickle.load(open(sorted(glob.glob(
    "/Users/yumx/code/x1_DM/x1_retargeted_motion/*.pkl"))[0], "rb"),
    encoding="latin1")["frames"], dtype=np.float32)
pos, quat, vel, ang, dof, _ = build_sequences(fr)
reals = []
for t0 in range(20, min(len(fr) - 10, 180), 10):
    w = []
    for k in range(t0, t0 + 10):
        kp, _ = fk.frame(pos[k], quat[k], dof[k])
        w.append(dict(pos=pos[k], quat=quat[k], key=kp,
                      jrot=fk.joint_rots(dof[k]), vel=vel[k], ang=ang[k]))
    ref_pos = torch.tensor(w[-1]["pos"][None])
    ref_rot = torch.tensor(w[-1]["quat"][None])
    po = disc_obs_fns.compute_tar_obs(
        ref_root_pos=ref_pos, ref_root_rot=ref_rot,
        root_pos=torch.tensor(np.stack([x["pos"] for x in w]))[None],
        root_rot=torch.tensor(np.stack([x["quat"] for x in w]))[None],
        joint_rot=torch.tensor(np.stack([x["jrot"] for x in w]))[None],
        key_pos=torch.tensor(np.stack([x["key"] for x in w]))[None],
        global_obs=True, root_height_obs=True)
    vo = disc_obs_fns.compute_disc_vel_obs(
        ref_root_rot=ref_rot,
        root_vel=torch.tensor(np.stack([x["vel"] for x in w]))[None],
        root_ang_vel=torch.tensor(np.stack([x["ang"] for x in w]))[None],
        dof_vel=torch.zeros(1, 10, 29), global_obs=True, dof_vel_obs=False)
    reals.append(torch.cat([po, vo], -1).reshape(1, H10).float())
R = m.normalize(torch.cat(reals, 0).reshape(-1, 10, 201)).reshape(-1, H10)
N = len(R)
print(f"real windows: {N}; normalized std={R.std():.3f}")

# data NN scale for the 3x criterion (from e6b: data-to-data NN ~0.04 per
# window pair L2; use 0.04 as reference scale)
NN_REF = 0.04

results = {"n_windows": N}

# ---- [1] eps-err vs t curve (I36-1) ----
print("\n[1] eps_hat error vs t (64 windows, fresh noise):")
ts_probe = list(range(8, 50, 3))
errs = {}
for t in ts_probe:
    tt = torch.full((N,), t, dtype=torch.long)
    noise = torch.randn_like(R)
    noised = m.diffusion_scheduler.add_noise(R, noise, tt)
    pred = m.ema_dmodel(noised, tt)
    e = (pred - noise).abs().mean().item()
    errs[t] = e
print("  " + " ".join(f"t{t}={v:.3f}" for t, v in errs.items()))
results["eps_err_vs_t"] = errs

# ---- [2] partial return (I36-2) ----
print("\n[2] partial return: noised real @ t' -> DDIM(50-down-from-t') -> L2-to-real")
print(f"  (criterion: L2 < 3 x NN_REF = {3*NN_REF:.3f})")
m.ddim_scheduler.set_timesteps(50, device=DEVICE)
all_ts = m.ddim_scheduler.timesteps  # e.g. 49,48,...,0
part = {}
for t_start in (45, 40, 35, 30, 25, 22):
    tt = torch.full((N,), t_start, dtype=torch.long)
    noised = m.diffusion_scheduler.add_noise(R, torch.randn_like(R), tt)
    x = noised.clone()
    # run DDIM steps from t_start down to 0
    for tt_step in all_ts:
        if int(tt_step) > t_start:
            continue
        tsb = torch.full((N,), int(tt_step), dtype=torch.long)
        out = m.ddim_scheduler.step(m.ema_dmodel(x, tsb), int(tt_step), x)
        x = out.prev_sample
    l2 = (x - R).norm(dim=-1)  # per-window
    part[t_start] = dict(mean=float(l2.mean()), p50=float(l2.median()),
                         max=float(l2.max()))
    ok = l2.median() < 3 * NN_REF
    print(f"  t'={t_start}: L2 mean={l2.mean():.3f} p50={l2.median():.3f} "
          f"max={l2.max():.3f} -> {'OK' if ok else 'FAIL'}")
results["partial_return"] = part

# ---- [3] truncated GSI sample stats (I36-3, only if some t' OK) ----
safe = [t for t in (30, 25, 22) if part.get(t, {}).get("p50", 9e9) < 3 * NN_REF]
print(f"\n[3] truncated sampling from safe t' {safe}")
if safe:
    t_start = safe[0]
    B = 16
    x = torch.randn(B, H10) * np.sqrt(1 - 0)  # start from noise
    # anneal is wrong for truncation: seed with noised REAL windows instead
    # (GSI semantics: noised data-like states)
    seed = m.diffusion_scheduler.add_noise(
        R[:B], torch.randn_like(R[:B]),
        torch.full((B,), t_start, dtype=torch.long))
    x = seed.clone()
    for tt_step in all_ts:
        if int(tt_step) > t_start:
            continue
        tsb = torch.full((B,), int(tt_step), dtype=torch.long)
        out = m.ddim_scheduler.step(m.ema_dmodel(x, tsb), int(tt_step), x)
        x = out.prev_sample
    # block stats with CORRECT frame-major reshape (I38-1)
    BLOCKS = {"rel_pos": (0, 3), "rot6": (3, 9), "jrot174": (9, 183),
              "key12": (183, 195), "vel3": (195, 198), "ang3": (198, 201)}
    g3 = x.reshape(B, 10, 201)
    r3 = R[:B].reshape(B, 10, 201)
    print(f"  {'block':8s} {'data_std':>9s} {'gen_std':>9s} {'ratio':>6s}")
    stats = {}
    for k, (a, b) in BLOCKS.items():
        ds_, gs_ = float(r3[:, :, a:b].std()), float(g3[:, :, a:b].std())
        stats[k] = (ds_, gs_)
        print(f"  {k:8s} {ds_:9.3f} {gs_:9.3f} {gs_/max(ds_,1e-9):6.2f}")
    results["truncated_gen_stats"] = {k: {"data": v[0], "gen": v[1]}
                                      for k, v in stats.items()}
    verdict = ("TRUNCATED SAMPLING VIABLE (all blocks <=3x)" if all(
        v[1] / max(v[0], 1e-9) <= 3 for v in stats.values()) else
        "PARTIAL (some blocks >3x)")
    print(f"  -> {verdict}")
    results["truncated_verdict"] = verdict

json.dump(results, open("/Users/yumx/code/x1_DM/analysis/e12_truncation.json", "w"),
          indent=1, default=float)
print("\nsaved analysis/e12_truncation.json")
