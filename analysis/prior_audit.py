#!/usr/bin/env python3
"""Prior accuracy & data-sufficiency audit (user directive 2026-10-04).

Questions:
  A. How accurately does the trained TinyMDM prior reconstruct its own
     training data? (per-diffusion-step x0_hat error, per obs block)
  B. What is the generation quality (sample stats vs data stats)?
  C. Memorization vs smoothing: nearest-neighbor distance of generated
     windows to training windows.
  D. Is 92s / 680 windows enough for a 2-layer DiT trained 200k iters
     @ batch 512 (=1.02e8 window draws, ~150k passes/window)?

Setup reuses the E6b shim (jit-identical disc_obs construction) and the
checkpointed normalizer. All local CPU.
"""
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, "/Users/yumx/code/x1_DM/.repos/mimickit_shim/mimickit")
os.chdir("/Users/yumx/code/x1_DM/.repos/mimickit_shim")

from learning.tinymdm.tinymdm_model import TinyMDMModel
from envs import disc_obs_fns

DEVICE = "cpu"
H = 10
K_STEPS = [22, 15, 8]
KEY_BODIES = ("left_ankle_roll_link", "right_ankle_roll_link",
              "left_wrist_roll_link", "right_wrist_roll_link")
N_JOINTS = 29
XML = "/Users/yumx/code/x1_DM/.repos/mk_api/x1_train.xml"
FPS = 30

# per-frame layout: [rel_pos(3), tan6(6), jrot tan6(174), key(12), v(3), w(3)]
BLOCKS = {"rel_pos": (0, 3), "rot6": (3, 9), "jrot174": (9, 183),
          "key12": (183, 195), "vel3": (195, 198), "ang3": (198, 201)}


def build_model():
    cfg = {"T": 50, "loss_type": "l1", "estimate_mode": "epsilon",
           "noise_schedule_mode": "squaredcos_cap_v2", "num_layers": 2,
           "num_attention_heads": 4, "model_ema": True,
           "model_ema_decay": 0.995, "model_ema_steps": 10,
           "model_ema_update_after": 5000, "normalizer_std_clip": 0.2,
           "arch_name": "DiT", "env_config": "data/envs/smp_x1_env.yaml",
           "input_dim": H * 201}
    m = TinyMDMModel(cfg, DEVICE)
    sd = torch.load("/Users/yumx/code/x1_DM/checkpoints/x1_prior.pt",
                    map_location=DEVICE, weights_only=False)
    m.load_state_dict(sd, strict=True)
    m.eval()
    m.obs_normalizer._mean.data = sd["obs_normalizer._mean"].clone()
    m.obs_normalizer._std.data = sd["obs_normalizer._std"].clone()
    for p in m.parameters():
        p.requires_grad = False
    return m


def load_data_windows():
    """Same construction as e6b (jit fns, ref=last frame)."""
    import glob
    import pickle
    import mujoco
    exec(open("/Users/yumx/code/x1_DM/analysis/e6b_reprojection_test.py")
         .read().split("def main()")[0], globals())
    fk = FK(XML)
    clips = sorted(glob.glob(os.path.join(REPO, "x1_retargeted_motion", "*.pkl")))
    wins = []
    for c in clips:
        fr = load_clip(c)
        pos, quat, vel, ang, dof, dof_vel = build_sequences(fr)
        n = len(fr)
        for t0 in range(0, n - H, 4):
            sl = slice(t0, t0 + H)
            key_seq = np.zeros((H, 4, 3), dtype=np.float32)
            jrot_seq = np.zeros((H, 29, 4), dtype=np.float32)
            for k in range(H):
                kp, _ = fk.frame(pos[sl][k], quat[sl][k], dof[sl][k])
                key_seq[k] = kp
                jrot_seq[k] = fk.joint_rots(dof[sl][k])
            w = dict(pos=pos[sl].copy(), quat=quat[sl].copy(),
                     vel=vel[sl].copy(), ang=ang[sl].copy(),
                     key=key_seq, jrot=jrot_seq)
            ref_pos = torch.tensor(w["pos"][-1:])
            ref_rot = torch.tensor(w["quat"][-1:])
            po = disc_obs_fns.compute_tar_obs(
                ref_root_pos=ref_pos, ref_root_rot=ref_rot,
                root_pos=torch.tensor(w["pos"][None]),
                root_rot=torch.tensor(w["quat"][None]),
                joint_rot=torch.tensor(w["jrot"][None]),
                key_pos=torch.tensor(w["key"][None]),
                global_obs=True, root_height_obs=True)
            vo = disc_obs_fns.compute_disc_vel_obs(
                ref_root_rot=ref_rot,
                root_vel=torch.tensor(w["vel"][None]),
                root_ang_vel=torch.tensor(w["ang"][None]),
                dof_vel=torch.zeros(1, H, 29), global_obs=True,
                dof_vel_obs=False)
            o = torch.cat([po, vo], dim=-1).reshape(1, H * 201)
            wins.append(o[0])
    return torch.stack(wins)  # (N, 2010)


def main():
    torch.set_grad_enabled(False)
    m = build_model()
    data = load_data_windows()
    N = len(data)
    print(f"windows: {N}")

    # normalize once
    nd = m.normalize(data.reshape(-1, H, 201)).reshape(N, H * 201)
    nd3 = nd.reshape(N, H, 201)  # block-indexable view

    results = {"n_windows": N}

    # ---------- A. reconstruction x0 error per t ----------
    print("\n[A] reconstruction (add-noise t -> one DDIM step -> x0_hat):")
    rec = {}
    for t in K_STEPS:
        tt = torch.full((N,), t, dtype=torch.long)
        noise = torch.randn_like(nd)
        noised = m.diffusion_scheduler.add_noise(nd, noise, tt)
        pred = m.dmodel(noised, tt)
        re = m.ddim_scheduler.step(pred, t, noised)
        x0h = re.pred_original_sample
        err = (x0h - nd).abs().reshape(N, H, 201)  # normalized units
        per_blk = {k: float(err[:, :, a:b].mean()) for k, (a, b) in BLOCKS.items()}
        rec[t] = {"mean_abs_err_norm": float(err.mean()), "blocks": per_blk}
        print(f"  t={t}: err={err.mean():.4f}  " +
              " ".join(f"{k}={v:.4f}" for k, v in per_blk.items()))
    results["reconstruction"] = rec

    # ---------- B/C. generation quality & memorization ----------
    print("\n[B] generation: 256 windows sampled (correct flat-2010 shape)")
    B = 256
    gen = m.sample_ema(shape=(H * 201,), batch_size=B, device=DEVICE,
                       sampler="ddim", num_inference_steps=50)
    gen = gen.reshape(B, H * 201)
    # stats vs data (normalized space)
    db = {k: (float(nd3[:, :, a:b].mean()), float(nd3[:, :, a:b].std()))
          for k, (a, b) in BLOCKS.items()}
    gb = {k: (float(gen[:, a:b].mean()), float(gen[:, a:b].std()))
          for k, (a, b) in [(k, (a * H, b * H)) for k, (a, b) in BLOCKS.items()]}
    print(f"  {'block':8s} {'data(mean,std)':>18s} {'gen(mean,std)':>18s}")
    for k in BLOCKS:
        print(f"  {k:8s} ({db[k][0]:+.3f},{db[k][1]:.3f})   ({gb[k][0]:+.3f},{gb[k][1]:.3f})")
    results["gen_stats"] = gb
    results["data_stats"] = db

    # memorization: NN distance (subsample for speed)
    print("\n[C] memorization: NN L2 distance of generated vs training windows")
    G = gen[:128]
    D = nd[:400]
    d2 = torch.cdist(G, D)  # (128,400)
    nn_d, _ = d2.min(dim=1)
    # baseline: data-to-data NN (excluding self) for scale
    d3 = torch.cdist(D[:128], D)
    d3.fill_diagonal_(1e9) if d3.shape[0] == d3.shape[1] else None
    dd_nn = d3.min(dim=1).values
    print(f"  gen->data NN dist: mean={nn_d.mean():.3f} p50={nn_d.median():.3f}")
    print(f"  data->data NN dist: mean={dd_nn.mean():.3f} p50={dd_nn.median():.3f}")
    ratio = float(nn_d.mean() / dd_nn.mean())
    results["nn_dist"] = {"gen_to_data": float(nn_d.mean()),
                          "data_to_data": float(dd_nn.mean()), "ratio": ratio}
    verdict_c = ("MEMORIZED (samples are near-copies of training windows)"
                 if ratio < 0.5 else
                 "SMOOTHED/GENERALIZED (samples differ from training data)" if ratio > 1.5 else
                 "PARTIAL overlap")
    print(f"  -> {verdict_c}")
    results["memorization_verdict"] = verdict_c

    # ---------- D. self-loop SDS at correct shape ----------
    print("\n[D] self-loop ESM_SDS_loss on generated windows (correct shape):")
    sl = m.ESM_SDS_loss(gen[:64], t_lst=K_STEPS).mean(dim=-1)
    dl = m.ESM_SDS_loss(nd[:64], t_lst=K_STEPS).mean(dim=-1)
    print(f"  generated: mean={sl.mean():.4f} p50={sl.median():.4f}")
    print(f"  real data: mean={dl.mean():.4f} p50={dl.median():.4f}")
    results["self_loop"] = {"gen": float(sl.mean()), "data": float(dl.mean())}

    # ---------- E. data sufficiency arithmetic ----------
    iters = 200_000
    bs = 512
    draws = iters * bs
    passes = draws / N
    print(f"\n[E] data sufficiency: {draws:.2e} window draws / {N} windows "
          f"= {passes:.0f} passes per window")
    uniq_frames = N * 1  # stride 4 sampling -> ~unique
    dur_s = sum(1 for _ in range(0)) # placeholder
    results["capacity_arithmetic"] = {"draws": draws, "passes_per_window": passes}

    # overall verdict
    recon_err = rec[8]["mean_abs_err_norm"]
    sufficiency = ("DATA SUFFICIENT for current capacity (prior memorizes/"
                   f"overfits: NN ratio {ratio:.2f}) - accuracy is capacity-"
                   "limited, not data-limited" if ratio < 0.7 else
                   "DATA LIKELY INSUFFICIENT or prior undertrained "
                   f"(NN ratio {ratio:.2f})")
    print(f"\nVERDICT: recon_err(t=8)={recon_err:.4f}; {sufficiency}")
    results["verdict"] = sufficiency

    json.dump(results, open("/Users/yumx/code/x1_DM/analysis/prior_audit.json", "w"),
              indent=1, default=float)
    print("saved analysis/prior_audit.json")


if __name__ == "__main__":
    main()
