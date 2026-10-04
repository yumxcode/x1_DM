#!/usr/bin/env python3
"""E6b test 3 (idear-0004 I14 / 0009 I30): anti-slip reprojection SDS test.

Question: does the irreducible Sds_Loss plateau (~0.20 in r4/r6 training)
come from the slip component baked into the retargeted data (stance-phase
foot sliding 25-35% of forward progress)?

Method:
  1. Build disc_obs windows (H=10) from the 7 retargeted pkls exactly as
     amp_env.compute_disc_obs does (reusing mimickit jit functions from the
     local shim), with FK key-body positions from MuJoCo.
  2. Baseline: ESM_SDS_loss(K=[22,15,8]) via the trained TinyMDM prior
     (x1_prior.pt) on original windows.
  3. Reprojection: for windows whose LAST frame is in stance (sole zmin <
     8mm), replace root_vel by (root_vel - stance-foot horizontal velocity)
     -> physically attainable velocity field; recompute loss.
  4. Also a no-vel ablation (root_vel=dof_vel=0) as an upper bound probe.

Verdict per idear-0009/I30: loss drop >30% => slip-dominated (stall-A
mechanism confirmed); <10% => another source.
"""
import os
import sys

import numpy as np
import torch

SHIM = "/Users/yumx/code/x1_DM/.repos/mimickit_shim"
sys.path.insert(0, os.path.join(SHIM, "mimickit"))
REPO = "/Users/yumx/code/x1_DM"
os.chdir(SHIM)  # relative asset paths in configs

import mujoco
from util import torch_util
from learning.tinymdm.tinymdm_model import TinyMDMModel
from envs import disc_obs_fns

DEVICE = "cpu"
H = 10          # num_disc_obs_steps (smp_x1_env.yaml)
FPS = 30
K_STEPS = [22, 15, 8]
KEY_BODIES = ("left_ankle_roll_link", "right_ankle_roll_link",
              "left_wrist_roll_link", "right_wrist_roll_link")
N_JOINTS = 29

XML = "/Users/yumx/code/x1_DM/.repos/mk_api/x1_train.xml"


# ---------- data loading (MimicKit pkl -> per-frame tuples) ----------
def load_clip(path):
    import pickle
    with open(path, "rb") as f:
        d = pickle.load(f)
    fr = np.asarray(d["frames"], dtype=np.float32)
    return fr  # [pos(3), expmap(3), dof(29)]


def build_sequences(fr):
    n = len(fr)
    pos = fr[:, 0:3]
    quat_xyzw = np.stack([torch_util_expmap_to_quat(r) for r in fr[:, 3:6]])
    dof = fr[:, 6:35]
    # velocities exactly as motion_lib does: frame_fps * diff
    vel = np.zeros_like(pos)
    vel[:-1] = FPS * (pos[1:] - pos[:-1])
    vel[-1] = vel[-2]
    # root ang vel via quat diff -> expmap / dt
    ang = np.zeros_like(pos)
    for i in range(n - 1):
        d = quat_diff(quat_xyzw[i], quat_xyzw[i + 1])
        ang[i] = d / (1.0 / FPS)
    ang[-1] = ang[-2]
    dof_vel = np.zeros_like(dof)
    dof_vel[:-1] = FPS * (dof[1:] - dof[:-1])
    dof_vel[-1] = dof_vel[-2]
    return pos, quat_xyzw, vel, ang, dof, dof_vel


def torch_util_expmap_to_quat(r):
    theta = np.linalg.norm(r)
    if theta < 1e-12:
        return np.array([0.0, 0.0, 0.0, 1.0])
    axis = r / theta
    return np.concatenate([axis * np.sin(theta / 2), [np.cos(theta / 2)]])


def quat_diff(a, b):
    """quat b * conj(a) -> expmap (matches torch_util.quat_diff + exp_map)."""
    import numpy as _np
    conj = _np.array([-a[0], -a[1], -a[2], a[3]])
    # b * conj(a) via wxyz convention helpers below (xyzw used everywhere)
    w1, x1, y1, z1 = b[3], b[0], b[1], b[2]
    w2, x2, y2, z2 = conj[3], conj[0], conj[1], conj[2]
    q = _np.array([
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2,
        w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    ])
    # expmap
    v = q[:3]
    s = np.linalg.norm(v)
    if s < 1e-12:
        return np.zeros(3)
    angle = 2 * np.arctan2(s, q[3])
    return v / s * angle


# ---------- FK for joint_rot + key_pos via MuJoCo ----------
class FK:
    def __init__(self, xml):
        with open(xml) as f:
            txt = f.read()
        import re
        MESH = "/Users/yumx/code/x1_DM/X1_29DOF/meshes"
        txt = re.sub(r'\s+stiffness="[^"]+"', "", txt)
        self.model = mujoco.MjModel.from_xml_string(txt)
        self.data = mujoco.MjData(self.model)
        self.axes = {}
        self.key_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, b)
                        for b in KEY_BODIES]
        self.sole_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM,
                                           f"{s}_ankle_roll_link_sole")
                         for s in ("left", "right")]
        import json
        order = json.load(open("/Users/yumx/code/x1_DM/analysis/x1_dof_order.json"))
        for i, name in enumerate(order):
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            self.axes[i] = (self.model.jnt_qposadr[jid], self.model.jnt_axis[jid])

    def frame(self, root_pos, quat_xyzw, dof):
        qpos = np.zeros(self.model.nq)
        qpos[0:3] = root_pos
        qpos[3] = quat_xyzw[3]
        qpos[4:7] = quat_xyzw[:3]
        for i in range(N_JOINTS):
            adr, _ = self.axes[i]
            qpos[adr] = dof[i]
        self.data.qpos[:] = qpos
        mujoco.mj_kinematics(self.model, self.data)
        key_pos = np.stack([self.data.xpos[b] for b in self.key_ids])
        sole_v = []
        for gid in self.sole_ids:
            sole_v.append(self.data.geom_xpos[gid].copy())
        return key_pos, np.stack(sole_v)

    def joint_rots(self, dof):
        """(29,4) xyzw local joint quats = axis_angle(axis, q)."""
        out = np.zeros((N_JOINTS, 4), dtype=np.float32)
        for i in range(N_JOINTS):
            _, axis = self.axes[i]
            half = 0.5 * float(dof[i])
            s = np.sin(half)
            out[i] = [axis[0] * s, axis[1] * s, axis[2] * s, np.cos(half)]
        return out


def main():
    # ---------- prior ----------
    cfg = {
        "T": 50, "loss_type": "l1", "estimate_mode": "epsilon",
        "noise_schedule_mode": "squaredcos_cap_v2",
        "num_layers": 2, "num_attention_heads": 4,
        "model_ema": True, "model_ema_decay": 0.995,
        "model_ema_steps": 10, "model_ema_update_after": 5000,
        "normalizer_std_clip": 0.2, "arch_name": "DiT",
        "env_config": "data/envs/smp_x1_env.yaml",
    }
    input_dim = H * 201  # per-frame disc_obs dim (see disc_obs_from_win)
    cfg["input_dim"] = input_dim
    model = TinyMDMModel(cfg, DEVICE)
    sd = torch.load("/Users/yumx/code/x1_DM/checkpoints/x1_prior.pt",
                    map_location=DEVICE, weights_only=False)
    missing = model.load_state_dict(sd, strict=False)
    print("loaded prior; missing keys:", len(missing.missing_keys),
          "unexpected:", len(missing.unexpected_keys))
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    # normalizer stats must come from the checkpoint itself
    model.obs_normalizer._mean.data = sd["obs_normalizer._mean"].clone()
    model.obs_normalizer._std.data = sd["obs_normalizer._std"].clone()
    model.obs_normalizer._count.data = sd["obs_normalizer._count"].clone()

    # ---------- windows ----------
    fk = FK(XML)
    import glob
    clips = sorted(glob.glob(os.path.join(REPO, "x1_retargeted_motion", "*.pkl")))
    torch.set_grad_enabled(False)

    def make_windows(fr):
        pos, quat, vel, ang, dof, dof_vel = build_sequences(fr)
        n = len(fr)
        wins, stance_flags = [], []
        for t0 in range(0, n - H, 4):
            sl = slice(t0, t0 + H)
            key_seq = np.zeros((H, len(KEY_BODIES), 3), dtype=np.float32)
            jrot_seq = np.zeros((H, N_JOINTS, 4), dtype=np.float32)
            soles = np.zeros((H, 2, 3), dtype=np.float32)
            for k in range(H):
                kp, sv = fk.frame(pos[sl][k], quat[sl][k], dof[sl][k])
                key_seq[k] = kp
                jrot_seq[k] = fk.joint_rots(dof[sl][k])
                soles[k] = sv
            wins.append(dict(pos=pos[sl].copy(), quat=quat[sl].copy(),
                             vel=vel[sl].copy(), ang=ang[sl].copy(),
                             dof=dof[sl].copy(), dof_vel=dof_vel[sl].copy(),
                             key=key_seq, jrot=jrot_seq, soles=soles))
            # NOTE: on x1_train.xml geometry the v3 data soles float
            # 1.3-3cm (calibrated on xyber geometry; empirical: zmin in
            # [0.0126, 0.064], p10=0.013). Stance threshold = 0.02 under
            # THIS geometry's convention (70% coverage), NOT the 0.008
            # xyber-calibrated STANCE_Z.
            zmin = np.minimum(soles[-1, 0, 2], soles[-1, 1, 2])
            stance_flags.append(zmin < 0.02)
        return wins, np.array(stance_flags)

    def disc_obs_from_win(w, vel_override=None, dof_vel_override=None):
        """Per-frame layout (VERIFIED vs amp_env/deepmimic_env jit fns):
        [root_rel_pos(3), rot tan6(6), jrot tan6(174), key_pos(12), v(3),
         w(3)] = 201; ref = LAST-frame root (enable_tar_obs=False ->
         _track_global_root()=False); dof_vel_obs=False (env yaml)."""
        rvel = w["vel"] if vel_override is None else vel_override
        dvel = w["dof_vel"] if dof_vel_override is None else dof_vel_override
        ref_pos = torch.tensor(w["pos"][-1:])    # (1,3); jit unsqueezes itself
        ref_rot = torch.tensor(w["quat"][-1:])   # (1,4)
        pos_obs = disc_obs_fns.compute_tar_obs(
            ref_root_pos=ref_pos, ref_root_rot=ref_rot,
            root_pos=torch.tensor(w["pos"][None]),
            root_rot=torch.tensor(w["quat"][None]),
            joint_rot=torch.tensor(w["jrot"][None]),
            key_pos=torch.tensor(w["key"][None]),
            global_obs=True, root_height_obs=True)
        vel_obs = disc_obs_fns.compute_disc_vel_obs(
            ref_root_rot=ref_rot,
            root_vel=torch.tensor(rvel[None]),
            root_ang_vel=torch.tensor(w["ang"][None]),
            dof_vel=torch.tensor(dvel[None]), global_obs=True,
            dof_vel_obs=False)
        o = torch.cat([pos_obs, vel_obs], dim=-1)
        assert o.shape[-1] == 201, o.shape
        return o.reshape(1, -1)

    def sds_loss(obs_flat):
        n = obs_flat.reshape(1, H, 201)
        norm = model.normalize(n).reshape(1, -1)
        losses = model.ESM_SDS_loss(norm_x_obs=norm, t_lst=K_STEPS)
        return losses.mean().item()

    # gather all windows
    all_wins, all_stance = [], []
    for c in clips:
        w, st = make_windows(load_clip(c))
        all_wins += w
        all_stance.append(st)
        print(f"{os.path.basename(c)}: {len(w)} windows, stance-last {st.mean():.2f}")
    all_stance = np.concatenate(all_stance)
    print(f"TOTAL windows: {len(all_wins)}, stance-last ratio: {all_stance.mean():.3f}")

    # baseline loss (all windows, batched)
    def batch_loss(wins, vel_fn=None):
        outs = []
        for w in wins:
            outs.append(disc_obs_from_win(w, *(() if vel_fn is None else vel_fn(w))))
        x = torch.cat(outs, 0)
        losses = []
        for i in range(0, len(x), 256):
            n = x[i:i+256].reshape(len(x[i:i+256]), H, 201)
            norm = model.normalize(n).reshape(len(n), -1)
            losses.append(model.ESM_SDS_loss(norm, t_lst=K_STEPS).mean(dim=-1))
        L = torch.cat(losses)
        return L.numpy()

    base = batch_loss(all_wins)
    print(f"\nBASELINE sds_loss: mean={base.mean():.4f} p50={np.median(base):.4f}")

    # anti-slip reprojection: remove stance-foot slip velocity from root_vel
    def reproject(w):
        soles = w["soles"]  # (H,2,3) sole centers
        svel = np.zeros_like(soles)
        svel[:-1] = FPS * (soles[1:] - soles[:-1])
        svel[-1] = svel[-2]
        contact = np.minimum(soles[:, :, 2].min(axis=1), 1e9) < 0.008
        # per-frame stance foot mask (either foot)
        foot_low = np.minimum(soles[:, 0, 2], soles[:, 1, 2]) < 0.02
        rvel = w["vel"].copy()
        for k in range(H):
            if foot_low[k]:
                f = 0 if soles[k, 0, 2] < soles[k, 1, 2] else 1
                rvel[k, :2] -= svel[k, f, :2]  # remove stance-foot slip velocity
        return (rvel, None)

    def batch_loss_variant(wins, mode):
        outs = []
        for w in wins:
            if mode == "reproject":
                rvel, _ = reproject(w)
                outs.append(disc_obs_from_win(w, rvel, None))
            elif mode == "noVel":
                zeros_v = np.zeros_like(w["vel"])
                zeros_d = np.zeros_like(w["dof_vel"])
                outs.append(disc_obs_from_win(w, zeros_v, zeros_d))
            elif mode == "grounded":
                # v3.2 probe: re-ground each frame on x1_train geometry
                # (root_z -= sole zmin of that frame), velocities recomputed
                soles = w["soles"]
                zmin = np.minimum(soles[:, 0, 2], soles[:, 1, 2])
                shift = zmin  # land each frame
                pos = w["pos"].copy()
                pos[:, 2] -= shift
                vel = np.zeros_like(w["vel"])
                vel[:-1] = FPS * (pos[1:] - pos[:-1])
                vel[-1] = vel[-2]
                # key positions follow root shift (bodies rigid w.r.t. root z)
                key = w["key"].copy()
                key[:, :, 2] -= shift[:, None]
                w2 = dict(w); w2["pos"] = pos; w2["vel"] = vel; w2["key"] = key
                outs.append(disc_obs_from_win(w2))
            else:
                outs.append(disc_obs_from_win(w))
        x = torch.cat(outs, 0)
        losses = []
        for i in range(0, len(x), 256):
            n = x[i:i+256].reshape(len(x[i:i+256]), H, 201)
            norm = model.normalize(n).reshape(len(n), -1)
            losses.append(model.ESM_SDS_loss(norm, t_lst=K_STEPS).mean(dim=-1))
        return torch.cat(losses).numpy()

    rep = batch_loss_variant(all_wins, "reproject")
    print(f"REPROJECTED sds_loss: mean={rep.mean():.4f} p50={np.median(rep):.4f}")
    drop = (base.mean() - rep.mean()) / base.mean()
    print(f"\nloss drop from anti-slip reprojection: {drop*100:.1f}%")

    nov = batch_loss_variant(all_wins, "noVel")
    print(f"noVel ablation sds_loss: mean={nov.mean():.4f} (probe)")

    gro = batch_loss_variant(all_wins, "grounded")
    print(f"GROUNDED sds_loss: mean={gro.mean():.4f} p50={np.median(gro):.4f}")
    gdrop = (base.mean() - gro.mean()) / base.mean()
    print(f"loss drop from v3.2 re-grounding: {gdrop*100:.1f}%")

    # stance-only subset comparison
    if all_stance.sum() > 5:
        idx = np.nonzero(all_stance)[0]
        b = base[idx]; r = rep[idx]
        print(f"\nstance-last windows (n={len(idx)}): base={b.mean():.4f} "
              f"reproj={r.mean():.4f} drop={100*(b.mean()-r.mean())/b.mean():.1f}%")

    verdict = "SLIP-DOMINATED (stall-A mechanism confirmed)" if drop > 0.30 else (
        "OTHER SOURCE (stall-A slip mechanism NOT confirmed)" if drop < 0.10 else
        "AMBIGUOUS (10-30%)")
    print(f"\nVERDICT: {verdict}")

    import json
    json.dump(dict(n_windows=len(all_wins), base_mean=float(base.mean()),
                   reproj_mean=float(rep.mean()), drop=float(drop),
                   noVel_mean=float(nov.mean()),
                   grounded_mean=float(gro.mean()), grounded_drop=float(gdrop),
                   stance_base=float(base[all_stance].mean()) if all_stance.any() else None,
                   stance_reproj=float(rep[all_stance].mean()) if all_stance.any() else None,
                   verdict=verdict),
              open("/Users/yumx/code/x1_DM/analysis/e6b_reprojection_result.json", "w"), indent=1)
    print("saved analysis/e6b_reprojection_result.json")


if __name__ == "__main__":
    main()
