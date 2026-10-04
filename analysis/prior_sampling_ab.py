"""Prior sampling A/B: locate why sample_ema produces out-of-distribution
garbage while scoring/denoising works."""
import glob
import os
import pickle
import sys

import numpy as np
import torch

sys.path.insert(0, "/Users/yumx/code/x1_DM/.repos/mimickit_shim/mimickit")
os.chdir("/Users/yumx/code/x1_DM/.repos/mimickit_shim")
from learning.tinymdm.tinymdm_model import TinyMDMModel
from envs import disc_obs_fns

H = 201 * 10
DEVICE = "cpu"
torch.set_grad_enabled(False)

# ---- e6b prelude (FK etc.) without nested-exec hazards ----
import mujoco  # noqa: F401
import json as _json
REPO = "/Users/yumx/code/x1_DM"
XML = "/Users/yumx/code/x1_DM/.repos/mk_api/x1_train.xml"
N_JOINTS = 29
X1_DOF_ORDER = _json.load(open("/Users/yumx/code/x1_DM/analysis/x1_dof_order.json"))


def expmap_to_quat(r):
    theta = np.linalg.norm(r)
    if theta < 1e-12:
        return np.array([0.0, 0.0, 0.0, 1.0])
    axis = r / theta
    return np.concatenate([axis * np.sin(theta / 2), [np.cos(theta / 2)]])


class FK:
    def __init__(self, xml):
        import re
        txt = open(xml).read()
        txt = re.sub(r'\s+stiffness="[^"]+"', "", txt)
        self.model = mujoco.MjModel.from_xml_string(txt)
        self.data = mujoco.MjData(self.model)
        self.jadr = {}
        self.axes = {}
        for i, name in enumerate(X1_DOF_ORDER):
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            self.jadr[i] = self.model.jnt_qposadr[jid]
            self.axes[i] = self.model.jnt_axis[jid]
        self.key_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, b)
                        for b in ("left_ankle_roll_link", "right_ankle_roll_link",
                                  "left_wrist_roll_link", "right_wrist_roll_link")]
        self.sole_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM,
                                           f"{s}_ankle_roll_link_sole")
                         for s in ("left", "right")]

    def frame(self, root_pos, quat_xyzw, dof):
        qpos = np.zeros(self.model.nq)
        qpos[0:3] = root_pos
        qpos[3] = quat_xyzw[3]
        qpos[4:7] = quat_xyzw[:3]
        for i in range(N_JOINTS):
            qpos[self.jadr[i]] = dof[i]
        self.data.qpos[:] = qpos
        mujoco.mj_kinematics(self.model, self.data)
        return np.stack([self.data.xpos[b] for b in self.key_ids]), \
            np.stack([self.data.geom_xpos[g] for g in self.sole_ids])

    def joint_rots(self, dof):
        out = np.zeros((N_JOINTS, 4), dtype=np.float32)
        for i in range(N_JOINTS):
            axis = self.axes[i]
            half = 0.5 * float(dof[i])
            s = np.sin(half)
            out[i] = [axis[0] * s, axis[1] * s, axis[2] * s, np.cos(half)]
        return out


FPS = 30


def build_sequences(fr):
    n = len(fr)
    pos = fr[:, 0:3]
    quat = np.stack([expmap_to_quat(r) for r in fr[:, 3:6]])
    dof = fr[:, 6:35]
    vel = np.zeros_like(pos)
    vel[:-1] = FPS * (pos[1:] - pos[:-1])
    vel[-1] = vel[-2]
    ang = np.zeros_like(pos)
    for i in range(n - 1):
        b, a = quat[i + 1], quat[i]
        conj = np.array([-a[0], -a[1], -a[2], a[3]])
        w1, x1, y1, z1 = b[3], b[0], b[1], b[2]
        w2, x2, y2, z2 = conj[3], conj[0], conj[1], conj[2]
        q = np.array([
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2,
            w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2])
        v, s = q[:3], np.linalg.norm(q[:3])
        ang[i] = (v / s * (2 * np.arctan2(s, q[3]))) if s > 1e-12 else 0
    ang[-1] = ang[-2]
    dof_vel = np.zeros_like(dof)
    dof_vel[:-1] = FPS * (dof[1:] - dof[:-1])
    dof_vel[-1] = dof_vel[-2]
    return pos, quat, vel, ang, dof, dof_vel


# ---- model ----
cfg = {"T": 50, "loss_type": "l1", "estimate_mode": "epsilon",
       "noise_schedule_mode": "squaredcos_cap_v2", "num_layers": 2,
       "num_attention_heads": 4, "model_ema": True, "model_ema_decay": 0.995,
       "model_ema_steps": 10, "model_ema_update_after": 5000,
       "normalizer_std_clip": 0.2, "arch_name": "DiT",
       "env_config": "data/envs/smp_x1_env.yaml", "input_dim": H}
m = TinyMDMModel(cfg, DEVICE)
sd = torch.load("/Users/yumx/code/x1_DM/checkpoints/x1_prior.pt",
                map_location=DEVICE, weights_only=False)
m.load_state_dict(sd, strict=True)
m.eval()
m.obs_normalizer._mean.data = sd["obs_normalizer._mean"].clone()
m.obs_normalizer._std.data = sd["obs_normalizer._std"].clone()

# ---- one real window ----
fk = FK(XML)
fr = np.asarray(pickle.load(open(sorted(glob.glob(
    "/Users/yumx/code/x1_DM/x1_retargeted_motion/*.pkl"))[0], "rb"),
    encoding="latin1")["frames"], dtype=np.float32)
pos, quat, vel, ang, dof, _ = build_sequences(fr)
w = []
for k in range(10):
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
real = torch.cat([po, vo], -1).reshape(1, H).float()
real_n = m.normalize(real.reshape(-1, 10, 201)).reshape(1, H)
print(f"real normalized: mean={real_n.mean():+.3f} std={real_n.std():.3f} "
      f"absmax={real_n.abs().max():.1f}")


def stats(x, tag):
    d = (x.reshape(1, -1) - real_n.reshape(1, -1)).norm().item()
    print(f"  {tag:12s} mean={x.mean():+.3f} std={x.std():7.3f} "
          f"absmax={x.abs().max():8.1f} L2-to-real={d:9.2f}")


print("\n[1] ema / ddim50:")
stats(m.sample_ema(shape=(H,), batch_size=4, device=DEVICE, sampler="ddim",
                   num_inference_steps=50)[0:1], "ema/ddim")
print("[2] ema / ddpm:")
stats(m.sample_ema(shape=(H,), batch_size=4, device=DEVICE, sampler="ddpm")[0:1],
      "ema/ddpm")
print("[3] raw / ddim50:")
stats(m.sample(shape=(H,), batch_size=4, device=DEVICE, sampler="ddim",
               num_inference_steps=50)[0:1], "raw/ddim")
print("[4] raw / ddpm:")
stats(m.sample(shape=(H,), batch_size=4, device=DEVICE, sampler="ddpm")[0:1],
      "raw/ddpm")

print("\n[5] return test: noised real at t=49 -> full DDIM(50)")
t = torch.full((1,), 49, dtype=torch.long)
noised = m.diffusion_scheduler.add_noise(real_n, torch.randn_like(real_n), t)
print(f"  noised start: std={noised.std():.2f}")
x = noised.clone()
m.ddim_scheduler.set_timesteps(50, device=DEVICE)
for tt in m.ddim_scheduler.timesteps:
    ts = torch.full((1,), int(tt), dtype=torch.long)
    out = m.ddim_scheduler.step(m.ema_dmodel(x, ts), int(tt), x)
    x = out.prev_sample
stats(x, "returned")

print("\n[6] single x0_pred from noised real @ t=49")
out = m.ddim_scheduler.step(m.ema_dmodel(noised, t), 49, noised)
stats(out.pred_original_sample, "x0_pred@49")
