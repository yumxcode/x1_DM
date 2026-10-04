#!/usr/bin/env python3
"""X1 SMP policy MuJoCo sim2sim validator (dm_worker, I5 protocol).

Aligns with x1_mimicKit training semantics:
  - obs (char_env.compute_char_obs, global_obs=True, root_height_obs=True):
      [root_h(1), root_rot tan6(6), root_vel(3), root_ang_vel(3),
       joint_rot tan6(29*6=174), dof_vel(29), key_pos rel-root(4*3=12)] = 228
  - joint order: X1_DOF_ORDER (depth-first, retarget_g1_x1.py:37)
  - joint_rot obs: per-joint quaternion = axis_angle(axis, dof) -> tan_norm 6D
  - action: position target (rad) at 30 Hz; physics 120 Hz (x1_train.xml semantics)
  - torque: tau = kp*(q*-q) - kd*qdot, kp/kd from data/assets/x1/x1.xml joint
    stiffness/damping (read at training time by isaac_lab_engine.py:870)

Usage:
  python sim2sim_validate.py --policy <model.pt> --xml <x1_train_sim.xml> \
      [--episodes 3] [--duration 20] [--out results.json] [--pd-scale 1.0] \
      [--latency-ms 0] [--render]

Metrics: humanoid_pose_standard 9 families (S-layer), saved to JSON + markdown.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

try:
    import torch
except ImportError:
    torch = None

X1_DOF_ORDER = (
    [f"lumbar_{n}_joint" for n in ("yaw", "roll", "pitch")]
    + [f"left_{n}_joint" for n in ("shoulder_pitch", "shoulder_roll",
                                    "shoulder_yaw", "elbow_pitch", "elbow_yaw",
                                    "wrist_pitch", "wrist_roll")]
    + [f"right_{n}_joint" for n in ("shoulder_pitch", "shoulder_roll",
                                     "shoulder_yaw", "elbow_pitch", "elbow_yaw",
                                     "wrist_pitch", "wrist_roll")]
    + [f"left_{n}_joint" for n in ("hip_pitch", "hip_roll", "hip_yaw",
                                    "knee_pitch", "ankle_pitch", "ankle_roll")]
    + [f"right_{n}_joint" for n in ("hip_pitch", "hip_roll", "hip_yaw",
                                     "knee_pitch", "ankle_pitch", "ankle_roll")]
)
KEY_BODIES = ("left_ankle_roll_link", "right_ankle_roll_link",
              "left_wrist_roll_link", "right_wrist_roll_link")
# home pose from x1_mimicKit smp_x1_env.yaml init_pose: 17 zeros (waist+arms)
# then left leg 6 then right leg 6, matching X1_DOF_ORDER (not MJCF key order)
HOME_QPOS = np.array(
    [0.0] * 17
    + [0.48891, 0.06213, -0.33853, 0.63204, -0.27224, 0.0]
    + [-0.48891, -0.06213, 0.33853, 0.63204, -0.27224, 0.0],
    dtype=np.float64)


# ---------------- quaternion utils (matches mimickit torch_util) ----------------
def quat_mul(a, b):
    x1, y1, z1, w1 = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    x2, y2, z2, w2 = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack([
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2,
        w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    ], axis=-1)


def quat_conj(q):
    out = q.copy() if isinstance(q, np.ndarray) else np.array(q)
    return np.stack([-out[0], -out[1], -out[2], out[3]], axis=-1)


def quat_rotate(q, v):
    w = q[..., 3]
    u = q[..., :3]
    return v + 2.0 * np.cross(u, np.cross(u, v) + w[..., None] * v
                              ) if v.ndim > 1 else v + 2.0 * np.cross(u, np.cross(u, v) + w * v)


def quat_to_tan_norm(q):
    """mimickit torch_util.quat_to_tan_norm (verified L216-226):
      tan  = quat_rotate(q, X_axis) -> rotation matrix column 0
      norm = quat_rotate(q, Z_axis) -> rotation matrix column 2
      returns [R[:,0], R[:,2]] (6D) — NOT the Zhou [R0,R1] variant.
    """
    R = quat_to_mat(q)
    return np.concatenate([R[..., 0], R[..., 2]], axis=-1)


def quat_to_mat(q):
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
        np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
        np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1),
    ], -2)


def axis_angle_to_quat(axis, angle):
    half = 0.5 * angle
    s = np.sin(half)
    return np.array([axis[0] * s, axis[1] * s, axis[2] * s, np.cos(half)])


def expmap_to_quat_xyzw(r):
    theta = np.linalg.norm(r)
    if theta < 1e-12:
        return np.array([0.0, 0.0, 0.0, 1.0])
    axis = r / theta
    return np.concatenate([axis * np.sin(theta / 2), [np.cos(theta / 2)]])


def rsi_init_state(clip_path=None, frame_idx=None, seed=None):
    """RSI (reference state init) per idear-0006 I21: init from a motion
    frame, matching training's rand_reset. Frame = [pos(3), expmap(3), dof(29)].
    Returns (root_pos, quat_xyzw, dof)."""
    import pickle
    if clip_path is None:
        clip_path = os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), "x1_retargeted_motion",
            "x1_run1_subject2_seg0.pkl")
    with open(clip_path, "rb") as f:
        d = pickle.load(f)
    frames = np.asarray(d["frames"], dtype=float)
    rng = np.random.RandomState(seed)
    k = int(rng.randint(0, len(frames))) if frame_idx is None else int(frame_idx)
    fr = frames[k]
    root_pos = fr[0:3].copy()
    quat_xyzw = expmap_to_quat_xyzw(fr[3:6])
    dof = fr[6:35].copy()
    return root_pos, quat_xyzw, dof


# ---------------- MJCF model wrapper ----------------
class X1Sim:
    def __init__(self, xml_path, pd_scale=1.0, gravity_on=True):
        import mujoco
        with open(xml_path) as f:
            txt = f.read()
        # training xml has joint stiffness/damping (spring semantics in MJ);
        # strip them - PD is applied externally to match IsaacLab implicit servo
        txt = self._strip_springs(txt)
        self.model = mujoco.MjModel.from_xml_string(txt)
        self.data = mujoco.MjData(self.model)
        self.mujoco = mujoco

        self.joint_qposadr = []
        self.joint_dofadr = []
        self.kp = np.zeros(29)
        self.kd = np.zeros(29)
        self.joint_axes = np.zeros((29, 3))
        for i, name in enumerate(X1_DOF_ORDER):
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            assert jid >= 0, f"joint {name} not in xml"
            self.joint_qposadr.append(self.model.jnt_qposadr[jid])
            self.joint_dofadr.append(self.model.jnt_dofadr[jid])
            self.joint_axes[i] = self.model.jnt_axis[jid]
        self.joint_qposadr = np.array(self.joint_qposadr)
        self.joint_dofadr = np.array(self.joint_dofadr)

        # kp/kd + effort limit (gear) from the training xml attributes
        self._parse_pd(xml_path, pd_scale)

        # key body ids
        self.key_body_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, b)
                             for b in KEY_BODIES]
        assert all(i >= 0 for i in self.key_body_ids)

        # sole box geoms (I23 fix: contact judged on sole corners, not body
        # origin — ankle origin sits ~4.9-5.0 cm above ground when standing)
        self.sole_geom_ids = []
        for side in ("left", "right"):
            gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM,
                                    f"{side}_ankle_roll_link_sole")
            assert gid >= 0, f"sole geom not found for {side}"
            self.sole_geom_ids.append(gid)
        self.sole_half = self.model.geom_size[self.sole_geom_ids[0]].copy()

    def sole_zmin(self):
        """Per-foot minimum z over the 8 corners of the sole box (R001
        STANCE_Z=0.008 compatible). Returns array(2,)."""
        out = np.zeros(2)
        for i, gid in enumerate(self.sole_geom_ids):
            p = self.data.geom_xpos[gid]
            R = self.data.geom_xmat[gid].reshape(3, 3)
            s = self.model.geom_size[gid]
            zmin = np.inf
            for dx in (-s[0], s[0]):
                for dy in (-s[1], s[1]):
                    for dz in (-s[2], s[2]):
                        c = p + R @ np.array([dx, dy, dz])
                        zmin = min(zmin, c[2])
            out[i] = zmin
        return out

    def sole_center(self):
        """Sole box centers (2,3) for sliding/attack-angle measurement."""
        return np.stack([self.data.geom_xpos[gid].copy()
                         for gid in self.sole_geom_ids])

    @staticmethod
    def _strip_springs(txt):
        """Remove only joint stiffness (spring-to-zero, absent in IsaacLab
        implicit servo). KEEP joint damping: MuJoCo integrates it implicitly
        (implicitfast), which mirrors IsaacLab ImplicitActuator damping and
        avoids the explicit-PD instability for low-armature joints
        (kd*dt/armature > 2 for wrists/elbows at 120 Hz)."""
        import re
        txt = re.sub(r'\s+stiffness="[\d.]+"', "", txt)
        return txt

    def _parse_pd(self, xml_path, pd_scale):
        import re
        txt = open(xml_path).read()
        self.gear = np.ones(29)
        for i, name in enumerate(X1_DOF_ORDER):
            m = re.search(
                rf'<joint name="{name}"[^>]*?stiffness="([\d.]+)"[^>]*?damping="([\d.]+)"',
                txt)
            if m:
                self.kp[i] = float(m.group(1)) * pd_scale
                self.kd[i] = float(m.group(2)) * pd_scale
            else:
                m2 = re.search(rf'<joint name="{name}"([^>]*)>', txt)
                raise RuntimeError(f"PD gains not found for {name}: {m2.group(1) if m2 else 'absent'}")
            mg = re.search(rf'<motor name="motor_{name}"[^>]*?gear="([\d.]+)"', txt)
            if mg:
                self.gear[i] = float(mg.group(1))

    def override_pd_from_xml(self, pd_xml_path, pd_scale=1.0):
        """Replace kp/kd(/gear) tables with those parsed from another MJCF
        (e.g. old-line x1_v4.xml gains) — for cross-line policy evaluation.
        Also rewrites model.dof_damping in place so the implicit damping
        matches the source xml."""
        self._parse_pd(pd_xml_path, pd_scale)
        for i in range(29):
            self.model.dof_damping[self.joint_dofadr[i]] = self.kd[i]

    def qpos2dof(self):
        return self.data.qpos[self.joint_qposadr]

    def qvel2dofvel(self):
        return self.data.qvel[self.joint_dofadr]

    def root_state(self):
        p = self.data.qpos[0:3].copy()
        q = self.data.qpos[3:7].copy()  # wxyz
        q_xyzw = np.array([q[1], q[2], q[3], q[0]])
        # world-frame linear velocity of free joint
        v = self.data.qvel[0:3].copy()
        w = self.data.qvel[3:6].copy()
        return p, q_xyzw, v, w

    def dof_to_joint_rots(self, dof):
        """29 dof -> (29,4) xyzw quats (joint local rotation, mimickit dof_to_rot)."""
        rots = np.zeros((29, 4))
        for i in range(29):
            rots[i] = axis_angle_to_quat(self.joint_axes[i], dof[i])
        return rots

    def build_obs(self, obs_norm=None, a_norm=None):
        p, q, v, w = self.root_state()
        dof = self.qpos2dof()
        dof_vel = self.qvel2dofvel()
        jrots = self.dof_to_joint_rots(dof)

        parts = [p[2:3],
                 quat_to_tan_norm(q),
                 v, w,
                 quat_to_tan_norm(jrots).reshape(-1),
                 dof_vel]
        key_rel = []
        for bid in self.key_body_ids:
            key_rel.append(self.data.xpos[bid] - p)
        parts.append(np.concatenate(key_rel))
        obs = np.concatenate(parts).astype(np.float32)
        assert obs.shape == (228,), obs.shape
        return obs

    def apply_action(self, target_dof):
        """Training-aligned servo. IsaacLab implicit: tau=kp(q*-q)-kd*qdot.
        Here the -kd*qdot part is provided by MuJoCo joint damping (kept in
        the model, integrated implicitly); we apply only the spring term.
        |tau| <= gear matches ImplicitActuatorCfg effort_limit=gear."""
        dof = self.qpos2dof()
        tau = self.kp * (target_dof - dof)
        ctrl = np.clip(tau / self.gear, -1.0, 1.0)
        self.data.ctrl[:] = ctrl

    def reset(self, qpos_dof=None, root_pos=None, root_quat_xyzw=None):
        self.mujoco.mj_resetData(self.model, self.data)
        if root_pos is not None:
            self.data.qpos[0:3] = root_pos
        else:
            self.data.qpos[0:3] = [0, 0, 0.6016]
        if root_quat_xyzw is not None:
            self.data.qpos[3] = root_quat_xyzw[3]
            self.data.qpos[4:7] = root_quat_xyzw[:3]
        if qpos_dof is None:
            qpos_dof = HOME_QPOS
        self.data.qpos[self.joint_qposadr] = qpos_dof
        self.mujoco.mj_forward(self.model, self.data)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True)
    ap.add_argument("--xml", default=os.path.join(os.path.dirname(__file__),
                                                  "x1_train_sim.xml"))
    ap.add_argument("--duration", type=float, default=20.0)
    ap.add_argument("--episodes", type=int, default=1)
    ap.add_argument("--out", default=None)
    ap.add_argument("--pd-scale", type=float, default=1.0)
    ap.add_argument("--latency-ms", type=float, default=0.0)
    ap.add_argument("--friction", type=float, default=1.0,
                    help="ground/foot tangential friction scale (E4 layer 1: "
                         "0.6/0.8/1.0/1.2 sweep per idear-0006 I21 protocol)")
    ap.add_argument("--pd-xml", default=None,
                    help="override kp/kd/gear tables (and model damping) from "
                         "another MJCF, e.g. old-line x1_v4.xml")
    ap.add_argument("--render", action="store_true",
                    help="render rollout to mp4 (X1 mesh via x1_render_sim.xml; "
                         "falls back to plain xml when mesh file unavailable)")
    ap.add_argument("--render-xml", default=os.path.join(
                        os.path.dirname(os.path.abspath(__file__)), "x1_render_sim.xml"))
    ap.add_argument("--video-out", default=None,
                    help="mp4 path (default: alongside --out, or ./sim2sim.mp4)")
    ap.add_argument("--init", choices=["rsi", "home"], default="rsi",
                    help="episode init: motion frame (RSI, training-aligned) "
                         "or home pose (transient-pollution control arm)")
    ap.add_argument("--init-frame", type=int, default=None,
                    help="fixed RSI frame index (default: random per episode)")
    ap.add_argument("--init-clip", default=None, help="pkl clip for RSI init")
    args = ap.parse_args()

    sim = X1Sim(args.xml, pd_scale=args.pd_scale)
    if args.pd_xml:
        sim.override_pd_from_xml(args.pd_xml)
    if args.friction != 1.0:
        # scale tangential friction of all geoms (ground + feet capsules)
        sim.model.geom_friction[:, 0] *= args.friction
    dt_ctrl = 1.0 / 30.0

    renderer = None
    video_out = args.video_out
    if args.render:
        import mujoco as _mj
        rmodel = _mj.MjModel.from_xml_path(args.render_xml)
        rdata = _mj.MjData(rmodel)
        renderer = _mj.Renderer(rmodel, height=480, width=854)
        cam = _mj.MjvCamera()
        cam.lookat[:] = [0.0, 0.0, 0.7]
        cam.distance = 3.2
        cam.elevation = -12
        if video_out is None:
            video_out = (args.out or "sim2sim").replace(".json", "") + ".mp4"
        _vdir = os.path.dirname(os.path.abspath(video_out))
        os.makedirs(_vdir, exist_ok=True)
    n_phys = int(round(dt_ctrl / sim.model.opt.timestep))

    policy = load_policy(args.policy)  # returns callable obs->action (rad)

    # I24 (idear-0007): true actuation latency at PHYSICS-step granularity
    # (8.33 ms). n_delay physics steps of pipeline delay; 0 -> no delay.
    n_delay = int(round(args.latency_ms / 1000.0 / sim.model.opt.timestep))
    from collections import deque
    pipe = deque(maxlen=n_delay + 1) if n_delay > 0 else None

    results = []
    render_frames = []
    for ep in range(args.episodes):
        if args.init == "rsi":
            rp, rq, qd = rsi_init_state(args.init_clip, args.init_frame,
                                         seed=ep if args.init_frame is None else None)
            sim.reset(qpos_dof=qd, root_pos=rp, root_quat_xyzw=rq)
        else:
            sim.reset()  # home pose control arm (quantifies transient pollution)
        obs = sim.build_obs()
        frames = []
        t = 0.0
        done = False
        while t < args.duration and not done:
            a = policy(obs)
            for _ in range(n_phys):
                if pipe is not None:
                    pipe.append(a.copy())
                    a_applied = pipe[0] if len(pipe) == pipe.maxlen else a
                else:
                    a_applied = a
                sim.apply_action(a_applied)
                sim.mujoco.mj_step(sim.model, sim.data)
                t += sim.model.opt.timestep
                if sim.data.qpos[2] < 0.25:
                    done = True
                    break
            obs = sim.build_obs()
            zmin = sim.sole_zmin()
            centers = sim.sole_center()
            frames.append(dict(t=t, root=sim.data.qpos[:3].copy(),
                               dof=sim.qpos2dof().copy(),
                               qvel=sim.qvel2dofvel().copy(),
                               sole_zmin=zmin.copy(),
                               sole_c=[centers[0].copy(), centers[1].copy()],
                               tau_last=sim.data.ctrl.copy()))
            if renderer is not None:
                rdata.qpos[:] = sim.data.qpos
                rdata.qvel[:] = sim.data.qvel
                _mj.mj_forward(rmodel, rdata)
                renderer.update_scene(rdata, camera=cam)
                render_frames.append(renderer.render())
        results.append(analyze_episode(frames, done, t))
        print(f"episode {ep}: dur={t:.2f}s fell={done} "
              f"v_fwd={results[-1]['fwd_vel_mean']:.2f} duty_L={results[-1]['duty_L']:.2f}")

    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
        print("saved", args.out)

    if renderer is not None and render_frames:
        import cv2
        h, w = render_frames[0].shape[:2]
        vw = cv2.VideoWriter(video_out, cv2.VideoWriter_fourcc(*"mp4v"), 30, (w, h))
        for fr in render_frames:
            vw.write(cv2.cvtColor(fr, cv2.COLOR_RGB2BGR))
        vw.release()
        print(f"rendered {len(render_frames)} frames -> {video_out}")
        renderer.close()


def analyze_episode(frames, fell, t_end):
    """9-family S-layer metrics from rollout frames (subset computable here)."""
    import numpy as _np
    roots = _np.array([f["root"] for f in frames])
    zmin = _np.array([f["sole_zmin"] for f in frames])  # (T,2)
    cL = _np.array([f["sole_c"][0] for f in frames])
    cR = _np.array([f["sole_c"][1] for f in frames])
    dof = _np.array([f["dof"] for f in frames])
    dofvel = _np.array([f["qvel"] for f in frames])
    dt = 1.0 / 30.0

    vel = _np.diff(roots, axis=0) / dt
    speed = _np.linalg.norm(vel[:, :2], axis=1)
    # I23 fix: STANCE_Z=0.008 on sole-corner zmin (validate_retarget_v3 L31,
    # R001-compatible). Body-origin threshold was always-False (ankle ~5cm).
    STANCE_Z = 0.008
    contact_L = zmin[:, 0] < STANCE_Z
    contact_R = zmin[:, 1] < STANCE_Z
    flight = (~contact_L) & (~contact_R)
    double = contact_L & contact_R

    # I25: slip ratio (stance horizontal displacement of sole center /
    # body forward displacement) and attack angle at touchdown
    def stance_slip(cxy, contact):
        tot, dur = 0.0, 0
        s = None
        for i, c in enumerate(contact):
            if c and s is None:
                s = i
            elif not c and s is not None:
                if i - s >= 3:
                    tot += float(_np.linalg.norm(cxy[i - 1] - cxy[s]))
                    dur += i - s
                s = None
        if s is not None and len(contact) - s >= 3:
            tot += float(_np.linalg.norm(cxy[-1] - cxy[s]))
            dur += len(contact) - s
        return tot, dur

    slipL, _ = stance_slip(cL[:, :2], contact_L)
    slipR, _ = stance_slip(cR[:, :2], contact_R)
    body_adv = float(_np.linalg.norm(roots[-1, :2] - roots[0, :2])) + 1e-9

    def attack_angles(c3, contact):
        angs = []
        for i in range(1, len(contact)):
            if contact[i] and not contact[i - 1]:
                v = (c3[i] - c3[i - 1]) / dt
                vh = _np.linalg.norm(v[:2])
                if vh > 0.05 and v[2] < -0.01:
                    angs.append(_np.degrees(_np.arctan2(-v[2], vh)))
        return angs

    angs = attack_angles(cL, contact_L) + attack_angles(cR, contact_R)

    r = {
        "duration_s": round(t_end, 2),
        "fell": bool(fell),
        "distance_m": round(float(_np.linalg.norm(roots[-1, :2] - roots[0, :2])), 2),
        "fwd_vel_mean": round(float(speed.mean()), 3),
        "fwd_vel_std": round(float(speed.std()), 3),
        "duty_L": round(float(contact_L.mean()), 3),
        "duty_R": round(float(contact_R.mean()), 3),
        "flight_ratio": round(float(flight.mean()), 3),
        "double_support": round(float(double.mean()), 3),
        "penet_L_mm": round(float(max(0.0, -zmin[:, 0].min())) * 1000, 1),
        "penet_R_mm": round(float(max(0.0, -zmin[:, 1].min())) * 1000, 1),
        "clear_L_cm": round(float(zmin[:, 0].max()) * 100, 1),
        "clear_R_cm": round(float(zmin[:, 1].max()) * 100, 1),
        "slip_ratio": round(float((slipL + slipR) / body_adv), 3),
        "attack_angle_p50": round(float(_np.median(angs)), 1) if angs else None,
        "n_steps_TDB": len(angs),
        "dof_vel_absmax": round(float(_np.abs(dofvel).max()), 1),
        "root_h_min": round(float(roots[:, 2].min()), 3),
        "step_SI": round(abs(contact_L.mean() - contact_R.mean())
                         / (contact_L.mean() + contact_R.mean() + 1e-9), 3),
    }
    return r


def load_policy(path):
    """Load mimickit exported policy checkpoint (state_dict format).

    Structure (verified against ppo_model.py / net_builder / rl_agent, branch
    dm/smp-reward-fix):
      actor forward (deterministic / test mode):
        norm_obs = clamp((obs - obs_mean)/obs_std, inf)   # obs normalizer
        h = relu(l2(relu(l1(norm_obs))))   # 228->1024->512
        norm_a_mean = mean_net(h)          # 512->29
        action = norm_a_mean * a_std + a_mean   # a_norm.unnormalize
      (logstd_net ignored in test mode; FIXED std)

    Returns callable obs(228,) -> action(29,) in radians (position targets).
    """
    assert torch is not None, "torch required for policy inference"
    sd = torch.load(path, map_location="cpu", weights_only=False)

    obs_mean = sd["_obs_norm._mean"].numpy().astype(np.float32)
    obs_std = sd["_obs_norm._std"].numpy().astype(np.float32)
    a_mean = sd["_a_norm._mean"].numpy().astype(np.float32)
    a_std = sd["_a_norm._std"].numpy().astype(np.float32)

    W1 = sd["_model._actor_layers.0.weight"]  # (1024, 228)
    b1 = sd["_model._actor_layers.0.bias"]
    W2 = sd["_model._actor_layers.2.weight"]  # (512, 1024)
    b2 = sd["_model._actor_layers.2.bias"]
    Wm = sd["_model._action_dist._mean_net.weight"]  # (29, 512)
    bm = sd["_model._action_dist._mean_net.bias"]

    def _run(obs):
        x = (obs - obs_mean) / obs_std
        h = np.maximum(x @ W1.T.numpy() + b1.numpy(), 0.0)
        h = np.maximum(h @ W2.T.numpy() + b2.numpy(), 0.0)
        norm_a = h @ Wm.T.numpy() + bm.numpy()
        return (norm_a * a_std + a_mean).astype(np.float32)

    return _run


if __name__ == "__main__":
    sys.exit(main())
