#!/usr/bin/env python3
"""X1 retargeted motion baseline metrics (dm_worker T1).

Input : x1_retargeted_motion/*.pkl  (MimicKit format:
        frames = [root_pos(3), root_expmap(3), dof(29)], fps, time_scale)
Robot : X1_29DOF/mjcf/robot/xyber_x1/xyber_x1_serial.xml
Output: baseline_metrics.json + markdown table on stdout.

Metric families (per humanoid_pose_standard skill):
  (1) swing rhythm symmetry  : hip_pitch L/R antiphase, SI
  (3) velocity tracking      : forward/lateral/yaw decomposition
  (4) foot-ground contact    : attack angle, sliding
  (5) temporal structure     : duty factor, flight, double support
  (6) joint smoothness       : max |qdot|, jerk proxy; joint limits margin
"""
import glob
import json
import os
import pickle
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MOTION_DIR = os.path.join(REPO, "x1_retargeted_motion")
XML = os.path.join(REPO, "X1_29DOF/mjcf/robot/xyber_x1/xyber_x1_serial.xml")
OUT_JSON = os.path.join(REPO, "analysis", "baseline_metrics.json")

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


def expmap_to_quat_xyzw(r):
    theta = np.linalg.norm(r)
    if theta < 1e-12:
        return np.array([0.0, 0.0, 0.0, 1.0])
    axis = r / theta
    return np.concatenate([axis * np.sin(theta / 2), [np.cos(theta / 2)]])


def rotmat_from_expmap(r):
    theta = np.linalg.norm(r)
    if theta < 1e-12:
        return np.eye(3)
    k = r / theta
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)


def load_clip(path):
    with open(path, "rb") as f:
        d = pickle.load(f)
    fr = np.asarray(d["frames"], dtype=float)
    return {
        "name": os.path.basename(path),
        "fps": d["fps"],
        "time_scale": d["time_scale"],
        "loop_mode": d["loop_mode"],
        "s_leg": d.get("s_leg"),
        "s_arm": d.get("s_arm"),
        "pos": fr[:, 0:3],
        "rot": fr[:, 3:6],
        "dof": fr[:, 6:35],
        "n": len(fr),
    }


def fk_feet(mj_model, clip):
    """Return (L_sole_z, R_sole_z, L_sole_xy, R_sole_xy, pitch_deg, roll_deg) arrays."""
    import mujoco

    data = mujoco.MjData(mj_model)
    foot_L_ids = [mj_model.geom(i).id for i in range(mj_model.ngeom)
                  if mj_model.geom(i).name == ""]  # placeholder, use body frame instead
    body_L = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "left_ankle_roll_link")
    body_R = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "right_ankle_roll_link")
    # sole center offset in ankle_roll frame: x=0, y=-0.0408(L), z=0 (mid of +-0.07)
    sole_off_L = np.array([0.0, -0.0408, 0.0])
    sole_off_R = np.array([0.0, 0.0408, 0.0])

    n = clip["n"]
    out = {k: np.zeros((n, 3)) for k in ("L", "R")}
    pitch = np.zeros(n)
    roll = np.zeros(n)
    for i in range(n):
        qpos = np.zeros(mj_model.nq)
        qpos[0:3] = clip["pos"][i]
        q = expmap_to_quat_xyzw(clip["rot"][i])
        qpos[3] = q[3]
        qpos[4:7] = q[0:3]
        qpos[7:] = clip["dof"][i]
        data.qpos[:] = qpos
        mujoco.mj_kinematics(mj_model, data)
        out["L"][i] = data.xpos[body_L] + data.xmat[body_L].reshape(3, 3) @ sole_off_L
        out["R"][i] = data.xpos[body_R] + data.xmat[body_R].reshape(3, 3) @ sole_off_R
        R = rotmat_from_expmap(clip["rot"][i])
        # world z in body frame -> roll/pitch decomposition (ZYX)
        rpy = np.degrees(np.arctan2(R[2, 1], R[2, 2]))  # roll(x)
        pitch[i] = np.degrees(np.arctan2(-R[2, 0], np.hypot(R[2, 1], R[2, 2])))
        roll[i] = rpy
    return out, pitch, roll


def heading_series(pos):
    """Smoothed heading angle from horizontal displacement."""
    d = np.diff(pos[:, :2], axis=0)
    ang = np.unwrap(np.arctan2(d[:, 1], d[:, 0]))
    # moving average over 15 frames
    k = np.ones(15) / 15
    return np.convolve(ang, k, mode="same"), d


def gait_events(z, h=0.02):
    """contact = sole height below h. Returns boolean array."""
    return z < h


def stance_segments(contact):
    segs = []
    start = None
    for i, c in enumerate(contact):
        if c and start is None:
            start = i
        elif not c and start is not None:
            segs.append((start, i))
            start = None
    if start is not None:
        segs.append((start, len(contact)))
    return segs


def analyse_clip(clip, mj_model, froude_h=None):
    fps = clip["fps"]
    dt = 1.0 / fps
    pos, rot, dof = clip["pos"], clip["rot"], clip["dof"]
    res = {
        "file": clip["name"],
        "frames": clip["n"],
        "duration_s": round(clip["n"] / fps, 2),
        "fps": fps,
        "time_scale": clip["time_scale"],
        "s_leg": round(clip["s_leg"], 4) if clip["s_leg"] else None,
    }

    # ---- (3) velocity decomposition
    vel = np.diff(pos, axis=0) / dt
    hdg, _ = heading_series(pos)
    hdg = np.concatenate([[hdg[0]], hdg])
    fwd, lat = np.zeros(len(vel)), np.zeros(len(vel))
    for i in range(len(vel)):
        c, s = np.cos(hdg[i]), np.sin(hdg[i])
        fwd[i] = vel[i, 0] * c + vel[i, 1] * s
        lat[i] = -vel[i, 0] * s + vel[i, 1] * c
    res["fwd_vel_mean"] = round(float(fwd.mean()), 3)
    res["fwd_vel_std"] = round(float(fwd.std()), 3)
    res["lat_drift_ratio"] = round(float(np.abs(lat).mean() / (np.abs(fwd).mean() + 1e-9)), 3)
    yaw_rate = np.diff(hdg) / dt
    res["yaw_rate_mean_deg_s"] = round(float(np.degrees(yaw_rate.mean())), 2)

    # ---- (2) torso
    feet, pitch, roll = fk_feet(mj_model, clip)
    res["root_h_mean"] = round(float(pos[:, 2].mean()), 3)
    res["root_h_min"] = round(float(pos[:, 2].min()), 3)
    res["pitch_deg_mean"] = round(float(pitch.mean()), 1)
    res["pitch_deg_range"] = round(float(pitch.max() - pitch.min()), 1)
    res["roll_deg_std"] = round(float(roll.std()), 2)

    # ---- (5) gait temporal structure
    zL, zR = feet["L"][:, 2], feet["R"][:, 2]
    cL, cR = gait_events(zL), gait_events(zR)
    duty_L = cL.mean()
    duty_R = cR.mean()
    res["duty_L"] = round(float(duty_L), 3)
    res["duty_R"] = round(float(duty_R), 3)
    both = cL & cR
    neither = (~cL) & (~cR)
    res["double_support_ratio"] = round(float(both.mean()), 3)
    res["flight_ratio"] = round(float(neither.mean()), 3)

    # step frequency from hip pitch oscillation (zero crossings)
    lp = dof[:, X1_DOF_ORDER.index("left_hip_pitch_joint")]
    rp = dof[:, X1_DOF_ORDER.index("right_hip_pitch_joint")]
    lp_d = lp - lp.mean()
    zc = np.nonzero(np.diff(np.sign(lp_d)))[0]
    if len(zc) > 2:
        cyc = 2 * (zc[-1] - zc[0]) / (len(zc) - 1) / fps  # s per cycle
        res["step_freq_hz"] = round(1.0 / cyc, 2) if cyc > 0 else None
    else:
        res["step_freq_hz"] = None

    # ---- (1) antiphase symmetry
    corr = np.corrcoef(lp_d, rp - rp.mean())[0, 1] if len(lp) > 3 else None
    res["hip_antiphase_corr"] = round(float(corr), 3) if corr is not None else None
    res["step_SI"] = round(abs(float(duty_L - duty_R)) / (float(duty_L + duty_R) + 1e-9), 3)

    # ---- (4) foot contact details
    # sliding: horizontal displacement during contact / stance duration
    def slide(feet_xy, contact):
        tot, dur = 0.0, 0
        for s, e in stance_segments(contact):
            if e - s < 3:
                continue
            seg = feet_xy[s:e]
            tot += float(np.linalg.norm(seg[-1] - seg[0]))
            dur += e - s
        return tot, dur

    slL, durL = slide(feet["L"][:, :2], cL)
    slR, durR = slide(feet["R"][:, :2], cR)
    res["slide_L_cm"] = round(slL * 100, 1)
    res["slide_R_cm"] = round(slR * 100, 1)
    # penetration
    res["penet_L_mm"] = round(float(max(0.0, -zL.min())) * 1000, 1)
    res["penet_R_mm"] = round(float(max(0.0, -zR.min())) * 1000, 1)
    res["clear_L_cm"] = round(float((zL.max() - zL.min())) * 100, 1)
    res["clear_R_cm"] = round(float((zR.max() - zR.min())) * 100, 1)

    # ---- (6) joint smoothness + limits
    dof_vel = np.diff(dof, axis=0) / dt
    res["dof_vel_absmax_deg_s"] = round(float(np.abs(np.degrees(dof_vel)).max()), 0)
    jerk = np.diff(dof, 2, axis=0) / dt ** 2
    res["dof_jerk_absmax"] = round(float(np.abs(jerk).max()), 0)
    res["knee_rom_deg"] = round(float(np.degrees(
        dof[:, X1_DOF_ORDER.index("left_knee_pitch_joint")].max()
        - dof[:, X1_DOF_ORDER.index("left_knee_pitch_joint")].min())), 1)

    # Froude number & duty expectation
    leg_len = 0.61 + 0.10  # hip to sole approx from keyframe; rough
    v = res["fwd_vel_mean"] / clip["time_scale"]  # un-slowed physical velocity
    fr = v ** 2 / (9.81 * leg_len)
    res["froude_phys"] = round(float(fr), 3)
    res["duty_vs_froude"] = "walk-regime(expected>0.5)" if res["duty_L"] > 0.5 else "run/gait-regime(<0.5)"
    return res


def main():
    import mujoco
    # fix relative meshdir by rewriting to an absolute path before load
    with open(XML) as f:
        xml_txt = f.read()
    mesh_abs = os.path.join(REPO, "X1_29DOF", "meshes")
    xml_txt = xml_txt.replace('meshdir="../meshes"', f'meshdir="{mesh_abs}"')
    mj_model = mujoco.MjModel.from_xml_string(xml_txt)
    results = []
    for p in sorted(glob.glob(os.path.join(MOTION_DIR, "*.pkl"))):
        clip = load_clip(p)
        try:
            r = analyse_clip(clip, mj_model)
        except Exception as e:  # noqa: BLE001
            r = {"file": clip["name"], "error": str(e)}
        results.append(r)
        print(json.dumps(r, ensure_ascii=False))

    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump({"robot_xml": XML, "unit_note": "angles deg, lengths m unless noted",
                   "clips": results}, f, indent=2, ensure_ascii=False)
    print(f"\nsaved -> {OUT_JSON}")

    # markdown summary
    cols = ["file", "frames", "duration_s", "time_scale", "fwd_vel_mean", "duty_L", "duty_R",
            "double_support_ratio", "flight_ratio", "step_freq_hz", "hip_antiphase_corr",
            "step_SI", "slide_L_cm", "penet_L_mm", "clear_L_cm", "root_h_mean",
            "pitch_deg_mean", "froude_phys"]
    print("\n| " + " | ".join(cols) + " |")
    print("|" + "---|" * len(cols))
    for r in results:
        print("| " + " | ".join(str(r.get(c, "-")) for c in cols) + " |")


if __name__ == "__main__":
    sys.exit(main())
