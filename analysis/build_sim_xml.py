#!/usr/bin/env python3
"""Build x1_train_sim.xml: x1_mimicKit training MJCF (geometry/inertia/contact
ground truth) + flat ground + gravity + 120Hz timestep for local sim2sim.

Source: x1_mimicKit data/assets/x1/x1.xml @ dm/smp-reward-fix (sha of file
noted in R002 report). Joint springs are stripped at load time by X1Sim.
"""
import re
import sys

SRC = "/Users/yumx/code/x1_DM/.repos/mk_api/x1_train.xml"
DST = "/Users/yumx/code/x1_DM/analysis/x1_train_sim.xml"

txt = open(SRC).read()

# 1) add option + ground plane into worldbody
ground = '''
  <option timestep="0.0083333333" gravity="0 0 -9.81" integrator="implicitfast" />
'''
# mujoco option element must be child of <mujoco>, before worldbody
txt = txt.replace("<worldbody>",
                  ground + "\n  <worldbody>\n"
                  '    <geom name="ground" type="plane" size="0 0 0.05" '
                  'pos="0 0 0" friction="1.0 0.05 0.05" condim="4" '
                  'contype="1" conaffinity="1" rgba="0.3 0.3 0.35 1" />', 1)

# 2) add camera for optional rendering
txt = txt.replace("</worldbody>",
                  '    <camera name="track" pos="0 -3.5 1.1" xyaxes="1 0 0 0 0.3 0.95" />\n'
                  "  </worldbody>", 1)

open(DST, "w").write(txt)
print("wrote", DST, len(txt), "chars")

# quick validation with mujoco
import mujoco
m = mujoco.MjModel.from_xml_path(DST)
print("nq:", m.nq, "nv:", m.nv, "nu:", m.nu, "nbody:", m.nbody)
for jn in ("lumbar_yaw_joint", "left_hip_pitch_joint", "right_ankle_roll_joint"):
    jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, jn)
    print(f"{jn}: id={jid} axis={m.jnt_axis[jid]}")
assert m.nq == 36, m.nq
print("OK")
