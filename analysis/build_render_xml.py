#!/usr/bin/env python3
"""Build x1_render_sim.xml: training physics (x1_train_sim.xml: PD/gear/
contact ground truth) + X1 mesh visual geoms (from X1_29DOF xyber serial xml).

The two body trees are identical (30 bodies, same names/order - verified),
so mesh geoms attach 1:1. Visual geoms get contype=0/conaffinity=0 (no
collision). A track camera is added for offscreen rendering.
"""
import re
import sys

XYBER = "/Users/yumx/code/x1_DM/X1_29DOF/mjcf/robot/xyber_x1/xyber_x1_serial.xml"
BASE = "/Users/yumx/code/x1_DM/analysis/x1_train_sim.xml"
OUT = "/Users/yumx/code/x1_DM/analysis/x1_render_sim.xml"

xyber = open(XYBER).read()
base = open(BASE).read()

# 1) asset block from xyber (mesh declarations)
m = re.search(r"<asset>(.*?)</asset>", xyber, re.S)
asset = m.group(1)
MESHDIR = "/Users/yumx/code/x1_DM/X1_29DOF/meshes"
# normalize every mesh file to an absolute path under the real meshes root
asset = re.sub(r'file="(?:\.\./meshes/)?([^"]+)"', rf'file="{MESHDIR}/\1"', asset)

# 2) visual mesh geoms per body from xyber: <geom type="mesh" mesh="X" .../> lines
#    map: body name -> list of geom xml fragments (visual only)
geoms_by_body = {}
for bm in re.finditer(r'<body name="([^"]+)"(.*?)</body>', xyber, re.S):
    pass  # nested bodies break naive regex; do a streaming approach instead

# streaming: track body open/close, collect direct mesh geoms
stack = []
for line in xyber.split("\n"):
    b = re.search(r'<body name="([^"]+)"', line)
    if b:
        stack.append(b.group(1))
        continue
    if "</body>" in line:
        if stack:
            stack.pop()
        continue
    g = re.search(r'<geom type="mesh"[^>]*/>', line)
    if g and stack:
        frag = g.group(0)
        frag = frag.replace('class="collision"', '')
        frag = re.sub(r'contype="[^"]*"', '', frag)
        frag = re.sub(r'conaffinity="[^"]*"', '', frag)
        frag = frag.rstrip("/>") + ' contype="0" conaffinity="0" group="2" />'
        geoms_by_body.setdefault(stack[-1], []).append(frag)

print(f"visual geoms collected for {len(geoms_by_body)} bodies, total "
      f"{sum(len(v) for v in geoms_by_body.values())}")

# 3) inject asset block
assert "<asset>" not in base
base = base.replace("<worldbody>", f"<asset>{asset}</asset>\n  <worldbody>", 1)

# 4) inject mesh geoms into each matching body (after the body opening tag)
out_lines = []
for line in base.split("\n"):
    out_lines.append(line)
    b = re.search(r'<body name="([^"]+)"[^>]*>', line)
    if b and b.group(1) in geoms_by_body:
        indent = line[:len(line) - len(line.lstrip())] + "  "
        for frag in geoms_by_body[b.group(1)]:
            out_lines.append(indent + frag)
base = "\n".join(out_lines)

# 5) nicer camera + offscreen framebuffer sized for 854x480 rendering
base = base.replace(
    'camera name="track" pos="0 -3.5 1.1" xyaxes="1 0 0 0 0.3 0.95"',
    'camera name="track" pos="0 -3.2 1.0" xyaxes="1 0 0 0 0.28 0.96"')
base = base.replace(
    '<option timestep="0.0083333333"',
    '<visual><global offwidth="854" offheight="480" /></visual>\n'
    '  <option timestep="0.0083333333"', 1)
# light must live inside worldbody (after the ground plane injection)
base = base.replace(
    '<geom name="ground"',
    '<light directional="true" pos="0 -2 3" dir="0 0.4 -1" diffuse="0.9 0.9 0.9" '
    'ambient="0.4 0.4 0.4" specular="0.1 0.1 0.1" />\n'
    '    <geom name="ground"', 1)

open(OUT, "w").write(base)
print("wrote", OUT, len(base), "chars")

import mujoco
mm = mujoco.MjModel.from_xml_path(OUT)
print("nq:", mm.nq, "nbody:", mm.nbody, "nmesh:", mm.nmesh, "ngeom:", mm.ngeom)
assert mm.nq == 36
print("OK")
