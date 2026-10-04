import cv2
import numpy as np

cap = cv2.VideoCapture('/Users/yumx/code/x1_DM/analysis/_smoke_render.mp4')
frames = []
while True:
    ret, f = cap.read()
    if not ret:
        break
    frames.append(f)
mid = frames[len(frames) // 2]
gray = cv2.cvtColor(mid, cv2.COLOR_BGR2GRAY)
lap = cv2.Laplacian(gray, cv2.CV_64F).var()
print(f"frames={len(frames)} mid mean={mid.mean():.1f} laplacian_var={lap:.1f}")
cv2.imwrite('/Users/yumx/code/x1_DM/analysis/_render_check2.png', mid)
edges = cv2.Canny(gray, 60, 150).sum() / 1000
print(f"canny edge mass: {edges:.0f}k px")
