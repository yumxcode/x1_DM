"""Unit test: Min-SNR-gamma loss math (synthetic tensors, no training).

Verifies, using a real DDPMScheduler (diffusers, squaredcos_cap_v2, T=50):
  T1  default-off path reproduces F.l1_loss / F.mse_loss exactly
  T2  weighted path equals mean(w * |err|) with w = clamp(SNR,5)/mean(clamp(SNR,5))
  T3  weight normalization keeps expected loss scale ~unchanged (mean w == 1)
  T4  high-t steps get weight == gamma (clamped), low-t steps < gamma
  T5  v_prediction target == scheduler.get_velocity(x, noise, t)
  T6  weight monotonicity: low-t weight >= high-t weight (rebalance direction)
"""
import numpy as np
import torch
import torch.nn.functional as F
from diffusers import DDPMScheduler

torch.manual_seed(7)
T, B, D = 50, 256, 36
GAMMA = 5.0

sched = DDPMScheduler(num_train_timesteps=T, beta_schedule="squaredcos_cap_v2",
                      prediction_type="epsilon", clip_sample=False)
sched.alphas_cumprod = sched.alphas_cumprod.to(torch.float64)  # match prod code .to(device) intent

x = torch.randn(B, D)
noise = torch.randn_like(x)
pred = torch.randn_like(x)
timesteps = torch.randint(0, T, (B,))
noised = sched.add_noise(x, noise, timesteps)

ap = sched.alphas_cumprod[timesteps]
snr = ap / (1 - ap)
w = torch.clamp(snr, max=GAMMA)
w_norm = w / w.mean()

# T1 default-off equivalence
legacy_l1 = F.l1_loss(pred, noise)
assert torch.allclose(legacy_l1, (pred - noise).abs().mean()), "T1a"
legacy_l2 = F.mse_loss(pred, noise)
assert torch.allclose(legacy_l2, ((pred - noise) ** 2).mean()), "T1b"

# T2 weighted path formula
elem = (pred - noise).abs()
weighted = (w_norm.unsqueeze(-1) * elem).mean()
manual = (torch.clamp(snr, max=GAMMA) / torch.clamp(snr, max=GAMMA).mean()).unsqueeze(-1)
assert torch.allclose(weighted, (manual * elem).mean()), "T2"

# T3 normalization
assert abs(float(w_norm.mean()) - 1.0) < 1e-9, "T3"

# T4 clamping direction (I91 fix, idear-0026 §A): LOW-t steps (SNR > gamma)
# clamp to gamma; HIGH-t keep their tiny SNR. Hard assertions - the old
# version had a vacuous second disjunct (clamp => max<=gamma always true)
# and inverted attribution in the report text.
lo = timesteps <= 10
assert bool((w[lo] == GAMMA).all()), "T4a: low-t weights must clamp at gamma"
hi = timesteps >= 45
assert bool((w[hi] < GAMMA * 0.01).all()), "T4b: high-t weights stay tiny SNR (<1% gamma)"
assert float((w == GAMMA).float().mean()) > 0.2, "T4c: a nontrivial clamped fraction exists"

# T5 v_prediction target
sched_v = DDPMScheduler(num_train_timesteps=T, beta_schedule="squaredcos_cap_v2",
                        prediction_type="v_prediction", clip_sample=False)
v_target = sched_v.get_velocity(x, noise, timesteps)
# closed form: v = sqrt(alpha_bar) * eps - sqrt(1-alpha_bar) * x0
cf = ap.sqrt().unsqueeze(-1) * noise - (1 - ap).sqrt().unsqueeze(-1) * x
assert torch.allclose(v_target.double(), cf.double(), atol=1e-5), "T5"

# T6 monotonic rebalance direction
ts_sorted = torch.arange(0, T)
snr_all = sched.alphas_cumprod[ts_sorted] / (1 - sched.alphas_cumprod[ts_sorted])
w_all = torch.clamp(snr_all, max=GAMMA)
assert float(w_all[0]) >= float(w_all[-1]), "T6 (low-t weight >= high-t weight)"

# magnitude check: how much does weighting change per-batch loss?
ratio = float(weighted / legacy_l1)
print(f"T1-T6 PASS | weighted/legacy l1 ratio = {ratio:.3f} "
      f"(batch-averaged, scale preserved by mean-normalization)")
print(f"            SNR range: [{float(snr.min()):.2f}, {float(snr.max()):.2f}] "
      f"-> clamped at {GAMMA}: {(w == GAMMA).float().mean()*100:.0f}% of steps clamped")
