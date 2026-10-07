"""Patch: Min-SNR-gamma loss weighting for TinyMDM (dm/prior-v2 branch).

Backward compatible: absent `min_snr_gamma` key -> byte-identical legacy loss.
Motivation (amendment 4 + R003 prior audit + E12):
  - 680-window dataset, const-weight l1-on-epsilon -> SMOOTHED/undertrained
    prior (NN ratio 2.03e5; self_loop gen 206.9 vs data 0.63)
  - literature (research subtask-d3e3ce00): const-epsilon loss overweights
    low-noise levels; unlearnable eps targets at high t -> Min-SNR-5 + v-pred
"""
SRC = '/Users/yumx/code/x1_DM/.repos/mk_api/tinymdm_model.py'

src = open(SRC).read()

# 1) config read (backward compatible)
old_cfg = '''        self.estimate_mode = config["estimate_mode"]
        self.loss_type = config["loss_type"]'''
new_cfg = '''        self.estimate_mode = config["estimate_mode"]
        self.loss_type = config["loss_type"]
        # Min-SNR-gamma (Hang et al. 2023): clamp per-timestep SNR weight so
        # near-unlearnable high-noise steps stop dominating the gradient.
        # None/absent -> legacy constant weighting (byte-identical loss).
        self.min_snr_gamma = config.get("min_snr_gamma", None)'''
assert src.count(old_cfg) == 1, 'cfg anchor'
src = src.replace(old_cfg, new_cfg)

# 2) weighted loss in forward()
old_loss = '''        if self.loss_type == "l1":
            loss = torch.nn.functional.l1_loss(pred, target.squeeze())
        elif self.loss_type == "l2":
            loss = torch.nn.functional.mse_loss(pred, target.squeeze())
        else:
            raise NotImplementedError '''
new_loss = '''        if (self.min_snr_gamma is not None):
            # Min-SNR-gamma weighting (Hang et al. 2023, "Efficient Diffusion
            # Training via Min-SNR"). w_t = clamp(SNR_t, max=gamma), then
            # mean-normalized so the overall loss scale (and thus the tuned
            # lr / grad-clip regime) is preserved.
            _ap = self.diffusion_scheduler.alphas_cumprod[timesteps]
            _snr = _ap / (1.0 - _ap)
            _w = torch.clamp(_snr, max=self.min_snr_gamma)
            _w = _w / _w.mean()
            if self.loss_type == "l1":
                _elem = (pred - target.squeeze()).abs()
            elif self.loss_type == "l2":
                _elem = (pred - target.squeeze()) ** 2
            else:
                raise NotImplementedError
            loss = (_w.unsqueeze(-1) * _elem).mean()
        elif self.loss_type == "l1":
            loss = torch.nn.functional.l1_loss(pred, target.squeeze())
        elif self.loss_type == "l2":
            loss = torch.nn.functional.mse_loss(pred, target.squeeze())
        else:
            raise NotImplementedError '''
assert src.count(old_loss) == 1, 'loss anchor'
src = src.replace(old_loss, new_loss)

open(SRC, 'w').write(src)
import ast
ast.parse(src)
print('tinymdm_model.py patched (min_snr_gamma, default off)')
