"""I48 level-2: decisive local test of the view-overwrite semantics in
SMPAgent._compute_rewards (branch smp_agent_head.py line order).

Replicates: task_r = get_data_flat("reward") [VIEW]; r = 0.5*task + 1.0*smp;
set_data_flat("reward", r) [IN-PLACE buf[:] = r]; info task_mean computed
AFTER the overwrite."""
import torch

buf = torch.zeros(100)
task_r = buf  # view semantics (get_data_flat returns self._flat_buffers[name])
smp_r = torch.full((100,), 0.4)

for env_task in (0.7, 0.0):
    task_r[:] = env_task
    r = 0.5 * task_r + 1.0 * smp_r
    buf[:] = r  # set_data_flat in-place
    logged = torch.mean(task_r)
    print(f"env task={env_task}: logged task_reward_mean={logged:.4f} "
          f"(smp_mean=0.4) -> identity at 3dp? {abs(logged - 0.4) < 0.0005}")
