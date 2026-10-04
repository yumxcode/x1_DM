"""I23 fix validation: contact detection on sole corners must register
standing contact (duty~1.0 pre-topple) which the old body-origin threshold
(z<0.02 vs ankle ~5cm) could never detect."""
import sys
sys.path.insert(0, '/Users/yumx/code/x1_DM/analysis')
import numpy as np
from sim2sim_validate import X1Sim, HOME_QPOS

sim = X1Sim('/Users/yumx/code/x1_DM/analysis/x1_train_sim.xml')
n_phys = int(round((1/30.0) / sim.model.opt.timestep))
sim.reset(qpos_dof=HOME_QPOS)

zmins, contacts = [], []
for step in range(int(1.0 * 30)):  # 1s before topple develops
    for _ in range(n_phys):
        sim.apply_action(HOME_QPOS)
        sim.mujoco.mj_step(sim.model, sim.data)
    z = sim.sole_zmin()
    zmins.append(z.copy())
    contacts.append(z < 0.008)

Z = np.array(zmins); C = np.array(contacts)
print(f"sole zmin L: mean={Z[:,0].mean():.4f} min={Z[:,0].min():.4f} max={Z[:,0].max():.4f}")
print(f"contact rate L={C[:,0].mean():.3f} R={C[:,1].mean():.3f} (expect ~1.0 standing)")
print(f"flight={((~C[:,0])&(~C[:,1])).mean():.3f} double={ (C[:,0]&C[:,1]).mean():.3f}")
ok = C[:,0].mean() > 0.9 and C[:,1].mean() > 0.9
print("I23 FIX VERIFIED" if ok else "STILL BROKEN")
