"""Patch: I75/I81 domain randomization (v1: friction + mass at reset).

Files (GitHub API commits to dm/smp-reward-fix):
  1. mimickit/engines/isaac_lab_engine.py  - set_dr_config + randomize_dynamics
  2. mimickit/envs/char_env.py             - DR cfg plumb + reset hook
  3. data/envs/smp_x1_env.yaml             - domain_rand block (r28)
  4. args/smp_x1_args.txt                  - 10M smoke budget

Design notes:
  - v1 scope = friction + mass (cleanest physx-view APIs; the two most-cited
    DR terms for contact-solver/dynamics sim2sim gap). motor/push/latency v2.
  - try/except per term + telemetry print (channel lesson: source-side
    injection must be verifiable in smoke logs).
"""
import base64
import subprocess

GH = ['gh', 'api']
REPO = 'repos/yumxcode/x1_mimicKit'
BRANCH = 'dm/smp-reward-fix'
ROOT = '/Users/yumx/code/x1_DM/.repos/mk_api'


def gh(*args, input=None):
    r = subprocess.run(GH + list(args), capture_output=True, text=True,
                       env={'GH_CONFIG_DIR': '/Users/yumx/.config/gh', 'PATH': '/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin'})
    if r.returncode != 0:
        raise RuntimeError(r.stderr[:400])
    return r.stdout.strip()


def put(path, msg, content):
    sha = gh(f'{REPO}/contents/{path}?ref={BRANCH}', '--jq', '.sha')
    b64 = base64.b64encode(content.encode()).decode()
    out = gh('-X', 'PUT', f'{REPO}/contents/{path}',
             '-f', f'message={msg}', '-f', f'content={b64}',
             '-f', f'branch={BRANCH}', '-f', f'sha={sha}',
             '--jq', '.commit.sha')
    print(f'{path} -> {out[:7]}')
    return out


# ---- 1) engine: DR methods (append after set_cmd) ----
src = open(f'{ROOT}/isaac_lab_engine.py').read()
anchor = '''        else:
            assert(False), "Unsupported control mode: {}".format(self._control_mode)
        return

'''
insert = '''        else:
            assert(False), "Unsupported control mode: {}".format(self._control_mode)
        return

    def set_dr_config(self, dr_cfg):
        """I75/I81 domain randomization config (dict or None)."""
        self._dr_cfg = dr_cfg if (isinstance(dr_cfg, dict) and dr_cfg) else None
        return

    def randomize_dynamics(self, env_ids, obj_id):
        """r28 DR v1: per-env friction + mass randomization at reset.

        I75/I81 recipe (Humanoid-Gym Table III core subset). Each term
        guarded: version-dependent IsaacLab APIs must degrade visibly
        (telemetry print) rather than silently (channel lesson #1).
        """
        cfg = getattr(self, "_dr_cfg", None)
        if (cfg is None):
            return
        try:
            env_ids_t = env_ids
        except Exception:
            env_ids_t = env_ids

        applied = []
        obj = self._objs[obj_id]
        try:
            mass_rng = cfg.get("mass_range", None)
            if (mass_rng is not None):
                base_mass = obj.root_physx_view.get_masses().clone()
                n_envs = base_mass.shape[0]
                scale = torch.empty([n_envs, 1], device=base_mass.device)
                scale.uniform_(1.0 + float(mass_rng[0]), 1.0 + float(mass_rng[1]))
                obj.root_physx_view.set_masses(base_mass * scale, env_ids_t)
                applied.append("mass[{},{}]".format(mass_rng[0], mass_rng[1]))
        except Exception as e:
            print(f"[DR] mass randomization FAILED: {e}", flush=True)

        try:
            fric_rng = cfg.get("friction_range", None)
            if (fric_rng is not None):
                props = obj.root_physx_view.get_material_properties().clone()
                n_envs = props.shape[0]
                new_fric = torch.empty([n_envs, 1, 1], device=props.device)
                new_fric.uniform_(float(fric_rng[0]), float(fric_rng[1]))
                props[..., 0] = new_fric  # static friction
                props[..., 1] = new_fric  # dynamic friction
                obj.root_physx_view.set_material_properties(props, env_ids_t)
                applied.append("fric[{},{}]".format(fric_rng[0], fric_rng[1]))
        except Exception as e:
            print(f"[DR] friction randomization FAILED: {e}", flush=True)

        if (applied):
            print(f"[DR] randomized envs n={len(env_ids_t)}: {', '.join(applied)}",
                  flush=True)
        return

'''
assert src.count(anchor) == 1, 'engine anchor'
src = src.replace(anchor, insert)
open(f'{ROOT}/isaac_lab_engine.py', 'w').write(src)
import ast
ast.parse(src)
put('mimickit/engines/isaac_lab_engine.py',
    'r28 DR v1: engine randomize_dynamics (friction+mass per-env at reset, I75/I81 recipe) + set_dr_config; per-term try/except with telemetry print (channel lesson: visible degradation, never silent)',
    src)

# ---- 2) char_env: plumb cfg + reset hook ----
src = open(f'{ROOT}/mimickit_envs_char_env.py').read()
old_init = '''        self._global_obs = env_config["global_obs"]
        self._root_height_obs = env_config.get("root_height_obs", True)
        self._zero_center_action = env_config.get("zero_center_action", False)'''
new_init = old_init + '''
        self._domain_rand_cfg = env_config.get("domain_rand", None)'''
assert src.count(old_init) == 1, 'char init anchor'
src = src.replace(old_init, new_init)

old_reset = '''        self._engine.set_body_vel(env_ids, char_id, 0.0)
        self._engine.set_body_ang_vel(env_ids, char_id, 0.0)
        return'''
new_reset = '''        self._engine.set_body_vel(env_ids, char_id, 0.0)
        self._engine.set_body_ang_vel(env_ids, char_id, 0.0)

        if (self._domain_rand_cfg):
            if (hasattr(self._engine, "set_dr_config") and not getattr(self, "_dr_configured", False)):
                self._engine.set_dr_config(self._domain_rand_cfg)
                self._dr_configured = True
            if (hasattr(self._engine, "randomize_dynamics")):
                self._engine.randomize_dynamics(env_ids, char_id)
        return'''
assert src.count(old_reset) == 1, 'char reset anchor'
src = src.replace(old_reset, new_reset)
open(f'{ROOT}/mimickit_envs_char_env.py', 'w').write(src)
ast.parse(src)
put('mimickit/envs/char_env.py',
    'r28 DR v1: char_env plumb domain_rand cfg from env yaml + randomize_dynamics hook at _reset_char (MRO-safe: char_env._reset_char is the running implementation)',
    src)

# ---- 3) env yaml: domain_rand block ----
src = open(f'{ROOT}/env_head.yaml').read()
src = src.rstrip() + '''

# r28 (I75/I81): domain randomization v1 - friction + mass per-env at reset.
# Targets the persistent MuJoCo sim2sim gap (engine-level contact/dynamics
# differences; E4 five-way 0.78-2.23 non-monotone but 100% MuJoCo falls).
# NOTE: enable_phase_obs stays True (r27R-4 config unchanged otherwise) -
# DR is the single NEW variable vs r27R-4.
domain_rand:
  friction_range: [0.1, 2.0]
  mass_range: [-0.3, 0.3]
'''
open(f'{ROOT}/env_head.yaml', 'w').write(src)
put('data/envs/smp_x1_env.yaml',
    'r28: domain_rand block (friction U[0.1,2.0], mass +/-30%) - Humanoid-Gym Table III core subset; single new variable vs r27R-4',
    src)

# ---- 4) args: 10M smoke ----
args = open(f'{ROOT}/smp_x1_args_r27R4.txt').read()
args = args.replace('370000000', '10000000')
open(f'{ROOT}/smp_x1_args_smoke_dr.txt', 'w').write(args)
put('args/smp_x1_args.txt',
    'r28 smoke: 10M budget to verify DR path (randomize_dynamics telemetry + no crash) before the 370M burn',
    args)

print('ALL PATCHES COMMITTED')
