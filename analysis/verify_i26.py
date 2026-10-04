"""I26 self-test (idear-0008 protocol): tick-by-tick semantics of the
latency pipeline as implemented in sim2sim_validate.py HEAD (107cd69).

Expected (post-fix): delay == n_delay physics steps after a warm-up of
n_delay steps (previously the interim version delayed only n_delay-1).
"""
import numpy as np
from collections import deque


def pipeline(actions, n_delay):
    """Replica of HEAD implementation (physics-step granularity)."""
    pipe = deque(maxlen=n_delay + 1) if n_delay > 0 else None
    applied = []
    for a in actions:
        for _ in range(1):  # one physics step per action here (1:1 for clarity)
            if pipe is not None:
                pipe.append(a.copy())
                a_applied = pipe[0] if len(pipe) == pipe.maxlen else a
            else:
                a_applied = a
            applied.append(a_applied.copy())
    return applied


# distinct unit actions a0..a7
acts = [np.array([float(i)]) for i in range(8)]

print(f"{'n_delay':>7s} | applied sequence (first 6) | steady-state delay")
ok = True
for n_delay in (0, 1, 2, 5):
    out = pipeline(acts, n_delay)
    seq = [int(o[0]) for o in out]
    # steady-state delay: compare tail where buffer is full
    # feed constant-then-step probe: easier check on the run above:
    # after warm-up (index >= n_delay), out[i] should equal acts[i - n_delay]
    delay_ok = all(seq[i] == i - n_delay for i in range(n_delay, len(seq)))
    print(f"{n_delay:7d} | {str(seq[:6]):>25s} | {'%d steps OK' % n_delay if delay_ok else 'WRONG'}")
    ok = ok and delay_ok

print("\nI26 VERIFIED: steady-state delay == n_delay exactly (post-warm-up)"
      if ok else "I26 FAIL")
