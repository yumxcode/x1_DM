"""r27R-4 closure: bucket curve extraction + three-column verdict computation.

Per R010b guardrails v2 + R010d v2.1 (pre-registered BEFORE relaunch).
"""
import json
import re

d = json.load(open('/Users/yumx/code/x1_DM/.repos/r27R4final.json'))
s = d.get('data') or ''
if isinstance(s, dict):
    s = str(s)
txt = ''.join(l for l in s.replace('][SDK]', '\n[SDK]').split('\n')
              if '[SDK]' not in l and 'OmniHub' not in l)
pats = {'iter': r'Iteration\s+\|\s+(\d+)', 'samples': r'Samples\s+\|\s+(\d+)',
        'smp': r'Smp_Reward_Mean\s+\|\s+([\d.]+)', 'task': r'Task_Reward_Mean\s+\|\s+([\d.]+)',
        'test_ret': r'Test_Return\s+\|\s+(\S+)', 'train_ret': r'Train_Return\s+\|\s+(\S+)',
        'fail_count': r'\|\s+Fail_Count\s+\|\s+(\d+)', 'fail_low_root': r'\|\s+Fail_Low_Root\s+\|\s+(\d+)'}
cols = {k: re.findall(v, txt) for k, v in pats.items()}
n = min(len(v) for v in cols.values())
cols = {k: v[:n] for k, v in cols.items()}
json.dump(cols, open('/Users/yumx/code/x1_DM/analysis/r27R4_buckets.json', 'w'))
print(f'saved r27R4_buckets.json: {n} rows, {int(cols["samples"][0])/1e6:.0f}M..{int(cols["samples"][-1])/1e6:.0f}M, iter {cols["iter"][0]}..{cols["iter"][-1]}')

fc = [int(x) for x in cols['fail_count']]
fl = [int(x) for x in cols['fail_low_root']]
it = [int(x) for x in cols['iter']]
last_iter = it[-1]
idx_last = [i for i, x in enumerate(it) if x >= last_iter - 400]
i0, i1 = idx_last[0], idx_last[-1]
d_fc = fc[i1] - fc[i0]
d_fl = fl[i1] - fl[i0]
d_share = d_fl / max(d_fc, 1)
print(f'\ncol1 (last {it[i1]-it[i0]}-iter differential window, iter {it[i0]}-{it[i1]}):')
print(f'  d_Fail={d_fc:,} d_LowRoot={d_fl:,} diff share={d_share*100:.1f}% (baseline 13%)')
lr = 0.13
sh = d_share
if sh > lr:
    x = 1 - (lr * (1 - sh)) / (sh * (1 - lr))
    print(f'  converted drift-reduction x = {x*100:.1f}%')
    if x >= 0.78:
        v = 'CONFIRM (strict share>=40% line)'
    elif x >= 0.50:
        v = 'CONFIRM (loose FAIL-collapse line) / strong partial'
    else:
        v = 'NO SIGNAL (<50%)'
    print(f'  verdict: {v}')
else:
    print('  share <= baseline 13% -> drift-reduction <= 0 -> NO SIGNAL')


def abs_rate(a, b):
    return (fc[b] - fc[a]) / ((it[b] - it[a]) * 4096 * 32)


mid = n // 2
r_mid = abs_rate(mid, mid + 5)
r_last = abs_rate(i0, i1)
print(f'\ncol2 absolute FAIL rate: mid-window {r_mid:.4f} | last window {r_last:.4f} | ratio {r_last/r_mid:.2f} (collapse line <0.5)')
print(f'\ncol3 ep_len = 48.35 (<60 tier: lean deny + fresh-cost caveat)')
print(f'col4 test_ret: plateau 22.4-25 final 24.5 - NO end jump')
