"""Compare two afm_sweep.py runs (for example the two line search curves).

Usage:  python afm_sweep_compare.py LABEL_A LABEL_B
Reads   afm_sweep_LABEL_A.csv and afm_sweep_LABEL_B.csv

Rows are paired by (repeat, point): the same initial noise and the same field
step. Ratios are B/A, so a ratio below 1 means that B needed fewer steps.
"""

import csv
import sys

import numpy as np

WINDOW = (3.3, 4.0)   # field window (T) around the expected spin flop


def load(label):
    with open(f"afm_sweep_{label}.csv") as f:
        rows = list(csv.DictReader(f))
    return {(int(r["repeat"]), int(r["point"])): r for r in rows}


def col(rows, keys, name):
    return np.array([float(rows[k][name]) for k in keys])


label_a, label_b = sys.argv[1], sys.argv[2]
A, B = load(label_a), load(label_b)
keys = sorted(set(A) & set(B))
if not keys:
    sys.exit("no common (repeat, point) pairs in the two files")

field = col(A, keys, "field")
steps_a, steps_b = col(A, keys, "steps"), col(B, keys, "steps")
ls_a, ls_b = col(A, keys, "line_searches"), col(B, keys, "line_searches")
t_a, t_b = col(A, keys, "time_s"), col(B, keys, "time_s")
e_a, e_b = col(A, keys, "energy"), col(B, keys, "energy")
nx_a, nx_b = col(A, keys, "n_x"), col(B, keys, "n_x")
rep = np.array([k[0] for k in keys])
pt = np.array([k[1] for k in keys])

if np.any(steps_a < 0) or np.any(steps_b < 0):
    print("WARNING: some rows have no parsed statistics (steps = -1)\n")

# ----- per field value, averaged over the repeats -----
print(f"{'B (T)':>7} {'steps ' + label_a:>14} {'steps ' + label_b:>14} {'ratio':>7} "
      f"{'n_x ' + label_a:>12} {'n_x ' + label_b:>12}")
for p in sorted(set(pt)):
    sel = pt == p
    sa, sb = steps_a[sel].mean(), steps_b[sel].mean()
    print(f"{field[sel][0]:7.2f} {sa:14.1f} {sb:14.1f} {sb / sa:7.2f} "
          f"{nx_a[sel].mean():12.4f} {nx_b[sel].mean():12.4f}")


# ----- totals, paired over the repeats -----
def totals(mask, label):
    per_rep_a = np.array([steps_a[mask & (rep == r)].sum() for r in sorted(set(rep))])
    per_rep_b = np.array([steps_b[mask & (rep == r)].sum() for r in sorted(set(rep))])
    ratio = per_rep_b / per_rep_a
    se = ratio.std(ddof=1) / np.sqrt(len(ratio)) if len(ratio) > 1 else float("nan")
    print(f"{label:28s} steps {label_a} {steps_a[mask].sum():8.0f}  {label_b} "
          f"{steps_b[mask].sum():8.0f}  ratio {steps_b[mask].sum() / steps_a[mask].sum():.3f}"
          f"  (per-repeat ratio {ratio.mean():.3f} +- {se:.3f})")
    print(f"{'':28s} line searches {ls_a[mask].sum():.0f} vs {ls_b[mask].sum():.0f}, "
          f"time {t_a[mask].sum():.1f} s vs {t_b[mask].sum():.1f} s")


print()
totals(np.ones(len(keys), dtype=bool), "all fields")
totals((field >= WINDOW[0]) & (field <= WINDOW[1]),
       f"spin flop window {WINDOW[0]}-{WINDOW[1]} T")

# ----- did both builds end in the same states? -----
rel_e = np.abs(e_b - e_a) / np.maximum(np.abs(e_a), 1e-300)
bad = (rel_e > 1e-6) | (np.abs(nx_b - nx_a) > 0.05)
print(f"\nstates that differ (|dE|/|E| > 1e-6 or |dn_x| > 0.05): {bad.sum()} of {len(keys)}")
for i in np.where(bad)[0][:20]:
    print(f"  repeat {rep[i]}, B={field[i]:.2f} T: E {e_a[i]:.6e} vs {e_b[i]:.6e}, "
          f"n_x {nx_a[i]:.4f} vs {nx_b[i]:.4f}")
print("\nNote: step counts are only comparable where both builds reach the same state.")
