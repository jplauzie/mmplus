from fractions import Fraction as F

import numpy as np


def lower(rows):
    s = len(rows)
    A = np.zeros((s, s))
    for i, r in enumerate(rows):
        for j, v in enumerate(r):
            A[i, j] = float(v)
    return A

# ---- Dormand-Prince (7 stages) ----
A_dp = lower([
    [],
    [F(1,5)],
    [F(3,40), F(9,40)],
    [F(44,45), F(-56,15), F(32,9)],
    [F(19372,6561), F(-25360,2187), F(64448,6561), F(-212,729)],
    [F(9017,3168), F(-355,33), F(46732,5247), F(49,176), F(-5103,18656)],
    [F(35,384), 0, F(500,1113), F(125,192), F(-2187,6784), F(11,84)],
])
b1_dp = np.array([F(35,384), 0, F(500,1113), F(125,192), F(-2187,6784), F(11,84), 0], float)
b2_dp = np.array([F(5179,57600), 0, F(7571,16695), F(393,640), F(-92097,339200), F(187,2100), F(1,40)], float)

# ---- Fehlberg (6 stages) ----
A_fe = lower([
    [],
    [F(1,4)],
    [F(3,32), F(9,32)],
    [F(1932,2197), F(-7200,2197), F(7296,2197)],
    [F(439,216), -8, F(3680,513), F(-845,4104)],
    [F(-8,27), 2, F(-3544,2565), F(1859,4104), F(-11,40)],
])
b1_fe = np.array([F(16,135), 0, F(6656,12825), F(28561,56430), F(-9,50), F(2,55)], float)
b2_fe = np.array([F(25,216), 0, F(1408,2565), F(2197,4104), F(-1,5), 0], float)

def R(z, A, b):
    s = len(b)
    return 1 + z * (b @ np.linalg.solve(np.eye(s) - z*A, np.ones(s)))

for name, A, bs in [("Dormand-Prince", A_dp, {"weights1 (5th)": b1_dp, "weights2 (4th)": b2_dp}),
                    ("Fehlberg",       A_fe, {"weights1 (5th)": b1_fe, "weights2 (4th)": b2_fe})]:
    print(f"\n{name}")
    for wname, b in bs.items():
        print(f"  {wname}")
        for y in (0.05, 0.1, 0.33, 0.7, 1.0, 1.5):
            g = abs(R(1j*y, A, b))**2 - 1
            print(f"    y={y:5.2f}  |R|^2-1 = {g:+.3e}   ({'GAIN' if g>0 else 'loss'})")