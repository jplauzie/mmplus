"""AFM spin-flop field sweep, for A/B testing the line search curve (LS_CAYLEY).

Usage:   python afm_sweep.py LABEL        (for example cayley0 or cayley1)
Output:  afm_sweep_LABEL.csv

The antiferromagnet starts (almost) antiparallel along the easy axis x, and the
bias field along x is increased step by step. world.minimize() is called at
every field value, starting from the state left by the previous field value.
Before every call both sublattices are rotated by KICK rad about z: an exactly
antiparallel state has zero torque, so without the kick it never spin-flops.
With the parameters below, the spin flop is expected at about 3.6 T:

    J    = 4|A0|/a^2 = 1.3e8 J/m3
    B_sf = 2 sqrt(K J)/Ms ~ 3.6 T

Per field step the script records the minimizer statistics (parsed from the
"Minimize: N steps, ..." line printed with verbose=1), the wall time, and a few
state quantities, so that two builds can be compared with afm_sweep_compare.py.

Set MUMAXPLUS_FP_PRECISION before importing mumaxplus if you want double.
"""

import csv
import os
import re
import sys
import tempfile
import time

import numpy as np

os.environ["MUMAXPLUS_FP_PRECISION"] = "DOUBLE"

from mumaxplus import Antiferromagnet, Grid, World

# ============================================================
# Settings
# ============================================================

LABEL = sys.argv[1] if len(sys.argv) > 1 else "run"
OUT_FILE = f"afm_sweep_{LABEL}.csv"

N_REPEATS = 5        # independent repeats (different initial noise)
SEED0 = 1234567      # repeat r uses seed SEED0 + r
NOISE = 0.02         # initial noise (it relaxes away at low field, so it does NOT keep
                     # the state off the easy axis, see KICK)
KICK = 0.1           # rad: before every minimize, rotate both sublattices about z by
                     # this angle. They stay antiparallel, but the Neel vector leaves the
                     # easy axis, which is the mode that drives the spin flop. Without it
                     # the exactly antiparallel state has zero torque and never flops.
                     # Set to 0 to disable.
MODE = 3             # line_search_mode
TOL, NSAMPLES = 1e-6, 10
SWEEP_DOWN = False   # True: after reaching the top field, sweep back down as well

# 3.0 ... 4.1 T in 0.1 T steps (integers avoid arange rounding adding a 4.2 T point)
FIELDS = np.round(np.arange(30, 42) / 10.0, 3)
if SWEEP_DOWN:
    FIELDS = np.concatenate([FIELDS, FIELDS[-2::-1]])

# ----- geometry -----
dx = 4e-9
nx, ny, nz = 64, 32, 1

# ----- material -----
Ms = 200e3
A0 = -4e-13          # afmex_cell: lowered so that the spin flop is at ~3.6 T
A = 10e-12           # aex of each sublattice
A_NN = -5e-12        # afmex_nn
K = 1e4
a = 0.35e-9          # lattice constant
alpha = 0

# ============================================================
# Capturing the C++ output of one minimize() call
# ============================================================


class CaptureFd:
    """Capture everything written to file descriptor 1 (including the output of
    C++ std::cout) inside a with-block. The text is available as .text."""

    def __enter__(self):
        sys.stdout.flush()
        self._tmp = tempfile.TemporaryFile(mode="w+b")
        self._saved = os.dup(1)
        os.dup2(self._tmp.fileno(), 1)
        self.text = ""
        return self

    def __exit__(self, *exc):
        sys.stdout.flush()
        os.dup2(self._saved, 1)
        os.close(self._saved)
        self._tmp.seek(0)
        self.text = self._tmp.read().decode(errors="replace")
        self._tmp.close()
        return False


PATTERN = re.compile(
    r"Minimize: (\d+) steps, (\d+) line searches, (\d+) GLL rescues"
    r"(?:, (\d+) line search evals)?")


def parse_stats(text):
    """Return (steps, line searches, rescues, evals), with -1 where not found."""
    m = PATTERN.search(text)
    if m is None:
        return (-1, -1, -1, -1)
    return tuple(int(g) if g is not None else -1 for g in m.groups())


# ============================================================
# System
# ============================================================

world = World((dx, dx, dx))
magnet = Antiferromagnet(world, Grid((nx, ny, nz)))
world.set_pbc((1, 1, 0))   # after adding the magnet: periodic in x and y

magnet.afmex_nn = A_NN
magnet.afmex_cell = A0
magnet.latcon = a

for sub in magnet.sublattices:
    sub.msat = Ms
    sub.alpha = alpha
    sub.aex = A
    sub.ku1 = K
    sub.anisU = (1, 0, 0)


def set_initial_state(seed):
    """Sublattices antiparallel along x, plus a little noise."""
    rng = np.random.default_rng(seed)
    shape = magnet.sub1.magnetization.shape
    for sub, sign in ((magnet.sub1, +1.0), (magnet.sub2, -1.0)):
        m = np.zeros(shape)
        m[0] = sign
        m += NOISE * rng.normal(size=shape)
        sub.magnetization = m   # normalized on set


def kick_state(angle):
    """Rotate both sublattices by the same angle about z. They stay antiparallel, so
    the exchange energy does not change, but the Neel vector leaves the easy axis x."""
    c, s = np.cos(angle), np.sin(angle)
    for sub in magnet.sublattices:
        m = np.array(sub.magnetization.eval())   # shape (3, nz, ny, nx)
        mx, my = m[0].copy(), m[1].copy()
        m[0] = c * mx - s * my
        m[1] = s * mx + c * my
        sub.magnetization = m


def measure():
    m1 = np.array(magnet.sub1.magnetization.average())
    m2 = np.array(magnet.sub2.magnetization.average())
    n = 0.5 * (m1 - m2)        # Neel vector (cell average)
    m = 0.5 * (m1 + m2)        # net magnetization (cell average)
    energy = sum(float(sub.total_energy.eval()) for sub in magnet.sublattices)
    return n[0], float(np.hypot(n[1], n[2])), m[0], energy


# ============================================================
# Sweep
# ============================================================

header = ["label", "repeat", "point", "field", "steps", "line_searches", "rescues",
          "evals", "time_s", "n_x", "n_perp", "m_x", "energy"]

warned = False
with open(OUT_FILE, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(header)

    for rep in range(N_REPEATS):
        set_initial_state(SEED0 + rep)

        for point, B in enumerate(FIELDS):
            world.bias_magnetic_field = (float(B), 0.0, 0.0)
            if KICK:
                kick_state(KICK)   # not included in the timing below

            t0 = time.perf_counter()
            with CaptureFd() as cap:
                world.minimize(TOL, NSAMPLES, line_search_mode=MODE, verbose=1)
            dt = time.perf_counter() - t0

            steps, nls, nresc, nevals = parse_stats(cap.text)
            if steps < 0 and not warned:
                print("WARNING: could not parse the 'Minimize:' line; statistics "
                      "are recorded as -1. Captured text was:\n" + cap.text)
                warned = True

            n_x, n_perp, m_x, energy = measure()
            writer.writerow([LABEL, rep, point, B, steps, nls, nresc, nevals,
                             f"{dt:.4f}", f"{n_x:.6f}", f"{n_perp:.6f}",
                             f"{m_x:.6e}", f"{energy:.9e}"])
            f.flush()
            print(f"[{LABEL}] rep {rep} B={B:5.2f} T: {steps:6d} steps, "
                  f"{nls:4d} line searches, {dt:7.3f} s, n_x={n_x:+.4f}, "
                  f"E={energy:.6e}", flush=True)

print(f"Wrote {OUT_FILE}")
