"""Quick checks of the adjoint force kernel, no time stepping.

Choose the case with an environment variable, one case per process:
  set CASE=edge_nopbc   magnet fills the whole grid, no PBC  (tests the +-4 neighbour reads at grid edges)
  set CASE=edge_pbc     magnet fills the whole grid, PBC in x,y
  set CASE=block3d      32x32x6 block (z stencils, xz/yz shear)
  set CASE=hetero       C11 varies up to 5x cell to cell (needs numpy assignment to parameters)

Prints, for random displacement fields u1, u2 with all couplings/damping off:
  net force / sum|F|              (momentum conservation; ~1e-16 in DOUBLE, ~1e-7 in SINGLE)
  |<u2,F(u1)> - <u1,F(u2)>| / max (symmetry of the force operator; same tolerance)
"""
import os
os.environ.setdefault("MUMAXPLUS_FP_PRECISION", "DOUBLE")   # set BEFORE importing mumaxplus
import numpy as np
from mumaxplus import World, Grid, Ferromagnet

case = os.environ.get("CASE", "edge_nopbc")
cs = (3.9e-9, 3.9e-9, 20e-9)
n = (48, 48, 1)

if case == "edge_nopbc":
    world = World(cs)
elif case == "edge_pbc":
    world = World(cs, mastergrid=Grid((n[0], n[1], 0)), pbc_repetitions=(2, 2, 0))
elif case == "block3d":
    n = (32, 32, 6)
    world = World(cs)
elif case == "hetero":
    world = World(cs)
else:
    raise SystemExit(f"unknown CASE={case}")

nx, ny, nz = n
print(f"case={case}  grid={n}  precision={os.environ['MUMAXPLUS_FP_PRECISION']}")

magnet = Ferromagnet(world, Grid(n))
magnet.msat = 1e6
magnet.enable_elastodynamics = True
magnet.rho = 8e3
magnet.C11 = 283e9
magnet.C12 = 166e9
magnet.C44 = 58e9
magnet.B1 = 0
magnet.B2 = 0
magnet.eta = 0
magnet.stiffness_damping = 0

rng = np.random.default_rng(1)
if case == "hetero":
    try:
        magnet.C11 = 283e9 * np.exp(rng.uniform(0, np.log(5), (1, nz, ny, nx)))
    except Exception as e:
        raise SystemExit(f"array assignment to a parameter is not supported here: {e}")


def force(u):
    magnet.elastic_displacement = u
    return magnet.internal_body_force.eval()


u1 = 1e-12 * rng.standard_normal((3, nz, ny, nx))
u2 = 1e-12 * rng.standard_normal((3, nz, ny, nx))
F1, F2 = force(u1), force(u2)

net = F1.sum(axis=(1, 2, 3)) / np.abs(F1).sum(axis=(1, 2, 3))
a, b = float(np.sum(u2 * F1)), float(np.sum(u1 * F2))
print("finite:", bool(np.isfinite(F1).all() and np.isfinite(F2).all()))
print("net force / sum|F| per component:", net)
print(f"<u2,F(u1)>={a:.6e}  <u1,F(u2)>={b:.6e}  relative mismatch={abs(a-b)/max(abs(a), abs(b), 1e-300):.2e}")
