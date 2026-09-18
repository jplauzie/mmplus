"""Standalone SBP-SAT momentum test on a square geometry.
No shapes library, no second magnet on the same world.
"""

import numpy as np
from mumaxplus import World, Grid, Ferromagnet
import os

os.environ["MUMAXPLUS_FP_PRECISION"] = "DOUBLE"

def H_weighted_momentum(mag):
    v   = mag.elastic_velocity.eval()                 # (3, nz, ny, nx)
    rho = np.squeeze(mag.rho.eval(), axis=0)          # (nz, ny, nx)
    nz_, ny_, nx_ = rho.shape

    # Weight = 1 in interior, 1/2 at geometry-boundary endpoints in each axis.
    # For a solid square the geometry boundary is the grid boundary in x/y,
    # and in z the geometry spans a single cell so all cells are "boundary".
    Hx = np.ones(nx_); Hx[0] = 0.5; Hx[-1] = 0.5
    Hy = np.ones(ny_); Hy[0] = 0.5; Hy[-1] = 0.5
    Hz = np.ones(nz_); Hz[0] = 0.5; Hz[-1] = 0.5   # single-slice case

    H = Hz[:, None, None] * Hy[None, :, None] * Hx[None, None, :]
    return np.sum((H * rho)[None, ...] * v, axis=(1, 2, 3))


# ------------------------------------------------------------------
# Grid / world
# ------------------------------------------------------------------
nx, ny, nz = 128, 128, 1
Lx, Ly, Lz = 1e-6, 1e-6, 20e-9
cx, cy, cz = Lx/nx, Ly/ny, Lz/nz

grid = Grid((nx, ny, nz))
world = World((cx, cy, cz))

# ------------------------------------------------------------------
# Square geometry as a raw boolean numpy array, shape (nz, ny, nx)
# Half-width 50 cells, centered.
# ------------------------------------------------------------------
half = 50
mask = np.zeros((nz, ny, nx), dtype=bool)
mask[0, ny//2 - half : ny//2 + half,
       nx//2 - half : nx//2 + half] = True

print(f"cells in geometry: {mask.sum()}")
print(f"y range: {np.where(mask.any(axis=(0, 2)))[0].min()}–"
      f"{np.where(mask.any(axis=(0, 2)))[0].max()}")
print(f"x range: {np.where(mask.any(axis=(0, 1)))[0].min()}–"
      f"{np.where(mask.any(axis=(0, 1)))[0].max()}")
print(f"y-symmetric: {np.array_equal(mask, mask[:, ::-1, :])}")
print(f"x-symmetric: {np.array_equal(mask, mask[:, :, ::-1])}")

# ------------------------------------------------------------------
# Magnet setup
# ------------------------------------------------------------------
magnet = Ferromagnet(world, grid, geometry=mask)
magnet.msat = 1.2e6
magnet.aex  = 18e-12
magnet.alpha = 0.004

magnet.enable_elastodynamics = True
magnet.rho  = 8e3
magnet.C11  = 283e9
magnet.C44  = 58e9
magnet.C12  = 166e9
magnet.eta  = 0.0

magnet.elastic_displacement = (0, 0, 0)
magnet.elastic_velocity     = (0, 0, 0)

# ------------------------------------------------------------------
# Uniform dilation seed
# ------------------------------------------------------------------
r        = magnet.elastic_velocity.meshgrid
rho_eval = np.squeeze(magnet.rho.eval(), axis=0)
com_mass = np.sum(rho_eval[None, ...] * r, axis=(1, 2, 3)) / rho_eval.sum()

a = 1e-3
u_seed = np.zeros_like(r)
for i in range(3):
    u_seed[i] = a * (r[i] - com_mass[i])

magnet.elastic_displacement = u_seed
try:
    f = magnet.effective_body_force.eval()
except Exception as e:
    try:
        f = magnet.internal_body_force.eval()
    except Exception as e2:
        f = None
        print("Could not evaluate body force:", e, e2)

if f is not None:
    print(f"max|f_body| = {np.max(np.abs(f)):.3e}")
    print(f"min|f_body| = {np.min(np.abs(f)):.3e}")
    # Where is it nonzero?
    fmax = np.max(np.abs(f), axis=0)          # (nz, ny, nx)
    nz_idx, ny_idx, nx_idx = np.unravel_index(np.argmax(fmax), fmax.shape)
    print(f"max f at (x={nx_idx}, y={ny_idx}, z={nz_idx})")
u_check = magnet.elastic_displacement.eval()
print(f"max|u| after set: {np.max(np.abs(u_check)):.3e}")
print(f"u_seed max:       {np.max(np.abs(u_seed)):.3e}")
print(f"shape u: {u_check.shape}, seed: {u_seed.shape}")
magnet.elastic_velocity     = np.zeros_like(magnet.elastic_velocity.eval())

strain = magnet.strain_tensor.eval()
stress = magnet.stress_tensor.eval()
f_body = magnet.effective_body_force.eval()

print(f"max|u|      = {np.max(np.abs(magnet.elastic_displacement.eval())):.3e}")
print(f"max|strain| = {np.max(np.abs(strain)):.3e}")
print(f"max|stress| = {np.max(np.abs(stress)):.3e}")
print(f"max|f_body| = {np.max(np.abs(f_body)):.3e}")

f = magnet.effective_body_force.eval()    # shape (3, nz, ny, nx)
print("f at (14, 14):  ", f[:, 0, 14, 14])   # corner
print("f at (14, 64):  ", f[:, 0, 14, 64])   # left edge, mid
print("f at (113, 14): ", f[:, 0, 113, 14])  # corner
print("f at (113, 64): ", f[:, 0, 113, 64])  # right edge, mid
print("f at (64, 14):  ", f[:, 0, 64, 14])   # bottom edge, mid
print("f at (64, 113): ", f[:, 0, 64, 113])  # top edge, mid
print("f at (64, 64):  ", f[:, 0, 64, 64])   # interior

f = magnet.effective_body_force.eval()
# Sum over geometry only. Use the mask for correctness.
f_sum = np.sum(f[:, 0][:, mask[0]], axis=-1)
print(f"Σ F = ({f_sum[0]:+.6e}, {f_sum[1]:+.6e}, {f_sum[2]:+.6e})")
print(f"Σ F / (single cell) = {f_sum[0] / 1.149e17:+.3e}")

# strain at corner cells
strain = magnet.strain_tensor.eval()
print("strain at (14, 14): ", strain[:, 0, 14, 14])
print("strain at (14, 64): ", strain[:, 0, 14, 64])

# grid and mastergrid sizes
print("grid size:        ", grid.size)
try:
    print("mastergrid size:  ", world.mastergrid.size)
except Exception as e:
    print("mastergrid not exposed:", e)

# where is the strain nonzero?
s = np.max(np.abs(strain), axis=(0, 1))   # (ny, nx)
ny_, nx_ = np.unravel_index(np.argmax(s), s.shape)
print(f"max strain at (x={nx_}, y={ny_})")
print(f"strain at (64, 64): {strain[:, 0, 64, 64]}")
print(f"strain at (14, 64): {strain[:, 0, 14, 64]}")

# Are strain and stress nonzero in the interior? Boundary?
print(f"strain at (64,64) = {strain[:, 0, 64, 64]}")
print(f"strain at (14,64) = {strain[:, 0, 14, 64]}")

# ------------------------------------------------------------------
# Run and report
# ------------------------------------------------------------------
def P_and_KE(mag):
    v   = mag.elastic_velocity.eval()
    rho = np.squeeze(mag.rho.eval(), axis=0)
    P   = np.sum(rho[None, ...] * v, axis=(1, 2, 3))
    KE  = 0.5 * np.sum(rho[None, ...] * v**2)
    return P, KE

print("=" * 60)
print("SQUARE TEST: uniform dilation, free surface")
print("=" * 60)

P0, _ = P_and_KE(magnet)
PH0   = H_weighted_momentum(magnet)
print(f"initial  P   = ({P0[0]:+.3e}, {P0[1]:+.3e}, {P0[2]:+.3e})")
print(f"initial  P_H = ({PH0[0]:+.3e}, {PH0[1]:+.3e}, {PH0[2]:+.3e})")

for k in range(200):
    world.timesolver.adaptive_timestep = False
    world.timesolver.timestep = 1e-15
    world.timesolver.run(5e-15)
    if k % 20 == 0:
        P, KE = P_and_KE(magnet)
        PH    = H_weighted_momentum(magnet)
        t     = world.timesolver.time        # property, no ()
        dt    = world.timesolver.timestep    # property, no ()
        print(f"step {k:4d}  Px={P[0]:+.4e}  Py={P[1]:+.4e}  "
              f"Px_H={PH[0]:+.4e}  Py_H={PH[1]:+.4e}  "
              f"KE={KE:.4e}  t={t:.3e}  dt={dt:.3e}")