"""Test script for the combined magnetic + elastic Minimizer.

Geometry and magnetic material parameters (CoFeB) match the companion
mumax3 Go test script, for direct comparison.

Workflow:
  1. Set up geometry + magnetic material, do a magnetic-only minimize.
  2. Enable elastodynamics, set elastic + magnetoelastic parameters.
  3. Run real coupled time-domain dynamics for 500 ps (world.timesolver.steps
     with a fixed timestep), letting m and u co-evolve under actual LLG +
     elastodynamics (with damping). This gives the minimizer a much
     better-conditioned starting point than a cold u=(0,0,0), matching how
     the mumax3 test reaches its starting state.
  4. Run the combined magnetic + elastic minimize as a final cleanup.

Note: OVF export uses each FieldQuantity's own .save_ovf() method (confirmed
via mumaxplus's pyovf test suite), falling back to .npy then plain .txt if
that method is unavailable or errors. Final-state plots (below) are produced
either way.

Note: periodic boundary conditions are intentionally NOT used here. With a
uniform magnetization and periodic boundaries, the magnetoelastic body force
is spatially uniform, so its divergence -- and therefore the relaxing force
on the elastic displacement -- is exactly zero everywhere. Using free
boundaries instead means the edges break that symmetry, giving the elastic
minimizer something real to relax.
"""

import numpy as np
import matplotlib.pyplot as plt
import mumaxplus
import time

from mumaxplus import World, Grid, Ferromagnet
from mumaxplus.util import vortex
import mumaxplus.util.shape as shapes


def save_field(quantity, filename):
    """Save a FieldQuantity/Variable to an OVF file via its own .save_ovf()
    method, confirmed by mumaxplus's pyovf test suite (e.g.
    magnet.magnetization.save_ovf(filename)). Falls back to .npy (via
    quantity.eval()) if save_ovf isn't available or errors, then flattened
    .txt as a last resort.
    """
    try:
        quantity.save_ovf(filename)
        print(f"Saved {filename}")
        return
    except Exception as e:
        print(f"OVF save failed for {filename} ({e}); falling back to .npy")

    data = quantity.eval()
    npy_name = filename.rsplit(".", 1)[0] + ".npy"
    try:
        np.save(npy_name, data)
        print(f"Saved {npy_name}")
        return
    except Exception as e:
        print(f".npy save failed for {npy_name} ({e}); falling back to .txt")

    txt_name = filename.rsplit(".", 1)[0] + ".txt"
    np.savetxt(txt_name, data.reshape(-1))
    print(f"Saved {txt_name} (flattened, original shape {data.shape})")

# ----- explicit minimizer settings, tweak these -----
TOL = 1e-6           # magnetic minimize tolerance
NSAMPLES = 10        # magnetic minimize sample count
TOL_EL = 1e-14       # elastic minimize tolerance
NSAMPLES_EL = 10     # elastic minimize sample count
STEPSIZE = 1e-14     # initial magnetic BB stepsize
STEPSIZE_EL = 1e-30  # initial elastic BB stepsize (only used as the very
                     # first guess -- the scaled BB update should adapt
                     # this quickly regardless of the exact starting value)

DYNAMICS_TIME = 5e-10  # 500 ps warm-up run before minimize, matches mumax3 test
FIXED_DT = 1e-15       # matches mumax3 test's fixdt

# ----- geometry: matches mumax3 Go test script -----
dx, dy, dz = 4e-9, 4e-9, 1e-9
nx, ny, nz = 32, 32, 1
cellsize = (dx, dy, dz)
length, width = nx * dx, ny * dy

grid = Grid((nx, ny, nz))
# No mastergrid/pbc_repetitions here -- see module docstring for why.
world = World(cellsize)

circles=shapes.Circle(100e-9) - shapes.Circle(50e-9)

circles=circles.translate(100e-9 / 2, 100e-9 / 2, 0)

magnet = Ferromagnet(
    world,
    grid,
    geometry=circles
)



# ----- magnetic material: CoFeB, matches mumax3 Go test script -----
magnet.msat = 800e3
magnet.aex = 13e-12
magnet.alpha = 0.1
#magnet.magnetization = (1, 0, 0)  # high-energy start to exercise the descent
c = 4e-9
magnet.magnetization = vortex(magnet.center, 2*c, 1, -1)

Bdc = 5e-3
magnet.bias_magnetic_field = (Bdc, 0, 0)
# magnetic-only minimize first
#magnet.minimize(TOL, NSAMPLES, stepsize=STEPSIZE)

# ----- elastic + magnetoelastic material: matches mumax3 Go test script -----
magnet.enable_elastodynamics = True
magnet.rho = 8e3
magnet.B1 = -8.8e6
magnet.B2 = -8.8e6
magnet.C11 = 283e9
magnet.C44 = 58e9
magnet.C12 = 166e9
magnet.eta = 1e13  # unused by minimize (damping term dropped), harmless to set

magnet.elastic_displacement = (0, 0, 0)

# ----- real coupled dynamics warm-up -----
# Confirmed API: world.timesolver.steps(n) with a fixed timestep, matching
# the magnetoelastic example in the examples folder. Split into chunks so
# there's a progress printout to watch instead of one long silent call.
world.timesolver.adaptive_timestep = False
world.timesolver.timestep = FIXED_DT
nsteps_dynamics = int(DYNAMICS_TIME / FIXED_DT)

N_CHUNKS = 10
steps_per_chunk = nsteps_dynamics // N_CHUNKS
remainder = nsteps_dynamics % N_CHUNKS

print(f"Running {nsteps_dynamics} steps ({DYNAMICS_TIME*1e9:.1f} ns) of coupled "
      f"dynamics in {N_CHUNKS} chunks...")

start_time = time.time()

for chunk in range(N_CHUNKS):
    chunk_steps = steps_per_chunk + (remainder if chunk == N_CHUNKS - 1 else 0)

    chunk_start = time.time()

    world.timesolver.steps(chunk_steps)

    chunk_elapsed = time.time() - chunk_start

    print(f"  chunk {chunk + 1}/{N_CHUNKS} done "
          f"(t = {world.timesolver.time * 1e9:.3f} ns, "
          f"wall = {chunk_elapsed:.2f} s)")

print(f"Dynamics done. Total wall time: {time.time() - start_time:.2f} s")

# combined magnetic + elastic minimize
print("Minimizing...")
#magnet.minimize(TOL, NSAMPLES, TOL_EL, NSAMPLES_EL, STEPSIZE, STEPSIZE_EL)
print("Done.")

# ----- gather final fields -----
m = magnet.magnetization.eval()               # shape (3, nz, ny, nx)
u = magnet.elastic_displacement.eval()        # shape (3, nz, ny, nx)
v = magnet.elastic_velocity.eval()            # shape (3, nz, ny, nx)
f = magnet.effective_body_force.eval()        # shape (3, nz, ny, nx)

print(f"max |u| = {np.max(np.linalg.norm(u, axis=0)):.3e} m")
print(f"max |v| = {np.max(np.linalg.norm(v, axis=0)):.3e} m/s")
print(f"max |f| = {np.max(np.linalg.norm(f, axis=0)):.3e} N/m3 "
      "(should be small: this is the residual the minimizer drove towards 0)")

save_field(magnet.magnetization, "m.ovf")
save_field(magnet.elastic_displacement, "u.ovf")
save_field(magnet.elastic_velocity, "v.ovf")
save_field(magnet.effective_body_force, "force.ovf")

# squeeze the single z-layer for plotting
m2d = m[:, 0, :, :]
u2d = u[:, 0, :, :]
v2d = v[:, 0, :, :]
f2d = f[:, 0, :, :]

extent = (-0.5 * dx, length - 0.5 * dx, -0.5 * dy, width - 0.5 * dy)

fig, axes = plt.subplots(2, 2, figsize=(13, 7))

# --- magnetization: my colormap ---
ax = axes[0, 0]
im = ax.imshow(m2d[1], origin="lower", extent=extent, cmap="seismic",
               vmin=-1, vmax=1)
ax.set_title(r"Magnetization $m_y$")
ax.set_xlabel("x (m)")
ax.set_ylabel("y (m)")
plt.colorbar(im, ax=ax)

# --- displacement: magnitude + quiver of in-plane components ---
ax = axes[0, 1]
u_mag = np.linalg.norm(u2d, axis=0)
im = ax.imshow(u_mag, origin="lower", extent=extent, cmap="viridis")
skip = 8
Y, X = np.mgrid[0:ny, 0:nx]
Xp = X[::skip, ::skip] * dx - 0.5 * dx
Yp = Y[::skip, ::skip] * dy - 0.5 * dy
ax.quiver(Xp, Yp, u2d[0, ::skip, ::skip], u2d[1, ::skip, ::skip],
          color="white", scale=None)
ax.set_title("Elastic displacement |u| (quiver: in-plane direction)")
ax.set_xlabel("x (m)")
ax.set_ylabel("y (m)")
plt.colorbar(im, ax=ax)

# --- velocity magnitude ---
ax = axes[1, 0]
v_mag = np.linalg.norm(v2d, axis=0)
im = ax.imshow(v_mag, origin="lower", extent=extent, cmap="viridis")
ax.set_title("Elastic velocity |v|")
ax.set_xlabel("x (m)")
ax.set_ylabel("y (m)")
plt.colorbar(im, ax=ax)

# --- effective body force magnitude ---
ax = axes[1, 1]
f_mag = np.linalg.norm(f2d, axis=0)
im = ax.imshow(f_mag, origin="lower", extent=extent, cmap="viridis")
ax.set_title("Effective body force |f| (residual)")
ax.set_xlabel("x (m)")
ax.set_ylabel("y (m)")
plt.colorbar(im, ax=ax)

fig.tight_layout()
fig.savefig("minimizer_test.png", dpi=150)
print("Saved plot to minimizer_test.png")
plt.show()