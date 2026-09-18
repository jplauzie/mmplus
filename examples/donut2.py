"""Donut test with SBP-SAT BCs, single vs double precision check.

Run with SINGLE (default) and DOUBLE to compare:
    set MUMAXPLUS_FP_PRECISION=DOUBLE      (Windows: set, Unix: export)
    python donutretest.py

Diagnostics reported each output step:
  - P, P_H        total and H-weighted momentum
  - KE            total kinetic energy
  - max|u|        peak displacement magnitude
  - <u_r>         mean radial displacement (outward drift indicator)
"""

import os
import numpy as np
import math
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# ------------------------------------------------------------------
# Precision: read BEFORE importing mumaxplus
# ------------------------------------------------------------------
fp = os.environ.get("MUMAXPLUS_FP_PRECISION", "SINGLE").upper()
print(f"Running with MUMAXPLUS_FP_PRECISION={fp}")

from mumaxplus import World, Grid, Ferromagnet
from mumaxplus.util import vortex, antivortex, twodomain
import mumaxplus.util.shape as shapes


# ==================================================================
# 1. Setup
# ==================================================================

length, width, thickness = 1e-6, 1e-6, 20e-9
nx, ny, nz = 256, 256, 1
cx, cy, cz = length/nx, width/ny, thickness/nz
cellsize = (cx, cy, cz)

grid = Grid((nx, ny, nz))
world = World(cellsize, mastergrid=Grid((nx, ny, 0)), pbc_repetitions=(2, 2, 0))

circles = shapes.Circle(500e-9) - shapes.Circle(200e-9)
circles = circles.translate(nx * cx / 2, ny * cy / 2, 0)

magnet = Ferromagnet(world, grid, geometry=circles)

magnet.msat = 1.2e6
magnet.aex = 18e-12
magnet.alpha = 0.004
Bdc = 5e-3
magnet.bias_magnetic_field = (Bdc, 0, 0)

magnet.magnetization = vortex(magnet.center, 12e-9, -1, 1)
magnet.minimize()

magnet.enable_elastodynamics = True
magnet.rho = 8e3
magnet.B1 = -8.8e6
magnet.B2 = -8.8e6
magnet.C11 = 283e9
magnet.C44 = 58e9
magnet.C12 = 166e9

magnet.elastic_displacement = (0, 0, 0)
magnet.elastic_velocity = (0, 0, 0)

# Damping — keep your production value.
magnet.eta = 1e10
magnet.alpha = 0.004


# ==================================================================
# 2. Diagnostics
# ==================================================================

r_mesh = magnet.elastic_velocity.meshgrid              # (3, nz, ny, nx)
rho_flat = np.squeeze(magnet.rho.eval(), axis=0)       # (nz, ny, nx)

# Center of mass of the donut, from the mask.
mask = magnet._get_mask_array(circles, grid, world, "mask").astype(bool)
mass_flat = rho_flat * mask                            # (nz, ny, nx)

com = np.array([
    np.sum(mass_flat * r_mesh[i]) / np.sum(mass_flat)
    for i in range(3)
])

# Radial unit vector at each cell (in-plane only).
dx = r_mesh[0] - com[0]
dy = r_mesh[1] - com[1]
rr = np.sqrt(dx**2 + dy**2) + 1e-30
er_x = dx / rr
er_y = dy / rr

def total_momentum(mag):
    v = mag.elastic_velocity.eval()
    return np.sum(rho_flat[None, ...] * v, axis=(1, 2, 3))

def kinetic_energy(mag):
    v = mag.elastic_velocity.eval()
    return 0.5 * np.sum(rho_flat[None, ...] * v**2)

def max_u(mag):
    return float(np.max(np.abs(mag.elastic_displacement.eval())))

def mean_radial_displacement(mag):
    u = mag.elastic_displacement.eval()
    return float(np.mean((u[0] * er_x + u[1] * er_y)[mask]))


# ==================================================================
# 3. Original excitation and simulation
# ==================================================================

f_B = 9.8e9
Bac = 0e-3
Bdiam = 200e-9

Bshape = shapes.Circle(Bdiam).translate(*magnet.center)
Bmask = magnet._get_mask_array(Bshape, grid, world, "mask")
magnet.bias_magnetic_field.add_time_term(
    lambda t: (0, Bac * math.sin(2*math.pi*f_B * t), 0), mask=Bmask)

def displacement_to_scatter_data(magnet, scale, skip):
    u = magnet.elastic_displacement.eval()
    coords = magnet.elastic_displacement.meshgrid + scale * u
    return np.transpose(coords[:2, 0, ::skip, ::skip].reshape(2, -1))

fig, ax = plt.subplots()
u_scale = 5e4
u_skip = 5

steps = 400
time_max = 0.3e-9
duration = time_max / steps

m_shape = np.transpose(magnet.magnetization.eval()[1, 0, :, :]).shape
u_shape = displacement_to_scatter_data(magnet, scale=u_scale, skip=u_skip).shape
m = np.zeros(shape=(steps, m_shape[0], m_shape[1]))
u = np.zeros(shape=(steps, u_shape[0], u_shape[1]))

# Diagnostic history
P_hist    = np.zeros((steps, 3))
KE_hist   = np.zeros(steps)
umax_hist = np.zeros(steps)
ur_hist   = np.zeros(steps)

print("Simulating...")
for i in tqdm(range(steps)):
    world.timesolver.run(duration)
    m[i, ...] = np.transpose(magnet.magnetization.eval()[1, 0, :, :])
    u[i, ...] = displacement_to_scatter_data(magnet, scale=u_scale, skip=u_skip)

    P_hist[i]    = total_momentum(magnet)
    KE_hist[i]   = kinetic_energy(magnet)
    umax_hist[i] = max_u(magnet)
    ur_hist[i]   = mean_radial_displacement(magnet)

    if i % 40 == 0:
        print(f"step {i:4d}  |P|={np.linalg.norm(P_hist[i]):.3e}  "
              f"KE={KE_hist[i]:.3e}  max|u|={umax_hist[i]:.3e}  "
              f"<u_r>={ur_hist[i]:+.3e}")


# ==================================================================
# 4. Post-run diagnostic plot
# ==================================================================

fig2, axs = plt.subplots(2, 2, figsize=(10, 8))

axs[0, 0].plot(np.linalg.norm(P_hist, axis=1))
axs[0, 0].set_ylabel("|P|")
axs[0, 0].set_xlabel("step")
axs[0, 0].set_title(f"total momentum ({fp})")

axs[0, 1].plot(KE_hist)
axs[0, 1].set_ylabel("kinetic energy")
axs[0, 1].set_xlabel("step")
axs[0, 1].set_title("KE (should be damped)")

axs[1, 0].plot(umax_hist)
axs[1, 0].set_ylabel("max |u|")
axs[1, 0].set_xlabel("step")
axs[1, 0].set_title("peak displacement")

axs[1, 1].plot(ur_hist)
axs[1, 1].set_ylabel("<u_r>")
axs[1, 1].set_xlabel("step")
axs[1, 1].set_title("mean radial displacement (outward drift)")

fig2.tight_layout()
fig2.savefig(f"diagnostics_{fp}.png", dpi=100)
plt.close(fig2)


# ==================================================================
# 5. Animation (unchanged)
# ==================================================================

offsets = displacement_to_scatter_data(magnet, scale=u_scale, skip=u_skip)
u_scatter = ax.scatter(offsets[:, 0], offsets[:, 1],
                       s=10, c="black", marker=".", alpha=0.5)

im_extent = (-0.5*cx, length - 0.5*cx, -0.5*cy, width - 0.5*cy)
vmax, vmin = np.max(m), np.min(m)
vmax = max(abs(vmax), abs(vmin))
vmin = -vmax
m_im = ax.imshow(m[0, ...], origin="lower", extent=im_extent,
                 vmin=vmin, vmax=vmax, cmap="seismic")

cbar = plt.colorbar(m_im)
cbar.ax.set_ylabel(r"$<m_y>$", rotation=270)

ax.set_xlabel("$x$ (m)")
ax.set_ylabel("$y$ (m)")
ax.set_xlim(im_extent[0], im_extent[1])
ax.set_ylim(im_extent[2], im_extent[3])

def update(i):
    m_im.set_data(m[i, ...])
    u_scatter.set_offsets(u[i, ...])
    return m_im, u_scatter

print("Animating...")
animation_fig = animation.FuncAnimation(fig, update, frames=steps,
                                        interval=40, blit=True)
animation_fig.save(f"magnetoelastic_{fp}.mp4")