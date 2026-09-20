"""Donut test, original version (eta = 1e10, 0.3 ns, no noise seeding)."""

import os
os.environ.setdefault("MUMAXPLUS_FP_PRECISION", "DOUBLE")   # must be set BEFORE importing mumaxplus

import numpy as np
import math
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.animation as animation

fp = os.environ["MUMAXPLUS_FP_PRECISION"].upper()
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

magnet.eta = 1e10
magnet.alpha = 0.004


# ==================================================================
# 2. Diagnostics
# ==================================================================

r_mesh = magnet.elastic_velocity.meshgrid              # (3, nz, ny, nx)
rho_flat = np.squeeze(magnet.rho.eval(), axis=0)       # (nz, ny, nx)

mask = magnet._get_mask_array(circles, grid, world, "mask").astype(bool)
mass_flat = rho_flat * mask

com = np.array([
    np.sum(mass_flat * r_mesh[i]) / np.sum(mass_flat)
    for i in range(3)
])

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

# ---- noise test: decouple magnetoelasticity, no mass damping, seed noise ----
magnet.B1 = 0
magnet.B2 = 0
magnet.eta = 0
magnet.stiffness_damping=0
world.timesolver.adaptive_timestep = False
world.timesolver.timestep = 1e-13
rng = np.random.default_rng(0)
magnet.elastic_displacement = (0, 0, 0)
magnet.elastic_velocity = 1e-2 * rng.standard_normal((3, nz, ny, nx)) * mask

import mumaxplus, glob, time
print("module:", mumaxplus.__file__, "| precision:", os.environ.get("MUMAXPLUS_FP_PRECISION"))
for f in glob.glob(os.path.join(os.path.dirname(mumaxplus.__file__), "**", "*.pyd"), recursive=True):
    print("  ", f, time.ctime(os.path.getmtime(f)))

rng = np.random.default_rng(1)
magnet.elastic_displacement = 1e-12 * rng.standard_normal((3, nz, ny, nx)) * mask
F = magnet.internal_body_force.eval()
print("net force / sum|F| per component:",
      F[:, mask].sum(axis=1) / np.abs(F[:, mask]).sum(axis=1))

V = cx * cy * cz
def E_kin(mag):
    return 0.5 * V * np.sum(rho_flat[None, ...] * mag.elastic_velocity.eval()**2)

def E_el(mag):
    e = mag.strain_tensor.eval()   # xx yy zz xy xz yz
    s = mag.elastic_stress.eval()
    w = np.array([1, 1, 1, 2, 2, 2])[:, None, None, None]
    return 0.5 * V * np.sum(w * e * s)

# ==================================================================
# 3. Excitation and simulation
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

steps = 400            # set to 20 for a quick smoke test
time_max = 5e-9
duration = time_max / steps

m_shape = np.transpose(magnet.magnetization.eval()[1, 0, :, :]).shape
u_shape = displacement_to_scatter_data(magnet, scale=u_scale, skip=u_skip).shape
m = np.zeros(shape=(steps, m_shape[0], m_shape[1]))
u = np.zeros(shape=(steps, u_shape[0], u_shape[1]))

P_hist    = np.zeros((steps, 3))
KE_hist   = np.zeros(steps)
umax_hist = np.zeros(steps)
ur_hist   = np.zeros(steps)
Ekin_hist = np.zeros(steps)
Eel_hist  = np.zeros(steps)
dt_hist   = np.zeros(steps)

print("Simulating...")
for i in tqdm(range(steps)):
    world.timesolver.run(duration)
    m[i, ...] = np.transpose(magnet.magnetization.eval()[1, 0, :, :])
    u[i, ...] = displacement_to_scatter_data(magnet, scale=u_scale, skip=u_skip)

    P_hist[i]    = total_momentum(magnet)
    KE_hist[i]   = kinetic_energy(magnet)
    umax_hist[i] = max_u(magnet)
    ur_hist[i]   = mean_radial_displacement(magnet)
    Ekin_hist[i] = E_kin(magnet)
    Eel_hist[i]  = E_el(magnet)
    dt_hist[i]   = world.timesolver.timestep

    if i % 40 == 0:
        print(f"step {i:4d}  |P|={np.linalg.norm(P_hist[i]):.3e}  "
              f"KE={KE_hist[i]:.3e}  max|u|={umax_hist[i]:.3e}  "
              f"<u_r>={ur_hist[i]:+.3e}")


# ==================================================================
# 4. Post-run diagnostic plot
# ==================================================================

Etot = Ekin_hist + Eel_hist
fig3, ax3 = plt.subplots(1, 3, figsize=(15, 4))
ax3[0].plot(Etot / Etot[0]); ax3[0].set_title("E_kin + E_el (normalized)")
ax3[1].semilogy(umax_hist);  ax3[1].set_title("max |u|")
ax3[2].plot(dt_hist);        ax3[2].set_title("timestep")
fig3.savefig(f"stability_{fp}.png", dpi=100)

fig2, axs = plt.subplots(2, 2, figsize=(10, 8))

axs[0, 0].plot(np.linalg.norm(P_hist, axis=1))
axs[0, 0].set_ylabel("|P|"); axs[0, 0].set_xlabel("step")
axs[0, 0].set_title(f"total momentum ({fp})")

axs[0, 1].plot(KE_hist)
axs[0, 1].set_ylabel("kinetic energy"); axs[0, 1].set_xlabel("step")
axs[0, 1].set_title("KE")

axs[1, 0].plot(umax_hist)
axs[1, 0].set_ylabel("max |u|"); axs[1, 0].set_xlabel("step")
axs[1, 0].set_title("peak displacement")

axs[1, 1].plot(ur_hist)
axs[1, 1].set_ylabel("<u_r>"); axs[1, 1].set_xlabel("step")
axs[1, 1].set_title("mean radial displacement")

fig2.tight_layout()
fig2.savefig(f"diagnostics_{fp}.png", dpi=100)
plt.close(fig2)

# Animation section omitted here to keep the run short; re-add it if you want the mp4.