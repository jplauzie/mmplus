"""Donut test, cleaned diagnostics for the elastic instability hunt."""

import os
os.environ.setdefault("MUMAXPLUS_FP_PRECISION", "DOUBLE")

import numpy as np
import math
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.ndimage import binary_erosion

fp = os.environ["MUMAXPLUS_FP_PRECISION"].upper()
print(f"Running with MUMAXPLUS_FP_PRECISION={fp}")

from mumaxplus import World, Grid, Ferromagnet
from mumaxplus.util import vortex
import mumaxplus.util.shape as shapes


# ==================================================================
# 1. Setup
# ==================================================================

length, width, thickness = 1e-6, 1e-6, 20e-9
nx, ny, nz = 256, 256, 1
cx, cy, cz = length/nx, width/ny, thickness/nz
cellsize = (cx, cy, cz)

grid  = Grid((nx, ny, nz))
world = World(cellsize)

circles = shapes.Rectangle(0.9 * nx * cx, 0.9 * ny * cy)
circles = circles.translate(nx * cx / 2, ny * cy / 2, 0)

magnet = Ferromagnet(world, grid, geometry=circles)

magnet.msat  = 1.2e6
magnet.aex   = 18e-12
magnet.alpha = 0.004
Bdc = 5e-3
magnet.bias_magnetic_field = (Bdc, 0, 0)

magnet.magnetization = vortex(magnet.center, 12e-9, -1, 1)
magnet.minimize()

magnet.enable_elastodynamics = True
magnet.rho  = 8e3
magnet.B1   = -8.8e6
magnet.B2   = -8.8e6
magnet.C11  = 283e9
magnet.C44  = 58e9
magnet.C12  = 166e9

magnet.elastic_displacement = (0, 0, 0)
magnet.elastic_velocity     = (0, 0, 0)

magnet.eta = 0.0
magnet.alpha = 0.004
magnet.stiffness_damping = 0.0

V = cx * cy * cz
rho_arr = np.asarray(magnet.rho.eval())
rho = float(rho_arr.max())
print(f"[setup] rho = {rho:.3e}   V = {V:.3e}")


# ==================================================================
# 2. Elastic operator diagnostics
# ==================================================================

mask  = magnet._get_mask_array(circles, grid, world, "mask").astype(bool)
mask3 = np.broadcast_to(mask, (3,) + mask.shape).astype(bool)
mask2d = mask[0]

interior2d = binary_erosion(mask2d, iterations=1)
int2d = np.argwhere(interior2d)
bnd2d = np.argwhere(mask2d & ~interior2d)
print(f"[setup] geometry cells = {int(mask.sum())}  interior = {len(int2d)}  boundary = {len(bnd2d)}")


def apply_K(u):
    magnet.elastic_displacement = u
    return np.asarray(magnet.internal_body_force.eval()).copy()


def K_col(b, y, x):
    u = np.zeros((3, 1, ny, nx))
    u[b, 0, y, x] = 1.0
    return apply_K(u)


# ------------------------------------------------------------------
# 2a. Energy identity: <u, Ku> should equal -V Σ ε:C:ε  (round-off)
# ------------------------------------------------------------------
print("\n=== Energy identity ===")
rng = np.random.default_rng(5)
u_test = rng.standard_normal((3, 1, ny, nx)) * mask3
magnet.elastic_displacement = u_test

K_u = np.asarray(magnet.internal_body_force.eval()).copy()
e   = magnet.strain_tensor.eval()
s   = magnet.elastic_stress.eval()
w   = np.array([1, 1, 1, 2, 2, 2])[:, None, None, None]

lhs = V * float(np.sum(u_test * K_u))
rhs = -V * float(np.sum(w * e * s))
print(f"  <u, Ku>          = {lhs:+.6e}")
print(f" -V Σ ε:C:ε        = {rhs:+.6e}")
rel = abs(lhs - rhs) / max(abs(lhs), abs(rhs), 1e-30)
print(f"  rel diff         = {rel:.3e}   (want ~1e-15)")
print(f"  verdict: {'CONSISTENT' if rel < 1e-10 else '*** FORCE AND STRESS DISAGREE ***'}")


# ------------------------------------------------------------------
# 2b. Symmetry check: 30 interior + 30 boundary cells, all (a,b) pairs
# ------------------------------------------------------------------
print("\n=== Broad symmetry check ===")
rng = np.random.default_rng(1)
int_sample = int2d[rng.choice(len(int2d), 200, replace=False)]
bnd_sample = bnd2d  # all 508
picks = np.concatenate([int_sample, bnd_sample], axis=0)

cols = {}
for (y, x) in picks:
    for b in range(3):
        cols[(int(y), int(x), b)] = K_col(b, int(y), int(x))

interior_set = set(map(tuple, int2d))
def tag(y, x):
    return "int" if (int(y), int(x)) in interior_set else "bnd"

worst_rel  = 0.0
worst_info = None
n_bad      = 0
for i, (py, px) in enumerate(picks):
    for j, (qy, qx) in enumerate(picks):
        if i >= j:
            continue
        for a in range(3):
            for b in range(3):
                k_pq = float(cols[(int(qy), int(qx), b)][a, 0, py, px])
                k_qp = float(cols[(int(py), int(px), a)][b, 0, qy, qx])
                scale = max(abs(k_pq), abs(k_qp), 1e-30)
                rel   = abs(k_pq - k_qp) / scale
                if rel > worst_rel:
                    worst_rel  = rel
                    worst_info = ((py, px), a, (qy, qx), b, k_pq, k_qp)
                if rel > 1e-6:
                    n_bad += 1

print(f"  cell pairs         = {len(picks)*(len(picks)-1)//2}")
print(f"  (a,b) combos/pair  = 9")
print(f"  bad entries (>1e-6)= {n_bad}")
print(f"  worst rel diff     = {worst_rel:.3e}")
if worst_info:
    (py, px), a, (qy, qx), b, kpq, kqp = worst_info
    print(f"  worst pair: {tag(py,px)}({py},{px}) a={a}   {tag(qy,qx)}({qy},{qx}) b={b}")
    print(f"      K_pq = {kpq:+.6e}   K_qp = {kqp:+.6e}")

# restore
magnet.elastic_displacement = (0, 0, 0)
magnet.elastic_velocity     = (0, 0, 0)


# ==================================================================
# 3. Pure-elastic energy conservation (the measurement)
# ==================================================================

print("\n=== Pure-elastic energy conservation, dt scan ===")
rng = np.random.default_rng(4)
u0 = rng.standard_normal((3, 1, ny, nx)) * mask3 * 1e-16
v0 = np.zeros_like(u0)

def KE():
    v = magnet.elastic_velocity.eval()
    return 0.5 * rho * V * float(np.sum(v**2))

def PE():
    e = magnet.strain_tensor.eval()
    s = magnet.elastic_stress.eval()
    return 0.5 * V * float(np.sum(w * e * s))

for dt in [1e-13, 1e-14, 1e-15]:
    magnet.elastic_displacement = u0
    magnet.elastic_velocity     = v0
    E0 = KE() + PE()
    world.timesolver.timestep = dt

    # same *physical times* for all dt, so we compare trajectories
    checkpoints = [1e-13, 5e-13, 1e-12, 2e-12, 5e-12]
    t_acc = 0.0
    hist  = []
    for t_target in checkpoints:
        steps_needed = int(round((t_target - t_acc) / dt))
        if steps_needed > 0:
            world.timesolver.run(steps_needed * dt)
            t_acc += steps_needed * dt
        hist.append((t_acc, KE() + PE()))

    print(f"  dt = {dt:.0e},  E0 = {E0:.4e}")
    for t, E in hist:
        print(f"     t = {t:.3e} s   E/E0 = {E/E0:.8f}")

# restore
magnet.elastic_displacement = (0, 0, 0)
magnet.elastic_velocity     = (0, 0, 0)


# ==================================================================
# 4. Diagnostics helpers for the simulation
# ==================================================================

r_mesh   = magnet.elastic_velocity.meshgrid
rho_flat = np.squeeze(magnet.rho.eval(), axis=0)
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

def E_kin(mag):
    return 0.5 * V * np.sum(rho_flat[None, ...] * mag.elastic_velocity.eval()**2)

def E_el(mag):
    e = mag.strain_tensor.eval()
    s = mag.elastic_stress.eval()
    return 0.5 * V * np.sum(w * e * s)


# ==================================================================
# 5. Excitation and simulation
# ==================================================================

f_B   = 9.8e9
Bac   = 0e-3
Bdiam = 200e-9

Bshape = shapes.Circle(Bdiam).translate(*magnet.center)
Bmask  = magnet._get_mask_array(Bshape, grid, world, "mask")
magnet.bias_magnetic_field.add_time_term(
    lambda t: (0, Bac * math.sin(2*math.pi*f_B * t), 0), mask=Bmask)

def displacement_to_scatter_data(mag, scale, skip):
    u = mag.elastic_displacement.eval()
    coords = mag.elastic_displacement.meshgrid + scale * u
    return np.transpose(coords[:2, 0, ::skip, ::skip].reshape(2, -1))

u_scale = 5e4
u_skip  = 5

steps    = 400
time_max = 0.3e-9
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

world.timesolver.disableAdaptiveTimeStep()
world.timesolver.timestep = 1e-14

print("\nSimulating...")
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
# 6. Post-run diagnostic plots
# ==================================================================

fig2, axs = plt.subplots(2, 2, figsize=(10, 8))
axs[0, 0].plot(np.linalg.norm(P_hist, axis=1));  axs[0, 0].set_ylabel("|P|")
axs[0, 1].plot(KE_hist);                         axs[0, 1].set_ylabel("kinetic energy")
axs[1, 0].plot(umax_hist);                       axs[1, 0].set_ylabel("max |u|")
axs[1, 1].plot(ur_hist);                         axs[1, 1].set_ylabel("<u_r>")
for ax in axs.flat:
    ax.set_xlabel("step")
axs[0, 0].set_title(f"total momentum ({fp})")
axs[0, 1].set_title("KE")
axs[1, 0].set_title("peak displacement")
axs[1, 1].set_title("mean radial displacement")

Etot = Ekin_hist + Eel_hist
fig3, ax3 = plt.subplots(1, 3, figsize=(15, 4))
ax3[0].plot(Etot / Etot[0]); ax3[0].set_title("E_kin + E_el (normalized)")
ax3[1].semilogy(umax_hist);  ax3[1].set_title("max |u|")
ax3[2].plot(dt_hist);        ax3[2].set_title("timestep")
fig3.savefig(f"stability_{fp}.png", dpi=100)

fig2.tight_layout()
fig2.savefig(f"diagnostics_{fp}.png", dpi=100)
plt.close(fig2)