"""Test script for the combined magnetic + elastic Minimizer.

Geometry and magnetic material parameters (CoFeB) match the companion
mumax3 Go test script, for direct comparison.

Workflow:
  1. Set up geometry + magnetic material, do a magnetic-only minimize.
  2. Enable elastodynamics, set elastic + magnetoelastic parameters.
  3. Run real coupled time-domain dynamics for DYNAMICS_TIME
     (world.timesolver.run in fixed-size chunks), letting m and u
     co-evolve under actual LLG + elastodynamics (with damping). This
     gives the minimizer a much better-conditioned starting point than a
     cold u=(0,0,0), matching how the mumax3 test reaches its starting
     state. Every SAVE_INTERVAL of simulated time, OVF snapshots of m,
     u, v, force (and a derived u_normalized) are written out, along
     with quick monitoring PNGs of m and normalized u.
  4. Run the combined magnetic + elastic minimize as a final cleanup.

Note: OVF export uses each FieldQuantity's own .save_ovf() method (confirmed
via mumaxplus's pyovf test suite), falling back to .npy then plain .txt if
that method is unavailable or errors. Derived arrays that have no backing
FieldQuantity (e.g. u_normalized) are saved via .npy/.txt directly, since
there's no live quantity object to call .save_ovf() on.

Note: periodic boundary conditions are intentionally NOT used here. With a
uniform magnetization and periodic boundaries, the magnetoelastic body force
is spatially uniform, so its divergence -- and therefore the relaxing force
on the elastic displacement -- is exactly zero everywhere. Using free
boundaries instead means the edges break that symmetry, giving the elastic
minimizer something real to relax.
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
import mumax_colorwheel as mcw

#os.environ["MUMAXPLUS_FP_PRECISION"] = "DOUBLE"

from mumaxplus import World, Grid, Ferromagnet
from mumaxplus.util import vortex, antivortex , twodomain
import mumaxplus.util.shape as shapes


# ============================================================
# I/O helpers
# ============================================================

OUTPUT_DIR = "miniresults"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def outpath(filename):
    """Prefix a bare filename with the shared OUTPUT_DIR."""
    return os.path.join(OUTPUT_DIR, filename)


def save_field(quantity, filename):
    """Save a FieldQuantity/Variable to an OVF file via its own .save_ovf()
    method, confirmed by mumaxplus's pyovf test suite (e.g.
    magnet.magnetization.save_ovf(filename)). Falls back to .npy (via
    quantity.eval()) if save_ovf isn't available or errors, then flattened
    .txt as a last resort.
    """
    filename = outpath(filename)
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


def save_array(array, base_filename):
    """Save a raw numpy array that has no associated FieldQuantity (e.g. a
    derived quantity like u_normalized) to .npy, falling back to a
    flattened .txt if that fails.
    """
    base_filename = outpath(base_filename)
    npy_name = base_filename + ".npy"
    try:
        np.save(npy_name, array)
        print(f"Saved {npy_name}")
        return
    except Exception as e:
        print(f".npy save failed for {npy_name} ({e}); falling back to .txt")

    txt_name = base_filename + ".txt"
    np.savetxt(txt_name, array.reshape(-1))
    print(f"Saved {txt_name} (flattened, original shape {array.shape})")


# ============================================================
# Reusable plotting primitives. Each draws into a supplied Axes (and
# returns the imshow handle, if any, for colorbar-ing). Shared by the
# quick in-loop monitoring PNGs, the final combined figure, and the
# final individually-saved panels.
# ============================================================

def plot_magnetization(ax, m2d, dx, dy, nx, ny, skip=4):
    Y, X = np.mgrid[0:ny, 0:nx]
    Xp = X[::skip, ::skip] * dx - 0.5 * dx
    Yp = Y[::skip, ::skip] * dy - 0.5 * dy
    ax.quiver(
        Xp, Yp,
        m2d[0, ::skip, ::skip],
        m2d[1, ::skip, ::skip],
        pivot="middle",
        scale=25,
    )
    ax.set_title("Magnetization (in-plane)")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_aspect("equal")
    return None


def plot_displacement_magnitude(ax, u2d, dx, dy, extent, skip=8):
    rgb = mcw.plot_vector_colorwheel(ax, u2d[0], u2d[1], dx, dy, extent,
                                      vz=u2d[2], mode="magnitude", skip=skip)
    ax.set_title("Elastic displacement (hue=direction, brightness=|u|)")
    return rgb  # RGB array, not a ScalarMappable -- no colorbar for this


def plot_velocity_magnitude(ax, v2d, extent):
    v_mag = np.linalg.norm(v2d, axis=0)
    im = ax.imshow(v_mag, origin="lower", extent=extent, cmap="viridis")
    ax.set_title("Elastic velocity |v|")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    return im


def plot_force_magnitude(ax, f2d, extent):
    f_mag = np.linalg.norm(f2d, axis=0)
    im = ax.imshow(f_mag, origin="lower", extent=extent, cmap="viridis")
    ax.set_title("Effective body force |f| (residual)")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    return im


def plot_normalized_displacement(ax, u2d, dx, dy, extent, skip=8):
    """u2d here is the RAW displacement (not yet normalized); this computes
    the magnitude-normalized version internally, plots it via the
    colorwheel, and also returns the normalized vector field in case the
    caller wants it."""
    u_mag = np.linalg.norm(u2d, axis=0)
    u_max = np.max(u_mag)
    u2d_norm = u2d / u_max if u_max > 0 else u2d

    rgb = mcw.plot_vector_colorwheel(ax, u2d_norm[0], u2d_norm[1], dx, dy, extent,
                                      vz=u2d_norm[2], mode="magnitude", vmax=1.0,
                                      skip=skip)
    ax.set_title("Normalized displacement (colorwheel)")
    return rgb, u2d_norm


def quick_plot_m(m2d, dx, dy, nx, ny, filename):
    """Lightweight single-panel PNG of magnetization, for in-loop monitoring."""
    filename = outpath(filename)
    fig, ax = plt.subplots(figsize=(5, 5))
    plot_magnetization(ax, m2d, dx, dy, nx, ny)
    fig.tight_layout()
    mcw.add_colorwheel_legend(fig, rect=(0.90, 0.75, 0.09, 0.18))
    fig.savefig(filename, dpi=120)
    plt.close(fig)
    print(f"Saved {filename}")


def quick_plot_u_normalized(u2d_raw, dx, dy, extent, filename):
    """Lightweight single-panel PNG of normalized |u|, for in-loop monitoring."""
    filename = outpath(filename)
    fig, ax = plt.subplots(figsize=(5, 5))
    plot_normalized_displacement(ax, u2d_raw, dx, dy, extent)
    fig.tight_layout()
    mcw.add_colorwheel_legend(fig, rect=(0.90, 0.75, 0.09, 0.18))
    fig.savefig(filename, dpi=120)
    plt.close(fig)
    print(f"Saved {filename}")


# ----- explicit minimizer settings, tweak these -----
TOL = 1e-6           # magnetic minimize tolerance
NSAMPLES = 10        # magnetic minimize sample count
TOL_EL = 3e-6       # elastic minimize tolerance
NSAMPLES_EL = 10     # elastic minimize sample count
STEPSIZE = 1e-14     # initial magnetic BB stepsize
STEPSIZE_EL = 1e-30  # initial elastic BB stepsize (only used as the very
                     # first guess -- the scaled BB update should adapt
                     # this quickly regardless of the exact starting value)
STEPSIZE_EL_FALLBACK=1e-30

DYNAMICS_TIME = 5e-9  # 500 ps warm-up run before minimize, matches mumax3 test
FIXED_DT = 1e-14       # matches mumax3 test's fixdt
RUN_CHUNK = 5e-11      # size of each timesolver.run() call in the warm-up loop
SAVE_INTERVAL = 1e-10   # save OVF/PNG snapshots roughly every 1 ns of sim time
                       # (with the default 500 ps DYNAMICS_TIME above, this
                       # means no periodic snapshot actually fires -- bump
                       # DYNAMICS_TIME up if you want to see some. Lower
                       # SAVE_INTERVAL for more frequent snapshots.)

# ----- geometry: matches mumax3 Go test script -----
dx, dy, dz = 4e-9, 4e-9, 30e-9
nx, ny, nz = 128, 32, 1
cellsize = (dx, dy, dz)
length, width = nx * dx, ny * dy
extent = (-0.5 * dx, length - 0.5 * dx, -0.5 * dy, width - 0.5 * dy)

grid = Grid((nx, ny, nz))
# No mastergrid/pbc_repetitions here -- see module docstring for why.
world = World(cellsize)
#circles=shapes.Circle(200e-9) - shapes.Circle(100e-9)

#circles=circles.translate(256e-9 / 2, 256e-9 / 2, 0)

magnet = Ferromagnet(
    world,
    grid,
    #geometry=circles
)


# ----- magnetic material: CoFeB, matches mumax3 Go test script -----
magnet.msat = 800e3
magnet.aex = 13e-12
magnet.alpha = 0.1
#magnet.magnetization = (1, 0, 0)  # high-energy start to exercise the descent
c = 4e-9
cx, cy, cz = magnet.center
offset = 0 * dx  # shift by a few cells, tune as needed
core = (cx + offset, cy-offset, cz)
#magnet.magnetization = vortex(magnet.center, 3*c, -1, 1)
dw = 10
#magnet.magnetization= twodomain((-1,0,0), (0,0,1), (1,0,0), nx*c/2, dw*c)
magnet.magnetization = (1, 0, 0)
#magnet.magnetization =(0.99503719,0.09950372,0)
#magnet.magnetization =(0.99503719,-0.09950372,0)

#magnet.ku1 = 1.5e6
#magnet.anisU = (0, 0, 1)
#magnet.magnetization = twodomain((0,0,-1), (1,0,0), (0,0,1), cx, dw*c)

Bdc = 5e-5
magnet.bias_magnetic_field = (Bdc, Bdc, 0)
# magnetic-only minimize first
#magnet.relax()
#magnet.minimize(TOL, NSAMPLES, stepsize=STEPSIZE)

# ----- elastic + magnetoelastic material: matches mumax3 Go test script -----
magnet.enable_elastodynamics = True
magnet.clean_elastic_rigid_modes = True
magnet.rho = 8e3
magnet.B1 = -8.8e6
magnet.B2 = -8.8e6
magnet.C11 = 283e9
magnet.C44 = 58e9
magnet.C12 = 166e9
magnet.eta = 5e13  # unused by minimize (damping term dropped), harmless to set
#magnet.ku1 = 1e4
#magnet.anisU = (1, 0, 0)

magnet.elastic_displacement = (0, 0, 0)

magnet.elastic_velocity = np.zeros_like(magnet.elastic_velocity.eval())

# ----- static, spatially-sinusoidal body force in y, no time dependence -----
Fac = 0          # start where you already have a working reference point
wavelength = 25.6e-9  # no longer needs to divide the domain evenly
sigma = 100e-9        # envelope width -- keep sigma/wavelength >~ 3-4, and
                       # envelope(edge)/envelope(peak) << 1

x0 = (nx*dx) / 2
x_centers = (np.arange(nx) + 0.5) * dx
envelope = np.exp(-((x_centers - x0) / sigma)**2)
profile = envelope * np.sin(2*np.pi*(x_centers - x0)/wavelength)
profile = np.broadcast_to(profile, (nz, ny, nx))

force_array = np.zeros((3, nz, ny, nx))
force_array[0] = Fac * profile   # x-component: longitudinal, not transverse
magnet.external_body_force = force_array

# ----- real coupled dynamics warm-up, with periodic snapshots -----
world.timesolver.adaptive_timestep = True
magnet.clean_elastic_rigid_modes = False

n_chunks = max(1, int(round(DYNAMICS_TIME / RUN_CHUNK)))
next_save_time = SAVE_INTERVAL

start = time.time()

for i in range(n_chunks):
    world.timesolver.run(RUN_CHUNK)
    t = world.timesolver.time
    print(f"time={t:.3e}, dt={world.timesolver.timestep:.3e}")

    if t >= next_save_time:
        tag = f"t{t * 1e9:.2f}ns"
        print(f"--- saving snapshot at {tag} ---")

        save_field(magnet.magnetization, f"m_{tag}.ovf")
        save_field(magnet.elastic_displacement, f"u_{tag}.ovf")
        save_field(magnet.elastic_velocity, f"v_{tag}.ovf")
        save_field(magnet.effective_body_force, f"force_{tag}.ovf")

        m_now = magnet.magnetization.eval()
        u_now = magnet.elastic_displacement.eval()
        u_mag_now = np.linalg.norm(u_now, axis=0)
        u_max_now = np.max(u_mag_now)
        u_norm_now = u_now / u_max_now if u_max_now > 0 else u_now.copy()
        save_array(u_norm_now, f"u_normalized_{tag}")

        quick_plot_m(m_now[:, 0, :, :], dx, dy, nx, ny, f"m_{tag}.png")
        quick_plot_u_normalized(u_now[:, 0, :, :], dx, dy, extent,
                                 f"u_normalized_{tag}.png")

        next_save_time += SAVE_INTERVAL

end = time.time()
print(f"Simulation runtime: {end - start:.3f} seconds")

#magnet.relax()

# combined magnetic + elastic minimize
print("Minimizing...")
#magnet.minimize(TOL, NSAMPLES, TOL_EL, NSAMPLES_EL, STEPSIZE_EL, STEPSIZE_EL_FALLBACK)

magnet.clean_elastic_rigid_modes = True
t0 = time.perf_counter()
magnet.minimize(TOL, NSAMPLES, TOL_EL, NSAMPLES_EL, STEPSIZE_EL, STEPSIZE_EL_FALLBACK, max_steps=20000,
                rigid_body_modes_interval=1,rigid_body_modes_delay=0,rigid_body_modes_method=0)
t_enabled = time.perf_counter() - t0

print(f"cost of removeRigidBodyModes over 5000 steps: {t_enabled: .3f}s")
print("Done.")

u = magnet.elastic_displacement.eval()
print(np.max(np.linalg.norm(u, axis=0)))
# manually remove average displacement (rigid-translation part) -- see
# chat explanation: this subtracts the per-component spatial mean of u
# from every cell, leaving only the spatially-varying (strain-relevant)
# part. It does NOT remove rigid rotation.
u = magnet.elastic_displacement.eval()
u_avg = magnet.elastic_displacement.average()
for i in range(3):
    u[i, ...] -= u_avg[i]
magnet.elastic_displacement = u
# sanity check: this should now print ~(0, 0, 0)
print(f"post-removal average displacement: {magnet.elastic_displacement.average()}")

# ----- gather final fields -----
m = magnet.magnetization.eval()               # shape (3, nz, ny, nx)
u = magnet.elastic_displacement.eval()        # shape (3, nz, ny, nx)
v = magnet.elastic_velocity.eval()            # shape (3, nz, ny, nx)
f = magnet.effective_body_force.eval()        # shape (3, nz, ny, nx)

print(f"max |u| = {np.max(np.linalg.norm(u, axis=0)):.3e} m")
print(f"max |v| = {np.max(np.linalg.norm(v, axis=0)):.3e} m/s")
print(f"max |f| = {np.max(np.linalg.norm(f, axis=0)):.3e} N/m3 ")
      
print(f"stress = {np.max(np.abs(magnet.stress_tensor.eval())):.3e} ")
      
      
      
print(f"total_energy = {magnet.total_energy.eval():.6e} J")
print(f"elastic_energy = {magnet.elastic_energy.eval():.6e} J")
print(f"kinetic_energy = {magnet.kinetic_energy.eval():.6e} J")


#def load_dump(path):
#    with open(path, "rb") as fh:
#        header = np.fromfile(fh, dtype=np.int32, count=4)
#        nx, ny, nz, ncomp = header
#        data = np.fromfile(fh, dtype=np.float32)  # float64 if built double-precision
#    data = data.reshape(ncomp, nz, ny, nx)  # component-major, matches mumax field layout
#    return data

#fbefore = load_dump("diag_force_before.bin")
#fafter  = load_dump("diag_force_after.bin")
#fdiff = fafter - fbefore

# mirror check along y (axis=2 in [comp, z, y, x]): compare row y against row (ny-1-y)
#ny = fdiff.shape[2]
#mirrored = fdiff[:, :, ::-1, :]

# for a genuinely antisymmetric (benign) perturbation under y -> -y,
# fdiff at mirrored y should equal -fdiff (for the y-component; x/z components
# may transform differently under the reflection -- check componentwise)
#asym_residual = fdiff + mirrored   # should be ~0 if perfectly antisymmetric

#print("max|fdiff| =", np.max(np.abs(fdiff)))
#print("max|fdiff + mirror(fdiff)| =", np.max(np.abs(asym_residual)))
#print("ratio =", np.max(np.abs(asym_residual)) / np.max(np.abs(fdiff)))

save_field(magnet.magnetization, "m.ovf")
save_field(magnet.elastic_displacement, "u.ovf")
save_field(magnet.elastic_velocity, "v.ovf")
save_field(magnet.effective_body_force, "force.ovf")

u_mag_full = np.linalg.norm(u, axis=0)
u_max_full = np.max(u_mag_full)
u_normalized_full = u / u_max_full if u_max_full > 0 else u.copy()
save_array(u_normalized_full, "u_normalized")

# squeeze the single z-layer for plotting
m2d = m[:, 0, :, :]
u2d = u[:, 0, :, :]
v2d = v[:, 0, :, :]
f2d = f[:, 0, :, :]

# --- combined figure, same 2x3 layout as before ---
fig, axes = plt.subplots(2, 3, figsize=(18, 7))

plot_magnetization(axes[0, 0], m2d, dx, dy, nx, ny)

plot_displacement_magnitude(axes[0, 1], u2d, dx, dy, extent)

im2 = plot_velocity_magnitude(axes[1, 0], v2d, extent)
plt.colorbar(im2, ax=axes[1, 0])

im3 = plot_force_magnitude(axes[1, 1], f2d, extent)
plt.colorbar(im3, ax=axes[1, 1])

plot_normalized_displacement(axes[0, 2], u2d, dx, dy, extent)

axes[1, 2].axis("off")  # unused panel in the original layout

fig.tight_layout()
mcw.add_colorwheel_legend(fig, rect=(0.90, 0.75, 0.09, 0.18))
fig.savefig(outpath("minimizer_test.png"), dpi=150)
print(f"Saved plot to {outpath('minimizer_test.png')}")

# --- also save each panel as its own standalone PNG ---
fig_i, ax_i = plt.subplots(figsize=(6, 5))
plot_magnetization(ax_i, m2d, dx, dy, nx, ny)
fig_i.tight_layout()
mcw.add_colorwheel_legend(fig_i, rect=(0.90, 0.75, 0.09, 0.18))
fig_i.savefig(outpath("final_magnetization.png"), dpi=150)
plt.close(fig_i)
print(f"Saved {outpath('final_magnetization.png')}")

fig_i, ax_i = plt.subplots(figsize=(6, 5))
plot_displacement_magnitude(ax_i, u2d, dx, dy, extent)
fig_i.tight_layout()
mcw.add_colorwheel_legend(fig_i, rect=(0.90, 0.75, 0.09, 0.18))
fig_i.savefig(outpath("final_displacement.png"), dpi=150)
plt.close(fig_i)
print(f"Saved {outpath('final_displacement.png')}")

fig_i, ax_i = plt.subplots(figsize=(6, 5))
im_i = plot_velocity_magnitude(ax_i, v2d, extent)
plt.colorbar(im_i, ax=ax_i)
fig_i.tight_layout()
fig_i.savefig(outpath("final_velocity.png"), dpi=150)
plt.close(fig_i)
print(f"Saved {outpath('final_velocity.png')}")

fig_i, ax_i = plt.subplots(figsize=(6, 5))
im_i = plot_force_magnitude(ax_i, f2d, extent)
plt.colorbar(im_i, ax=ax_i)
fig_i.tight_layout()
fig_i.savefig(outpath("final_force.png"), dpi=150)
plt.close(fig_i)
print(f"Saved {outpath('final_force.png')}")

fig_i, ax_i = plt.subplots(figsize=(6, 5))
plot_normalized_displacement(ax_i, u2d, dx, dy, extent)
fig_i.tight_layout()
mcw.add_colorwheel_legend(fig_i, rect=(0.90, 0.75, 0.09, 0.18))
fig_i.savefig(outpath("final_normalized_displacement.png"), dpi=150)
plt.close(fig_i)
print(f"Saved {outpath('final_normalized_displacement.png')}")

plt.show()
