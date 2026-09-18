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


def strain_magnitude(strain2d):
    """strain2d: (6, ny, nx) array ordered [xx, yy, zz, xy, xz, yz]
    (matches the component order written by k_strainTensor in
    straintensor.cu). Returns the Frobenius norm of the full symmetric
    3x3 strain tensor at each cell:
        sqrt(xx^2 + yy^2 + zz^2 + 2*(xy^2 + xz^2 + yz^2))
    """
    xx, yy, zz, xy, xz, yz = strain2d
    return np.sqrt(xx**2 + yy**2 + zz**2 + 2 * (xy**2 + xz**2 + yz**2))


def plot_strain_magnitude(ax, strain2d, extent):
    s_mag = strain_magnitude(strain2d)
    im = ax.imshow(s_mag, origin="lower", extent=extent, cmap="viridis")
    ax.set_title("Strain magnitude (Frobenius norm)")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    return im


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


def quick_plot_strain(strain2d, extent, filename):
    """Lightweight single-panel PNG of strain magnitude, for in-loop
    monitoring (warm-up loop and periodic snapshots)."""
    filename = outpath(filename)
    fig, ax = plt.subplots(figsize=(5, 5))
    im = plot_strain_magnitude(ax, strain2d, extent)
    fig.colorbar(im, ax=ax, label="strain (dimensionless)")
    fig.tight_layout()
    fig.savefig(filename, dpi=120)
    plt.close(fig)
    print(f"Saved {filename}")


def fit_rigid_decoupled(r, v, w, com):
    """Weighted least-squares fit of a rigid translation + rotation to a
    vector field v(r), about a chosen origin `com`, with per-cell weight
    w. This is the decoupled shortcut form of

        c = (B^T W B)^{-1} B^T W v

    valid specifically because `com` is the W-weighted centroid, which
    makes the translation/rotation blocks of (B^T W B) decouple (see
    chat notes on the projection-operator P_W = B(B^T W B)^{-1} B^T W).
    """
    W = np.sum(w)
    T = np.sum(w * v, axis=(1, 2, 3)) / W          # removed [None, ...]

    r_rel = r - com[:, None, None, None]
    rx, ry, rz = r_rel[0], r_rel[1], r_rel[2]

    Ixx = np.sum(w * (ry**2 + rz**2))
    Iyy = np.sum(w * (rx**2 + rz**2))
    Izz = np.sum(w * (rx**2 + ry**2))
    Ixy = -np.sum(w * rx * ry)
    Ixz = -np.sum(w * rx * rz)
    Iyz = -np.sum(w * ry * rz)
    I = np.array([[Ixx, Ixy, Ixz],
                  [Ixy, Iyy, Iyz],
                  [Ixz, Iyz, Izz]])
    Iinv = np.linalg.inv(I)

    L = np.sum(w * np.cross(r_rel, v, axis=0), axis=(1, 2, 3))   # removed [None, ...]
    omega = Iinv @ L

    v_fit = T[:, None, None, None] + np.cross(omega[:, None, None, None], r_rel, axis=0)
    resid = np.sqrt(np.mean(np.sum((v - v_fit)**2, axis=0)))
    return T, omega, resid


# ============================================================
# Explicit rigid-mode projection operator P_W = B (B^T W B)^-1 B^T W.
# Used only by the optional diagnostic block below -- not part of the
# normal simulation flow. See chat notes: fit_rigid_decoupled() above is
# the cheap decoupled shortcut for exactly this operator, valid only
# when `com` is the W-weighted centroid.
# ============================================================

def build_B(r, origin, rotation_axes=(2,)):
    """r: (3, nz, ny, nx) meshgrid. origin: (3,) rotation center.
    Returns B with shape (3*ncells, 3 + len(rotation_axes)): 3
    translation columns + one rotation column per requested axis
    (0=x, 1=y, 2=z). Defaults to z-only rotation, matching a nz=1,
    in-plane-symmetric setup.
    """
    ncells = r[0].size
    r_flat = r.reshape(3, ncells)
    r_rel = r_flat - origin[:, None]

    cols = []
    for i in range(3):
        col = np.zeros((3, ncells))
        col[i, :] = 1.0
        cols.append(col.reshape(-1))

    for axis in rotation_axes:
        e = np.zeros(3)
        e[axis] = 1.0
        rot = np.cross(e[:, None], r_rel, axis=0)
        cols.append(rot.reshape(-1))

    return np.stack(cols, axis=1)


def build_P(r, origin, w, rotation_axes=(2,)):
    """w: (nz, ny, nx) scalar weight per cell (mass or geometric)."""
    B = build_B(r, origin, rotation_axes)
    w_flat = np.repeat(w.reshape(-1), 3)

    BtW = B.T * w_flat[None, :]
    BtWB = BtW @ B
    BtWB_inv = np.linalg.inv(BtWB)
    P = B @ BtWB_inv @ BtW
    return P


def run_rigid_removal_leak_test(magnet, world, nonrigid_field, r, com_geom,
                                 T0, theta0, amplitude=1.0, label=""):
    """Ground-truth leakage test for removeRigidBodyModes.

    Seeds elastic_displacement with a KNOWN rigid motion (T0, theta0)
    plus `amplitude * nonrigid_field`, lets ONE timesolver step run (this
    is what actually fires the C++ postStepCallback ->
    cleanElasticRigidModesCallback -> removeRigidBodyModes), and reports:

      1. strain computed from the non-rigid part ALONE (ground truth --
         what strain SHOULD look like if removal were perfectly clean)
      2. strain after the seeded field has gone through one real
         removal step
      3. the RMS difference between the two -- this is the leakage
         signature described in chat: over-removal shows up as strain
         that is smaller / different in pattern than the true non-rigid
         strain, because real deformation got misattributed as rigid
         and subtracted.

    NOTE: this does one full world.timesolver.steps(1), which also
    briefly evolves the field under real elastodynamics (and, if
    magnet.clean_elastic_rigid_modes is on, also exercises the
    acceleration-field removal inside evalElasticAcceleration). With
    velocity seeded to zero and a small fixed/adaptive dt this
    contamination should be small relative to the injected amplitude,
    but it is not perfectly isolated -- treat this as a strong signal,
    not a bit-exact unit test. The T/theta actually used for the
    removal are only visible via the C++ printf output (search the
    console for "[removeRigidBodyModes:u]" right after this call) --
    compare those printed values to the T0/theta0 printed below.

    Returns the RMS strain leakage (float).
    """
    r_rel_geo = r - com_geom[:, None, None, None]
    u_rigid = T0[:, None, None, None] + np.cross(theta0[:, None, None, None],
                                                  r_rel_geo, axis=0)
    u_nonrigid_scaled = amplitude * nonrigid_field

    # --- ground truth: strain of the non-rigid part alone ---
    magnet.elastic_displacement = u_nonrigid_scaled
    strain_reference = magnet.strain_tensor.eval()

    # --- seed the combined field and let one real removal step run ---
    u_seed = u_rigid + u_nonrigid_scaled
    magnet.elastic_displacement = u_seed
    magnet.elastic_velocity = np.zeros_like(magnet.elastic_velocity.eval())

    print(f"[leak_test:{label}] injected T0={T0}, theta0={theta0}")
    print(f"[leak_test:{label}] check console above/below for the C++ "
          f"'[removeRigidBodyModes:u] ... T=... theta=...' printout to "
          f"compare against T0/theta0")

    was_clean = magnet.clean_elastic_rigid_modes
    magnet.clean_elastic_rigid_modes = True
    world.timesolver.steps(1)
    magnet.clean_elastic_rigid_modes = was_clean

    strain_after = magnet.strain_tensor.eval()

    diff = strain_after - strain_reference
    rms_leak = float(np.sqrt(np.mean(diff**2)))
    rms_ref = float(np.sqrt(np.mean(strain_reference**2)))
    rel_leak = rms_leak / rms_ref if rms_ref > 0 else float("nan")

    print(f"[leak_test:{label}] rms(strain_reference)={rms_ref:.6e}  "
          f"rms(strain_after - strain_reference)={rms_leak:.6e}  "
          f"relative={rel_leak:.6e}")
    return rms_leak


def run_projection_diagnostic(r, com_mass, com_geom, w_rho, w_geo):
    """Builds P_mass and P_geom explicitly, checks their structural
    sanity (idempotent, correct rank), reports how much they disagree
    overall, and returns the displacement direction where they disagree
    most (via SVD of the difference) -- useful as a targeted worst-case
    perturbation for the seeded-injection test below.
    """
    P_mass = build_P(r, com_mass, w_rho)
    P_geom = build_P(r, com_geom, w_geo)

    print("idempotent check (mass):", np.max(np.abs(P_mass @ P_mass - P_mass)))
    print("idempotent check (geom):", np.max(np.abs(P_geom @ P_geom - P_geom)))
    print("rank(P_mass):", np.linalg.matrix_rank(P_mass))
    print("rank(P_geom):", np.linalg.matrix_rank(P_geom))

    diff = P_mass - P_geom
    diff_norm = np.linalg.norm(diff, ord=2)
    print(f"||P_mass - P_geom||_2 = {diff_norm:.4e}")

    U, S, Vt = np.linalg.svd(diff)
    print("largest singular value (max disagreement):", S[0])
    worst_direction = Vt[0].reshape(3, *r.shape[1:])
    return P_mass, P_geom, worst_direction


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

DYNAMICS_TIME = 0e-9  # 500 ps warm-up run before minimize, matches mumax3 test
FIXED_DT = 1e-14       # matches mumax3 test's fixdt
RUN_CHUNK = 5e-11      # size of each timesolver.run() call in the warm-up loop
SAVE_INTERVAL = 1e-10   # save OVF/PNG snapshots roughly every 1 ns of sim time
                       # (with the default 500 ps DYNAMICS_TIME above, this
                       # means no periodic snapshot actually fires -- bump
                       # DYNAMICS_TIME up if you want to see some. Lower
                       # SAVE_INTERVAL for more frequent snapshots.)

# ----- geometry: matches mumax3 Go test script -----
dx, dy, dz = 4e-9, 4e-9, 30e-9
nx, ny, nz = 32,32,1
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
#magnet.rho = 8e3
def rho_profile(x, y, z):
    # NOTE: threshold updated for the shrunk 32x32x1 grid (dx=4e-9) so
    # there's still a genuine density contrast in-domain. The original
    # (32+64)*4e-9 threshold was sized for the full 128x32x1 grid and
    # would put the whole shrunk domain at rho=1000 uniformly.
    return 9000.0 if ((x > (16)*4e-9)&(y>16*4e-9)) else 1000.0
magnet.rho.set(rho_profile)
magnet.B1 = -0e6
magnet.B2 = -0e6
magnet.C11 = 283e9
magnet.C44 = 58e9
magnet.C12 = 166e9
magnet.eta = 5e13  # unused by minimize (damping term dropped), harmless to set
#magnet.ku1 = 1e4
#magnet.anisU = (1, 0, 0)

# seed a real, localized, non-rigid displacement bump (not a rigid injection)
r_full = magnet.elastic_displacement.meshgrid   # (3, nz, ny, nx)
cx, cy, cz = magnet.center
x, y = r_full[0], r_full[1]
sigma = 60e-9
bump = 1e-11 * np.exp(-((x-cx)**2 + (y-cy)**2) / (2*sigma**2))
u0 = np.zeros_like(r_full)
u0[2] = bump   # out-of-plane bump, e.g. a small "dimple" -- genuinely non-rigid
magnet.elastic_displacement = u0
magnet.elastic_velocity = np.zeros_like(magnet.elastic_velocity.eval())


rho_eval = magnet.rho.eval()
r = magnet.elastic_velocity.meshgrid
com_mass = np.sum(rho_eval * r, axis=(1, 2, 3)) / np.sum(rho_eval)
com_geom = np.mean(r, axis=(1, 2, 3))
w_rho = np.squeeze(rho_eval, axis=0)      # (nz, ny, nx)
w_geo = np.ones_like(w_rho)               # (nz, ny, nx)


# ============================================================
# OPTIONAL DIAGNOSTIC BLOCK -- rigid-mode removal validation.
#
# Set RUN_RIGID_MODE_DIAGNOSTIC = True to run this instead of the normal
# seeded-bump simulation above. It overwrites elastic_displacement with a
# KNOWN rigid motion (T0, theta0) plus a deliberately asymmetric non-rigid
# perturbation, then lets you compare the T/theta recovered by
# removeRigidBodyModes (printed from the C++ side) against the injected
# T0/theta0 -- any discrepancy is leakage from the non-rigid part, which
# is the real test of "does this weighting correctly isolate the rigid
# subspace", as opposed to comparing raw strain magnitudes across runs
# (which mostly reflects accumulated trajectory differences, not the
# quality of a single removal step).
#
# This intentionally OVERWRITES magnet.elastic_displacement a second
# time (after the bump seeded above), so leave this off for a normal
# production/warm-up run.
# ============================================================
RUN_RIGID_MODE_DIAGNOSTIC = True

if RUN_RIGID_MODE_DIAGNOSTIC:
    # a genuinely asymmetric, off-center non-rigid part -- avoids the
    # even/odd-symmetry accident that made the original centered
    # Gaussian bump artificially favor the geometric fit
    sigma_x, sigma_y = 40e-9, 90e-9
    x_off, y_off = cx - 20e-9, cy + 35e-9
    x, y = r[0], r[1]
    u_nonrigid = np.zeros_like(r)
    u_nonrigid[2] = 1e-11 * np.exp(-((x - x_off)**2/(2*sigma_x**2) +
                                      (y - y_off)**2/(2*sigma_y**2)))

    T0 = np.array([0.0, 0.0, 3e-12])
    theta0 = np.array([0.0, 0.0, 5e-6])
    r_rel_geo = r - com_geom[:, None, None, None]
    u_rigid = T0[:, None, None, None] + np.cross(theta0[:, None, None, None],
                                                  r_rel_geo, axis=0)

    u_seed = u_rigid + u_nonrigid
    magnet.elastic_displacement = u_seed
    magnet.elastic_velocity = np.zeros_like(magnet.elastic_velocity.eval())

    print(f"[diagnostic] injected T0={T0}, theta0={theta0}")
    print("[diagnostic] compare against the T=... theta=... printed by "
          "removeRigidBodyModes on the next cleanElasticRigidModesCallback firing")

    # explicit projection-operator comparison (see build_P / run_projection_diagnostic)
    P_mass, P_geom, worst_direction = run_projection_diagnostic(
        r, com_mass, com_geom, w_rho, w_geo)
    # `worst_direction` is the displacement pattern P_mass and P_geom
    # disagree about most. It necessarily contains non-rigid content
    # (see chat: two projections onto the SAME subspace can only differ
    # on vectors outside that subspace), but to use it as a clean
    # "purely non-rigid" test vector, strip out whatever rigid part it
    # still has using either projection (arbitrary choice -- removing
    # something already known-rigid is unambiguous either way).
    worst_flat = worst_direction.reshape(-1)
    worst_rigid_part = P_geom @ worst_flat
    worst_nonrigid = (worst_flat - worst_rigid_part).reshape(3, nz, ny, nx)

    T0 = np.array([0.0, 0.0, 3e-12])
    theta0 = np.array([0.0, 0.0, 5e-6])
    amplitude = 1e-11 / np.max(np.abs(worst_nonrigid))  # scale to a physically reasonable size

    # NOTE: the actual mass-weighted vs. geometric choice for u/v removal
    # is controlled by the `unweighted` bool compiled into the
    # removeRigidBodyModes call sites in mumaxworld.cpp. This script
    # can't flip that at runtime -- run it once per weighting (rebuild
    # between runs) and compare the two printed "relative" leakage
    # numbers below; smaller relative leakage = that weighting preserved
    # more of the true non-rigid strain.
    print("\n--- leakage test (weighting set by current C++ build) ---")
    magnet.clean_elastic_rigid_modes = True
    run_rigid_removal_leak_test(magnet, world, worst_nonrigid, r, com_geom,
                                 T0, theta0, amplitude=0,
                                 label="current_build")


# ----- real coupled dynamics warm-up, with periodic snapshots -----
for i in range(1):
    was_clean = magnet.clean_elastic_rigid_modes
    magnet.clean_elastic_rigid_modes = True
    world.timesolver.run(FIXED_DT)   # was: world.timesolver.steps(1)
    magnet.clean_elastic_rigid_modes = was_clean
    if i % 100 == 0:
        v = magnet.elastic_velocity.eval()
        _, _, res_mass = fit_rigid_decoupled(r, v, w_rho, com_mass)
        _, _, res_geom = fit_rigid_decoupled(r, v, w_geo, com_geom)
        print(i, f"res_mass={res_mass:.3e}  res_geom={res_geom:.3e}  "
                 f"|v|={np.sqrt(np.mean(v**2)):.3e}")

        strain_now = magnet.strain_tensor.eval()   # (6, nz, ny, nx)
        quick_plot_strain(strain_now[:, 0, :, :], extent, f"strain_warmup_{i:05d}.png")




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
        save_field(magnet.strain_tensor, f"strain_{tag}.ovf")

        m_now = magnet.magnetization.eval()
        u_now = magnet.elastic_displacement.eval()
        strain_now = magnet.strain_tensor.eval()
        u_mag_now = np.linalg.norm(u_now, axis=0)
        u_max_now = np.max(u_mag_now)
        u_norm_now = u_now / u_max_now if u_max_now > 0 else u_now.copy()
        save_array(u_norm_now, f"u_normalized_{tag}")

        quick_plot_m(m_now[:, 0, :, :], dx, dy, nx, ny, f"m_{tag}.png")
        quick_plot_u_normalized(u_now[:, 0, :, :], dx, dy, extent,
                                 f"u_normalized_{tag}.png")
        quick_plot_strain(strain_now[:, 0, :, :], extent, f"strain_{tag}.png")

        next_save_time += SAVE_INTERVAL

end = time.time()
print(f"Simulation runtime: {end - start:.3f} seconds")

#magnet.relax()

# combined magnetic + elastic minimize
print("Minimizing...")
#magnet.minimize(TOL, NSAMPLES, TOL_EL, NSAMPLES_EL, STEPSIZE_EL, STEPSIZE_EL_FALLBACK)

magnet.clean_elastic_rigid_modes = True
t0 = time.perf_counter()
#magnet.minimize(TOL, NSAMPLES, TOL_EL, NSAMPLES_EL, STEPSIZE_EL, STEPSIZE_EL_FALLBACK, max_steps=0,
#                rigid_body_modes_interval=1,rigid_body_modes_delay=20000,rigid_body_modes_method=0)
t_enabled = time.perf_counter() - t0

#magnet.minimize(TOL, NSAMPLES, TOL_EL, NSAMPLES_EL, STEPSIZE_EL, STEPSIZE_EL_FALLBACK, max_steps=0,
#                rigid_body_modes_interval=1,rigid_body_modes_delay=0,rigid_body_modes_method=1)

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
