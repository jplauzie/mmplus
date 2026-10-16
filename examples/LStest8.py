"""Test script for the magnetic Minimizer (elastics disabled).

Geometry and magnetic material parameters (CoFeB) match the companion
mumax3 Go test script, for direct comparison.
"""

import os
import time

import matplotlib.pyplot as plt
import mumax_colorwheel as mcw
import numpy as np

os.environ["MUMAXPLUS_FP_PRECISION"] = "DOUBLE"

from mumaxplus import Ferromagnet, Grid, World
from mumaxplus.util import antivortex, twodomain, vortex
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
    method. Falls back to .npy (via quantity.eval()) if save_ovf isn't
    available or errors, then flattened .txt as a last resort.
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


# ============================================================
# Plotting
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


# ----- explicit minimizer settings, tweak these -----
TOL = 1e-6           # minimize tolerance
NSAMPLES = 10        # minimize sample count

# ----- geometry: matches mumax3 Go test script -----
dx, dy, dz = 4e-9, 4e-9, 4e-9
nx, ny, nz = 256, 128, 1
cellsize = (dx, dy, dz)

grid = Grid((nx, ny, nz))
world = World(cellsize)
#circles = shapes.Circle(200e-9) - shapes.Circle(100e-9)
#circles = circles.translate(256e-9 / 2, 256e-9 / 2, 0)

magnet = Ferromagnet(
    world,
    grid,
    #geometry=circles
)

# ----- magnetic material: CoFeB, matches mumax3 Go test script -----
magnet.msat = 800e3
magnet.aex = 15e-12
magnet.alpha = 0.1

c = 4e-9
dw = 10
#magnet.magnetization = vortex(magnet.center, 3*c, -1, 1)
#magnet.magnetization = twodomain((-1, 0, 0), (0, 0, 1), (1, 0, 0), nx*c/2, dw*c)
#magnet.magnetization = (1, 0, 0)
#magnet.magnetization = (0.99503719, 0.09950372, 0)
#magnet.magnetization = (0.99503719, -0.09950372, 0)
#np.random.seed(1234567)
#magnet.magnetization = np.random.normal(size=magnet.magnetization.shape)

#magnet.ku1 = 1.5e6
#magnet.anisU = (0, 0, 1)

Bdc = 5e-5
magnet.bias_magnetic_field = (Bdc, Bdc, 0)

# ----- minimize -----
N_RUNS = 1
MODE = 3             # line_search_mode to test
SAME_SEED = False    # True: identical start every run (timing); False: new seed each run

times = []
energies = []
for run in range(N_RUNS):
    seed = 1234567 if SAME_SEED else 1234567 + run
    np.random.seed(seed)
    #magnet.magnetization = vortex(magnet.center, 2*c, -1, 1)
    magnet.magnetization = twodomain((-1, 0, 0), (0, 0, 1), (1, 0, 0), nx*c/2, dw*c)

    t0 = time.perf_counter()
    magnet.minimize(TOL, NSAMPLES,max_step_angle=89, line_search_mode=MODE, verbose=3)
    dt = time.perf_counter() - t0
    E = magnet.total_energy.eval()
    times.append(dt)
    energies.append(E)

    print(f"run {run} (seed {seed}): {dt:.3f} s, total_energy = {E:.9e} J")

print(f"times: {['%.3f' % t for t in times]}")
print(f"min {min(times):.3f} s, median {sorted(times)[len(times) // 2]:.3f} s")
print(f"energies: {['%.6e' % e for e in energies]}")

# ----- results -----
m = magnet.magnetization.eval()  # shape (3, nz, ny, nx)
print(f"average m = {magnet.magnetization.average()}")
print(f"total_energy = {magnet.total_energy.eval():.6e} J")

save_field(magnet.magnetization, "m.ovf")

m2d = m[:, 0, :, :]

fig, ax = plt.subplots(figsize=(6, 5))
plot_magnetization(ax, m2d, dx, dy, nx, ny)
fig.tight_layout()
mcw.add_colorwheel_legend(fig, rect=(0.90, 0.75, 0.09, 0.18))
fig.savefig(outpath("final_magnetization.png"), dpi=150)
print(f"Saved {outpath('final_magnetization.png')}")

plt.show()