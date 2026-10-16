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

from mumaxplus import Antiferromagnet, Ferromagnet, Grid, World
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
dx, dy, dz = 1e-9, 1e-9, 1e-9
nx, ny, nz = 128, 64, 1
cellsize = (dx, dy, dz)

grid = Grid((nx, ny, nz))
world = World(cellsize)
#circles = shapes.Circle(200e-9) - shapes.Circle(100e-9)
#circles = circles.translate(256e-9 / 2, 256e-9 / 2, 0)



# ----- magnetic material: CoFeB, matches mumax3 Go test script -----
Ms = 200e3
A0 = -400e-12
A = 10e-12
A12 = -15e-12
K = 1e3
a = 0.35e-9
mu0 = 1.256637062E-6
gamma = 1.7595E11 / (2 * np.pi) # gyromagnetic ratio in Hz/T
alpha = 0



magnet = Antiferromagnet(world, Grid((nx, ny, nz)))
world.set_pbc((1, 1, 0))
magnet.afmex_nn = -5e-12#A12
magnet.afmex_cell = -4e-12
magnet.latcon=a

for sub in magnet.sublattices:
    sub.msat = Ms
    sub.alpha = alpha
    sub.aex = 10e-12
    sub.ku1 = K
    sub.anisU = (1, 0, 0)

c = 4e-9
dw = 10
#magnet.magnetization = vortex(magnet.center, 3*c, -1, 1)
#magnet.magnetization = twodomain((-1, 0, 0), (0, 0, 1), (1, 0, 0), nx*c/2, dw*c)
#magnet.magnetization = (1, 0, 0)
#magnet.magnetization = (0.99503719, 0.09950372, 0)
#magnet.magnetization = (0.99503719, -0.09950372, 0)
np.random.seed(1234567)
#magnet.magnetization = np.random.normal(size=magnet.full_magnetization.shape)
np.random.seed(1234567)
for sub in magnet.sublattices:
    sub.magnetization = np.random.normal(size=sub.magnetization.shape)
#magnet.sub1.magnetization = (0.2, 0.8, 0.0)
#magnet.sub2.magnetization = (-0.2, -0.8, 0.0)

#for sub in magnet.sublattices:
#    sub.magnetization = np.random.normal(size=sub.magnetization.shape)

#magnet.ku1 = 1.5e6
#magnet.anisU = (0, 0, 1)

Bdc = 4
world.bias_magnetic_field = (Bdc, 0, 0)

# ----- minimize -----
print("Minimizing...")
t0 = time.perf_counter()
#world.relax()
world.minimize(TOL, NSAMPLES, line_search_mode=2, verbose=2)
t_min = time.perf_counter() - t0
print(f"Minimize runtime: {t_min:.3f} s")
print("Done.")

print("Sublattice 1 average:", magnet.sub1.magnetization.average())
print("Sublattice 2 average:", magnet.sub2.magnetization.average())
# ----- Plot AFM sublattices -----

m1 = magnet.sub1.magnetization.eval()
m2 = magnet.sub2.magnetization.eval()

print("Sublattice 1 shape:", m1.shape)
print("Sublattice 2 shape:", m2.shape)

# Extract the only z layer: expected shape (3, ny, nx)
m1 = m1[:, 0, :, :]
m2 = m2[:, 0, :, :]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for ax, m, title in zip(
    axes,
    [m1, m2],
    ["Sublattice 1", "Sublattice 2"],
):
    plot_magnetization(ax, m, dx, dy, nx, ny, skip=4)
    ax.set_title(title)

fig.tight_layout()
fig.savefig(outpath("afm_sublattices.png"), dpi=150)
plt.show()
