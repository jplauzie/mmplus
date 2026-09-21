"""Production-style donut test: couplings on, default stiffness damping, zero excitation.

Environment variables (all optional):
  MUMAXPLUS_FP_PRECISION  SINGLE | DOUBLE                       (default DOUBLE)
  RK_METHOD               Heun | BogackiShampine | CashKarp | Fehlberg | DormandPrince  (default Fehlberg)
  T_MAX                   simulated time in seconds             (default 5e-9)
  STEPS                   number of output chunks               (default 400)
Saves hist_<precision>_<method>.npz for comparing runs.
"""
import os, time, math
#os.environ.setdefault("MUMAXPLUS_FP_PRECISION", "DOUBLE")   # must be set BEFORE importing mumaxplus
import numpy as np
from tqdm import tqdm

fp = os.environ["MUMAXPLUS_FP_PRECISION"].upper()
method = os.environ.get("RK_METHOD", "Fehlberg")
time_max = float(os.environ.get("T_MAX", 5e-9))
steps = int(os.environ.get("STEPS", 400))
tag = f"{fp}_{method}"
print(f"precision={fp}  method={method}  T_MAX={time_max:.2e}  STEPS={steps}")

import mumaxplus
from mumaxplus import World, Grid, Ferromagnet
from mumaxplus.util import vortex
import mumaxplus.util.shape as shapes
print("module:", mumaxplus.__file__)

# ==================================================================
# 1. Setup (same as your original script)
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
magnet.bias_magnetic_field = (5e-3, 0, 0)

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

magnet.eta = 1e10          # your production value from the original script; edit if different
# stiffness_damping is deliberately NOT set -> default 0.05/(1e12*pi) s

def pmax(p):
    try:
        return float(np.max(np.abs(p.eval())))
    except Exception as e:
        return f"n/a ({e})"
print("parameters:",
      f"B1={pmax(magnet.B1):.3g} B2={pmax(magnet.B2):.3g} C11={pmax(magnet.C11):.3g} "
      f"C12={pmax(magnet.C12):.3g} C44={pmax(magnet.C44):.3g} rho={pmax(magnet.rho):.3g} "
      f"eta={pmax(magnet.eta):.3g} stiffness_damping={pmax(magnet.stiffness_damping)}")

# ==================================================================
# 2. Excitation (kept at zero, as in all your runs)
# ==================================================================
f_B = 9.8e9
Bac = 0.0                  # set to your real excitation amplitude when ready
Bshape = shapes.Circle(200e-9).translate(*magnet.center)
Bmask = magnet._get_mask_array(Bshape, grid, world, "mask")
magnet.bias_magnetic_field.add_time_term(
    lambda t: (0, Bac * math.sin(2*math.pi*f_B * t), 0), mask=Bmask)

# ==================================================================
# 3. Time solver (call set_method after all parameters are set)
# ==================================================================
ts = world.timesolver
ts.set_method(method)
print("initial timestep:", ts.timestep)

# ==================================================================
# 4. Diagnostics (per-volume-summed units, same as your earlier KE prints)
# ==================================================================
r_mesh = magnet.elastic_velocity.meshgrid
rho_flat = np.squeeze(magnet.rho.eval(), axis=0)
mask = magnet._get_mask_array(circles, grid, world, "mask").astype(bool)
mass_flat = rho_flat * mask
com = np.array([np.sum(mass_flat * r_mesh[i]) / np.sum(mass_flat) for i in range(3)])
dx, dy = r_mesh[0] - com[0], r_mesh[1] - com[1]
rr = np.sqrt(dx**2 + dy**2) + 1e-30
er_x, er_y = dx / rr, dy / rr
w6 = np.array([1, 1, 1, 2, 2, 2])[:, None, None, None]

def diagnostics(mag):
    v = mag.elastic_velocity.eval()
    u = mag.elastic_displacement.eval()
    P = np.sum(rho_flat[None] * v, axis=(1, 2, 3))
    pscale = np.sum(rho_flat[None] * np.abs(v)) + 1e-300
    KE = 0.5 * np.sum(rho_flat[None] * v**2)                 # sum over cells of 0.5*rho*v^2
    Eel = 0.5 * np.sum(w6 * mag.strain_tensor.eval() * mag.elastic_stress.eval())  # same units
    Eme = float(mag.magnetoelastic_energy_density.eval()[0][mask].sum())           # same units
    umax = float(np.max(np.abs(u)))
    ur = float(np.mean((u[0] * er_x + u[1] * er_y)[mask]))
    mavg = mag.magnetization.eval()[:, mask].mean(axis=1)
    return dict(KE=KE, Eel=Eel, Eme=Eme, umax=umax, ur=ur,
                dP=float(np.linalg.norm(P) / pscale), mavg=mavg)

# ==================================================================
# 5. Run
# ==================================================================
duration = time_max / steps
keys = ["t", "dt", "KE", "Eel", "Eme", "umax", "ur", "dP", "wall"]
H = {k: np.zeros(steps) for k in keys}
H["mavg"] = np.zeros((steps, 3))
n_done = steps
t_start = time.time()
print_every = max(1, steps // 10)

print("Simulating...")
for i in tqdm(range(steps)):
    tw = time.time()
    ts.run(duration)
    d = diagnostics(magnet)
    H["t"][i], H["dt"][i], H["wall"][i] = ts.time, ts.timestep, time.time() - tw
    for k in ("KE", "Eel", "Eme", "umax", "ur", "dP"):
        H[k][i] = d[k]
    H["mavg"][i] = d["mavg"]

    if not (np.isfinite(d["umax"]) and np.isfinite(d["KE"])) or d["umax"] > 1e-7:
        print(f"ABORT at chunk {i}: max|u|={d['umax']:.3e} KE={d['KE']:.3e}")
        n_done = i + 1
        break
    if i % print_every == 0 or i == steps - 1:
        print(f"chunk {i:4d} t={ts.time*1e9:6.3f}ns dt={ts.timestep:.2e} KE={d['KE']:.4e} "
              f"Eel={d['Eel']:.4e} Eme={d['Eme']:.4e} Etot={d['KE']+d['Eel']+d['Eme']:.3e} "
              f"max|u|={d['umax']:.3e} <u_r>={d['ur']:+.3e} "
              f"|P|/sum(rho|v|)={d['dP']:.1e} <m>=({d['mavg'][0]:+.4f},{d['mavg'][1]:+.4f},{d['mavg'][2]:+.4f}) "
              f"wall={H['wall'][i]:.1f}s")

print(f"total wall time: {time.time() - t_start:.0f} s   chunks done: {n_done}")
np.savez(f"hist_{tag}.npz", **{k: v[:n_done] for k, v in H.items()})
print(f"saved hist_{tag}.npz")