"""Pure elastic test on the ring geometry. Magnetoelastic coupling is disabled
(B1 = B2 = 0). The elastics are excited with the analytic radial breathing
eigenmode of the ring (Bessel J1/Y1 profile with stress-free edges), so only
that one mode should ring. Kinetic/elastic energy and the spectrum of a probe
displacement are plotted; the spectrum is compared with the predicted frequency.
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq
from scipy.special import jv, jvp, yv, yvp
from tqdm import tqdm

from mumaxplus import Ferromagnet, Grid, World
from mumaxplus.util import vortex
import mumaxplus.util.shape as shapes

J_to_eV = 6.24150907e18

# ---------------------------------------------------------------- setup
length, width, thickness = 1e-6, 1e-6, 20e-9
nx, ny, nz = 256, 256, 1
cx, cy, cz = length/nx, width/ny, thickness/nz
cellsize = (cx, cy, cz)

grid = Grid((nx, ny, nz))
world = World(cellsize, mastergrid=Grid((nx, ny, 0)), pbc_repetitions=(2, 2, 0))

# Note: shapes.Circle takes a DIAMETER
d_outer, d_inner = 500e-9, 200e-9
r_in, r_out = 0.5 * d_inner, 0.5 * d_outer
ring_center = (1024e-9 / 2, 1024e-9 / 2)
ring = (shapes.Circle(d_outer) - shapes.Circle(d_inner)).translate(*ring_center, 0)

magnet = Ferromagnet(world, grid, geometry=ring)

# magnetic part (left alone)
magnet.msat = 1.2e6
magnet.aex = 18e-12
magnet.alpha = 0.004
Bdc = 5e-3
magnet.bias_magnetic_field = (Bdc, 0, 0)

magnet.magnetization = vortex(magnet.center, 12e-9, -1, 1)
magnet.minimize()

# elastic part
rho = 8e3
C11, C44, C12 = 283e9, 58e9, 166e9
eta = 0e13  # elastic force damping

magnet.enable_elastodynamics = True
world.timesolver.adaptive_timestep = True
magnet.rho = rho
magnet.B1 = 0  # magnetoelastic coupling disabled (was -8.8e6)
magnet.B2 = 0  # magnetoelastic coupling disabled (was -8.8e6)
magnet.C11, magnet.C44, magnet.C12 = C11, C44, C12
magnet.eta = eta
magnet.stiffness_damping=0


# ---------------------------------------------------------------- breathing eigenmode
def breathing_mode(r_in, r_out, rho, c11, c12):
    """Lowest axisymmetric radial mode u_r(r) = A J1(kr) + B Y1(kr) of a free annulus.

    Solves rho*u_tt = c11*(u'' + u'/r - u/r^2) with stress-free edges
    sigma_rr = c11*u' + c12*u/r = 0 at r_in and r_out.
    Returns the profile function and the frequency in Hz.
    """
    def sigma_coeffs(k, r):  # sigma_rr of the J1 and Y1 solutions at radius r
        return (c11*k*jvp(1, k*r) + c12*jv(1, k*r)/r,
                c11*k*yvp(1, k*r) + c12*yv(1, k*r)/r)

    def det(k):
        a1, b1 = sigma_coeffs(k, r_in)
        a2, b2 = sigma_coeffs(k, r_out)
        return a1*b2 - a2*b1

    ks = np.linspace(1e6, 5e7, 4000)
    d = np.array([det(k) for k in ks])
    i = np.nonzero(d[:-1] * d[1:] < 0)[0][0]  # first sign change = lowest mode
    k = brentq(det, ks[i], ks[i+1])
    a1, b1 = sigma_coeffs(k, r_in)
    A, B = -b1, a1  # satisfies the BC at r_in; r_out is satisfied by the root k
    profile = lambda r: A*jv(1, k*r) + B*yv(1, k*r)
    return profile, k*np.sqrt(c11/rho)/(2*np.pi)


# Effective in-plane moduli. With a single cell in z it is not obvious whether the
# solver behaves like plane strain (eps_zz = 0) or plane stress (sigma_zz = 0);
# the spectrum below tells which one matches. Pick the one used for the initial shape:
PLANE = "strain"  # "strain" or "stress"

moduli = {"strain": (C11, C12),
          "stress": (C11 - C12**2/C11, C12 - C12**2/C11)}
modes = {name: breathing_mode(r_in, r_out, rho, *c) for name, c in moduli.items()}
for name, (_, f0) in modes.items():
    print(f"predicted breathing mode, plane {name}: {f0/1e9:.2f} GHz")

profile, f_mode = modes[PLANE]

# ---------------------------------------------------------------- initial displacement
amplitude = 1e-12  # peak radial displacement (m); linear problem, only scales energies

pos = magnet.elastic_displacement.meshgrid   # absolute cell positions, (3, Z, Y, X)
dx = pos[0] - ring_center[0]
dy = pos[1] - ring_center[1]
r = np.hypot(dx, dy)
r_safe = np.where(r > 0, r, 1.0)             # avoid 0/0 at the exact center
in_ring = (r >= r_in) & (r <= r_out)

rr = np.linspace(r_in, r_out, 1001)          # normalize so the peak value is +amplitude
p = profile(rr)
norm = p[np.argmax(np.abs(p))]
u_r = np.where(in_ring, amplitude * profile(np.clip(r, r_in, r_out)) / norm, 0.0)

u = np.zeros_like(pos)
u[0] = u_r * dx / r_safe
u[1] = u_r * dy / r_safe
magnet.elastic_displacement = u

# remove average displacement (rigid translation)
u = magnet.elastic_displacement.eval()
u_avg = magnet.elastic_displacement.average()
for i in range(3):
    u[i, ...] -= u_avg[i]
magnet.elastic_displacement = u

# probe cell on the +x side, middle of the ring (u_x there = radial displacement)
probe_target = (ring_center[0] + 0.5*(r_in + r_out), ring_center[1])
dist = np.hypot(pos[0, 0] - probe_target[0], pos[1, 0] - probe_target[1])
jy, ix = np.unravel_index(np.argmin(dist), dist.shape)

# ---------------------------------------------------------------- simulate
steps = 400
time_max = 0.5e-9
duration = time_max/steps

t_ns = (np.arange(steps) + 1) * duration * 1e9
E_kin = np.zeros(steps)
E_el = np.zeros(steps)
u_probe = np.zeros(steps)

print("Simulating...")
for i in tqdm(range(steps)):
    world.timesolver.run(duration)
    E_kin[i] = magnet.kinetic_energy.eval() * J_to_eV
    E_el[i] = magnet.elastic_energy.eval() * J_to_eV
    u_probe[i] = magnet.elastic_displacement.eval()[0, 0, jy, ix]

# ---------------------------------------------------------------- spectrum of the probe
window = np.hanning(steps)
nfft = 16 * steps  # zero padding, only interpolates the spectrum
spec = np.abs(np.fft.rfft((u_probe - u_probe.mean()) * window, nfft))
freqs_GHz = np.fft.rfftfreq(nfft, d=duration) / 1e9

# ---------------------------------------------------------------- plot
fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(7, 8))

ax1.plot(t_ns, E_kin, label="kinetic energy")
ax1.plot(t_ns, E_el, label="elastic energy")
ax1.set_xlabel("Time (ns)")
ax1.set_ylabel("Energy (eV)")
ax1.set_title("Elastic energies (no magnetoelastic coupling)")
ax1.legend()

ax2.semilogy(freqs_GHz, spec / spec.max())
for name, (_, f0) in modes.items():
    ax2.axvline(f0/1e9, ls="--", color="C1" if name == "strain" else "C2",
                label=f"predicted, plane {name}: {f0/1e9:.2f} GHz")
ax2.set_xlim(0, 40)
ax2.set_ylim(1e-4, 2)
ax2.set_xlabel("Frequency (GHz)")
ax2.set_ylabel("|FFT| of probe $u_r$ (norm.)")
ax2.set_title("Spectrum of radial displacement at a probe cell")
ax2.legend()

fig.tight_layout()
fig.savefig("elastic_energies.png", dpi=200)
plt.show()
