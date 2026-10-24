"""Pure elastic test on the ring geometry, compared with the exact analytic mode.

Magnetoelastic coupling and elastic damping are disabled. The elastics are
excited with the analytic radial breathing eigenmode of the ring (Bessel J1/Y1
profile, stress-free edges, plane strain), released from rest. For an undamped
eigenmode the exact time-dependent solution is

    u_r(r, t) = u_r(r, 0) * cos(2*pi*f*t)
    E_kin(t)  = E_tot * sin^2(2*pi*f*t),   E_el(t) = E_tot * cos^2(2*pi*f*t)

which is plotted against the simulation (energies, a probe cell, the radial
profile at the end of the run, and the spectrum).
"""

import time

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq
from scipy.special import jv, jvp, yv, yvp
from tqdm import tqdm

from mumaxplus import Ferromagnet, Grid, World
from mumaxplus.util import vortex
import mumaxplus.util.shape as shapes

J_to_eV = 6.24150907e18
trapz = getattr(np, "trapezoid", None) or np.trapz  # numpy 2.x / 1.x

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

# --- switches for tracking down the slowdown ---
CENTER_ON_CELL = False       # False = centering at 512 nm (between cells), noticeably faster
KILL_STIFFNESS_DAMPING = True  # False = leave the default stiffness damping on
FIXED_DT = 1e-13             # None = adaptive; e.g. 1e-13, then 5e-14, for a convergence test

if CENTER_ON_CELL:
    # Assumes cell (i, j) has its center at (i*cx, j*cy); verified against the grid below.
    ring_center = ((nx // 2) * cx, (ny // 2) * cy)
else:
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

magnet.enable_elastodynamics = True
if FIXED_DT is None:
    world.timesolver.adaptive_timestep = True
else:
    world.timesolver.adaptive_timestep = False
    world.timesolver.timestep = FIXED_DT
magnet.rho = rho
magnet.B1 = 0  # magnetoelastic coupling disabled (was -8.8e6)
magnet.B2 = 0  # magnetoelastic coupling disabled (was -8.8e6)
magnet.C11, magnet.C44, magnet.C12 = C11, C44, C12

# damping disabled. Stiffness (Rayleigh) damping is on by default, so switch it
# off explicitly as well as the mass-like damping eta.
magnet.eta = 0
if KILL_STIFFNESS_DAMPING:
    magnet.stiffness_damping = 0


# ---------------------------------------------------------------- analytic eigenmode
def breathing_mode(r_in, r_out, rho, c11, c12):
    """Lowest axisymmetric radial mode u_r(r) = A J1(kr) + B Y1(kr) of a free annulus.

    Solves rho*u_tt = c11*(u'' + u'/r - u/r^2) with stress-free edges
    sigma_rr = c11*u' + c12*u/r = 0 at r_in and r_out (plane strain, eps_zz = 0).
    Returns the profile u(r), its derivative u'(r) and the frequency in Hz.
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
    dprofile = lambda r: k*(A*jvp(1, k*r) + B*yvp(1, k*r))
    return profile, dprofile, k*np.sqrt(c11/rho)/(2*np.pi)


def mode_energy(u0, du0, rr, f, h, rho, c11, c12):
    """Peak kinetic energy (all energy in motion) and peak elastic energy (all
    energy in strain) of a radial mode with amplitude u0(r); they must be equal."""
    dV = 2*np.pi*rr*h
    E_kin_max = 0.5*rho*(2*np.pi*f)**2 * trapz(u0**2 * dV, rr)
    E_el_max = 0.5*trapz((c11*du0**2 + 2*c12*du0*u0/rr + c11*u0**2/rr**2) * dV, rr)
    return E_kin_max, E_el_max


profile, dprofile, f_mode = breathing_mode(r_in, r_out, rho, C11, C12)
w_mode = 2*np.pi*f_mode
print(f"predicted breathing mode (plane strain): {f_mode/1e9:.3f} GHz")

# ---------------------------------------------------------------- initial displacement
amplitude = 1e-12  # peak radial displacement (m); linear problem, only scales energies

rr = np.linspace(r_in, r_out, 4001)
p = profile(rr)
scale = amplitude / p[np.argmax(np.abs(p))]  # so the peak value is +amplitude

# analytic energy of this mode (continuum ring)
E_kin_max, E_el_max = mode_energy(scale*profile(rr), scale*dprofile(rr), rr,
                                  f_mode, thickness, rho, C11, C12)
print(f"analytic total energy: {E_kin_max*J_to_eV:.4e} eV "
      f"(from kinetic form) vs {E_el_max*J_to_eV:.4e} eV (from strain form)")
E_tot = E_kin_max

pos = magnet.elastic_displacement.meshgrid   # absolute cell positions, (3, Z, Y, X)

# check that the ring center sits on a cell center
d_cell = np.hypot(pos[0, 0] - ring_center[0], pos[1, 0] - ring_center[1]).min()
if CENTER_ON_CELL and d_cell > 1e-3 * min(cx, cy):
    print(f"WARNING: ring center is {d_cell/cx:.2f} cells away from the nearest "
          "cell center; check the cell-position convention")

dx = pos[0] - ring_center[0]
dy = pos[1] - ring_center[1]
r = np.hypot(dx, dy)
r_safe = np.where(r > 0, r, 1.0)             # avoid 0/0 at the exact center
in_ring = (r >= r_in) & (r <= r_out)

u_r = np.where(in_ring, scale * profile(np.clip(r, r_in, r_out)), 0.0)

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
r_probe = np.hypot(pos[0, 0, jy, ix] - ring_center[0], pos[1, 0, jy, ix] - ring_center[1])

# ---------------------------------------------------------------- simulate
steps = 400
time_max = 5e-9
duration = time_max/steps

t = (np.arange(steps) + 1) * duration
E_kin = np.zeros(steps)
E_el = np.zeros(steps)
u_probe = np.zeros(steps)

dts = np.zeros(steps)
t_run = 0.0

# mechanical (kinetic + elastic) energy at t = 0, the reference for the energy loss
E_mech0 = (magnet.kinetic_energy.eval() + magnet.elastic_energy.eval()) * J_to_eV
print(f"initial mechanical energy: {E_mech0:.4e} eV (analytic: {E_tot*J_to_eV:.4e} eV)")

print("Simulating...")
for i in tqdm(range(steps)):
    t0 = time.perf_counter()
    world.timesolver.run(duration)
    t_run += time.perf_counter() - t0
    dts[i] = world.timesolver.timestep
    E_kin[i] = magnet.kinetic_energy.eval() * J_to_eV
    E_el[i] = magnet.elastic_energy.eval() * J_to_eV
    u_probe[i] = magnet.elastic_displacement.eval()[0, 0, jy, ix]

print(f"run(): {t_run:.1f} s, timestep min/mean/max: "
      f"{dts.min():.2e} / {dts.mean():.2e} / {dts.max():.2e} s")

u_end = magnet.elastic_displacement.eval()

# ---------------------------------------------------------------- analytic curves
t_fine = np.linspace(0, time_max, 2000)
E_kin_an = E_tot * np.sin(w_mode * t_fine)**2 * J_to_eV
E_el_an = E_tot * np.cos(w_mode * t_fine)**2 * J_to_eV
u_probe_an = scale * profile(r_probe) * np.cos(w_mode * t_fine)

# radial cut along +x through the ring center at the end of the run
x_cut = pos[0, 0, jy, :] - ring_center[0]
cut = (x_cut >= r_in) & (x_cut <= r_out)
u_cut_sim = u_end[0, 0, jy, cut]
u_cut_an = scale * profile(x_cut[cut]) * np.cos(w_mode * time_max)

# ---------------------------------------------------------------- spectrum of the probe
window = np.hanning(steps)
nfft = 16 * steps  # zero padding, only interpolates the spectrum
spec = np.abs(np.fft.rfft((u_probe - u_probe.mean()) * window, nfft))
freqs_GHz = np.fft.rfftfreq(nfft, d=duration) / 1e9
near = np.abs(freqs_GHz - f_mode/1e9) < 2
f_peak = freqs_GHz[near][np.argmax(spec[near])]
print(f"simulated peak: {f_peak:.2f} GHz  (predicted {f_mode/1e9:.3f} GHz)")

# ---------------------------------------------------------------- plot
fig, axd = plt.subplot_mosaic([["en", "probe"], ["prof", "spec"], ["loss", "loss"]],
                              figsize=(12, 11))
ax1, ax2, ax3, ax4, ax5 = (axd["en"], axd["probe"], axd["prof"], axd["spec"], axd["loss"])

ax1.plot(t*1e9, E_kin, "C0", label="kinetic (sim)")
ax1.plot(t*1e9, E_el, "C1", label="elastic (sim)")
ax1.plot(t_fine*1e9, E_kin_an, "k--", lw=1, label="analytic")
ax1.plot(t_fine*1e9, E_el_an, "k--", lw=1)
ax1.set_xlabel("Time (ns)")
ax1.set_ylabel("Energy (eV)")
ax1.set_title("Energies")
ax1.legend()

ax2.plot(t*1e9, u_probe, "C0.", ms=4, label="sim")
ax2.plot(t_fine*1e9, u_probe_an, "k--", lw=1, label="analytic")
ax2.set_xlabel("Time (ns)")
ax2.set_ylabel("$u_r$ at probe (m)")
ax2.set_title(f"Probe at r = {r_probe*1e9:.0f} nm")
ax2.legend()

ax3.plot(x_cut[cut]*1e9, u_cut_sim, "C0o", ms=3, label="sim")
ax3.plot(x_cut[cut]*1e9, u_cut_an, "k--", lw=1, label="analytic $J_1/Y_1$")
ax3.set_xlabel("r (nm)")
ax3.set_ylabel("$u_r$ (m)")
ax3.set_title(f"Radial profile at t = {time_max*1e9:.1f} ns")
ax3.legend()

ax4.semilogy(freqs_GHz, spec / spec.max())
ax4.axvline(f_mode/1e9, ls="--", color="k", label=f"predicted: {f_mode/1e9:.2f} GHz")
ax4.set_xlim(0, 40)
ax4.set_ylim(1e-4, 2)
ax4.set_xlabel("Frequency (GHz)")
ax4.set_ylabel("|FFT| of probe $u_r$ (norm.)")
ax4.set_title("Spectrum")
ax4.legend()

# change of the mechanical energy: it should be conserved without damping, so any
# change here is numerical. Positive = energy gained, negative = energy lost.
E_mech = E_kin + E_el
dE_pct = 100 * (E_mech - E_mech0) / E_mech0            # relative to t = 0
dE_pct_1 = 100 * (E_mech - E_mech[0]) / E_mech[0]      # relative to the first sample

# --- decompose the energy change into a slow trend and an oscillation ---
# Fit  y = a + b*t + q*t^2 + c*cos(Om*t) + s*sin(Om*t)  (t in ns), scanning Om around the
# energy-exchange frequency (2f). Then compare the oscillating part of E_mech with
# that of E_kin: for a consistent system E_mech has none. If the reported kinetic
# energy were scaled by alpha and evaluated delta later than the elastic energy,
# Z_mech / Z_kin = alpha * exp(i*Om*delta) - 1.
def fit_osc(tn, y, Om):
    M = np.column_stack([np.ones_like(tn), tn, tn**2, np.cos(Om*tn), np.sin(Om*tn)])
    coef = np.linalg.lstsq(M, y, rcond=None)[0]
    return coef, np.sum((M @ coef - y)**2)

tn = t * 1e9
Om_scan = 2*np.pi * np.linspace(0.9, 1.1, 801) * 2 * f_mode/1e9   # rad/ns
Om = Om_scan[np.argmin([fit_osc(tn, E_kin, W)[1] for W in Om_scan])]
ck, _ = fit_osc(tn, E_kin, Om)
cm, _ = fit_osc(tn, E_mech, Om)
Zk, Zm = ck[3] - 1j*ck[4], cm[3] - 1j*cm[4]
ratio = 1 + Zm/Zk
alpha, delta = abs(ratio), np.angle(ratio) / Om * 1e-9   # delta in s
base_pct = 100 * (cm[0] + cm[1]*tn + cm[2]*tn**2 - E_mech0) / E_mech0   # slow trend, % of E0
# test of the "pure scale mismatch" diagnosis: if rescaling the kinetic energy by
# 1/alpha flattens the oscillation, the mismatch is a constant factor
E_corr = E_el + E_kin / alpha
dE_corr_pct = 100 * (E_corr - E_mech0) / E_mech0
print(f"energy-exchange fit: f = {Om/(2*np.pi)/2:.3f} GHz (mode), "
      f"predicted {f_mode/1e9:.3f} GHz")
print(f"  oscillation of E_mech: amplitude {100*abs(Zm)/E_mech0:.3f} % of E0")
print(f"  slow trend of E_mech: slope {100*cm[1]/E_mech0:+.3f} % of E0 per ns at t=0, "
      f"curvature {100*cm[2]/E_mech0:+.3f} % of E0 per ns^2")
print(f"  kinetic-vs-elastic amplitude mismatch alpha-1 = {100*(alpha-1):+.3f} %, "
      f"time offset delta = {delta*1e12:+.3f} ps (mean timestep {dts.mean()*1e12:.3f} ps)")
ax5.plot(t*1e9, dE_pct, label="relative to $t=0$")
ax5.plot(t*1e9, dE_pct_1, "--", label="relative to first sample")
ax5.plot(t*1e9, dE_corr_pct, label=r"kinetic rescaled by $1/\alpha$")
ax5.plot(t*1e9, base_pct, "k:", label="fitted slow trend")
ax5.axhline(0, color="k", lw=0.5)
ax5.set_xlabel("Time (ns)")
ax5.set_ylabel("Energy change (% of initial)")
ax5.set_title("Mechanical energy change (kinetic + elastic); negative = lost")
ax5.legend()
print(f"  peak-to-peak of the energy change: {np.ptp(dE_pct):.3f} % as reported, "
      f"{np.ptp(dE_corr_pct):.3f} % with kinetic rescaled by 1/alpha")
print(f"mechanical energy change by the end: {dE_pct[-1]:+.3e} % (vs t=0), "
      f"{dE_pct_1[-1]:+.3e} % (vs first sample); jump in the first step: {dE_pct[0]:+.3e} %")

fig.tight_layout()
fig.savefig("elastic_energies.png", dpi=200)
plt.show()
