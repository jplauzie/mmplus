"""Magnetoelastic ring simulation (geometry/material from script 1) with an extended
animation: y-magnetization + amplified displacement, energies vs time, and
kinetic/elastic energy density maps (plotting taken from script 2).
"""

import math

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from mumaxplus import Ferromagnet, Grid, World
from mumaxplus.util import vortex
import mumaxplus.util.shape as shapes

J_to_eV = 6.24150907e18
DEBUG = False  # print NaN checks / net body force every step

# ---------------------------------------------------------------- setup
length, width, thickness = 1e-6, 1e-6, 20e-9
nx, ny, nz = 256, 256, 1
cx, cy, cz = length/nx, width/ny, thickness/nz
cellsize = (cx, cy, cz)

grid = Grid((nx, ny, nz))
world = World(cellsize)

circles = shapes.Circle(500e-9) - shapes.Circle(200e-9)
circles = circles.translate(1024e-9 / 2, 1024e-9 / 2, 0)

magnet = Ferromagnet(world, grid, geometry=circles)

magnet.msat = 1.2e6
magnet.aex = 18e-12
magnet.alpha = 0.004
Bdc = 5e-3
magnet.bias_magnetic_field = (Bdc, 0, 0)

magnet.magnetization = vortex(magnet.center, 12e-9, -1, 1)
magnet.minimize()

magnet.enable_elastodynamics = True
world.timesolver.adaptive_timestep = True
magnet.rho = 8e3
magnet.B1 = -8.8e6
magnet.B2 = -8.8e6
magnet.C11 = 283e9
magnet.C44 = 58e9
magnet.C12 = 166e9

magnet.elastic_displacement = (0, 0, 0)

# magnetic field excitation
f_B = 9.8e9
Bac = 0e-3
Bdiam = 200e-9

Bshape = shapes.Circle(Bdiam).translate(*magnet.center)
Bmask = magnet._get_mask_array(Bshape, grid, world, "mask")
magnet.bias_magnetic_field.add_time_term(
    lambda t: (0, Bac * math.sin(2*math.pi*f_B * t), 0), mask=Bmask)

magnet.alpha = 0.004
magnet.eta = 1e13  # elastic force damping


def displacement_to_scatter_data(magnet, scale, skip):
    """takes magnet.elastic_displacement.eval() array and turns it into an array
    of positions usable by plt.scatter().set_offsets"""
    u = magnet.elastic_displacement.eval()  # ((ux, uy, uz), Z, Y, X)
    coords = magnet.elastic_displacement.meshgrid + scale*u
    return np.transpose(coords[:2, 0, ::skip, ::skip].reshape(2, -1))  # (X*Y, 2)


u_scale = 5e2  # amplification of displacement
u_skip = 5     # don't show every displacement

steps = 4000
time_max = 5e-9
duration = time_max/steps

# ---------------------------------------------------------------- storage
m_shape = np.transpose(magnet.magnetization.eval()[1, 0, :, :]).shape
u_shape = displacement_to_scatter_data(magnet, scale=u_scale, skip=u_skip).shape

m = np.zeros((steps, *m_shape))
u = np.zeros((steps, *u_shape))
E_kin = np.zeros(steps)
E_el = np.zeros(steps)
u_max = np.zeros(steps)  # max |u| over the magnet (unamplified)
kin_dens = np.zeros((steps, *m_shape), dtype=np.float32)
el_dens = np.zeros((steps, *m_shape), dtype=np.float32)
t_ns = (np.arange(steps) + 1) * duration * 1e9

# ---------------------------------------------------------------- simulate
print("Simulating...")
for i in tqdm(range(steps)):
    world.timesolver.run(duration)

    if DEBUG:
        u0 = magnet.elastic_displacement.eval()
        m0 = magnet.magnetization.eval()
        print(np.isnan(u0).any(), np.abs(u0).max(), np.isnan(m0).any())
        f = magnet.internal_body_force.eval()
        print(f.reshape(3, -1).sum(axis=1))

    m[i, ...] = np.transpose(magnet.magnetization.eval()[1, 0, :, :])
    u[i, ...] = displacement_to_scatter_data(magnet, scale=u_scale, skip=u_skip)

    E_kin[i] = magnet.kinetic_energy.eval() * J_to_eV
    E_el[i] = magnet.elastic_energy.eval() * J_to_eV
    u_max[i] = np.abs(magnet.elastic_displacement.eval()).max()
    kin_dens[i] = np.transpose(magnet.kinetic_energy_density.eval()[0, 0, ...])
    el_dens[i] = np.transpose(magnet.elastic_energy_density.eval()[0, 0, ...])

# ---------------------------------------------------------------- figure
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(11, 10))
(ax1, ax2), (ax3, ax4) = axes

# ax1: y-magnetization + amplified displacement
im_extent = (-0.5*cx, length - 0.5*cx, -0.5*cy, width - 0.5*cy)
vmax = max(abs(m.max()), abs(m.min()))
m_im = ax1.imshow(m[0, ...], origin="lower", extent=im_extent,
                  vmin=-vmax, vmax=vmax, cmap="seismic")
offsets = displacement_to_scatter_data(magnet, scale=u_scale, skip=u_skip)
u_scatter = ax1.scatter(offsets[:, 0], offsets[:, 1], s=10, c="black",
                        marker=".", alpha=0.5)
cbar = fig.colorbar(m_im, ax=ax1, fraction=0.046, pad=0.04)
cbar.ax.set_ylabel(r"$<m_y>$", rotation=270, labelpad=12)
ax1.set_title("$m_y$ and amplified displacement")
ax1.set_xlabel("$x$ (m)")
ax1.set_ylabel("$y$ (m)")
ax1.set_xlim(im_extent[0], im_extent[1])
ax1.set_ylim(im_extent[2], im_extent[3])

# ax2: energies vs time (+ max displacement on a twin axis)
kinetic, = ax2.plot([], [], label="kinetic energy")
elastic, = ax2.plot([], [], label="elastic energy")
ax2.set_title("Energies and displacement")
ax2.set_xlabel("Time (ns)")
ax2.set_ylabel("Energy (eV)")
ax2.set_xlim(t_ns.min(), t_ns.max())
e_lo, e_hi = min(E_kin.min(), E_el.min()), max(E_kin.max(), E_el.max())
pad = 0.05 * (e_hi - e_lo if e_hi > e_lo else 1)
ax2.set_ylim(e_lo - pad, e_hi + pad)

ax2b = ax2.twinx()
disp_line, = ax2b.plot([], [], color="tab:green", ls="--", label="max |u|")
ax2b.set_ylabel("max |u| (m)")
ax2b.set_ylim(0, 1.05 * max(u_max.max(), 1e-30))

lines = [kinetic, elastic, disp_line]
ax2.legend(lines, [l.get_label() for l in lines], loc="upper left")

# ax3 / ax4: energy density maps
kin_im = ax3.imshow(kin_dens[0], origin="lower", vmin=0,
                    vmax=max(kin_dens.max(), 1e-30))
ax3.set_axis_off()
ax3.set_title("Kinetic Energy Density")
fig.colorbar(kin_im, ax=ax3, fraction=0.046, pad=0.04)

el_im = ax4.imshow(el_dens[0], origin="lower", vmin=0,
                   vmax=max(el_dens.max(), 1e-30))
ax4.set_axis_off()
ax4.set_title("Elastic Energy Density")
fig.colorbar(el_im, ax=ax4, fraction=0.046, pad=0.04)

plt.tight_layout()


def update(i):
    m_im.set_data(m[i, ...])
    u_scatter.set_offsets(u[i, ...])
    kinetic.set_data(t_ns[:i+1], E_kin[:i+1])
    elastic.set_data(t_ns[:i+1], E_el[:i+1])
    disp_line.set_data(t_ns[:i+1], u_max[:i+1])
    kin_im.set_array(kin_dens[i])
    el_im.set_array(el_dens[i])
    return m_im, u_scatter, kinetic, elastic, disp_line, kin_im, el_im


# ---------------------------------------------------------------- animate
print("Animating...")
animation_fig = animation.FuncAnimation(fig, update, frames=steps, interval=40, blit=True)
animation_fig.save("magnetoelastic.mp4")