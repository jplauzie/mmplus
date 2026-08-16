"""mumax3-style HSV "colorwheel" rendering for 2D vector fields.

Reproduces the standard mumax3 look: hue encodes in-plane direction
(atan2(vy, vx)), saturation is fixed at 1, and brightness (value) encodes
magnitude (or, if you pass a z-component, lightness is pulled toward
white/black by +z/-z the way mumax3 renders magnetization).

  (1, 0, 0)  -> red        (hue = 0 deg)
  (0, 1, 0)  -> green-ish  (hue = 90 deg)
  (-1, 0, 0) -> cyan       (hue = 180 deg)
  (0, -1, 0) -> violet     (hue = 270 deg)

Drop this file next to your test script and `import mumax_colorwheel as mcw`.
"""

import numpy as np
from matplotlib.colors import hsv_to_rgb


def vector_to_rgb(vx, vy, vz=None, vmax=None, mode="magnitude"):
    """Convert a 2D array of vectors (vx, vy[, vz]) to an (ny, nx, 3) RGB image.

    Parameters
    ----------
    vx, vy : 2D arrays, same shape (ny, nx)
        In-plane vector components.
    vz : 2D array, optional
        Out-of-plane component. If given and mode="lightness", it pulls the
        color toward white (+z) / black (-z), mumax3-magnetization style.
    vmax : float, optional
        Value used to normalize magnitude to [0, 1]. Defaults to the max
        magnitude present in the field (so the brightest pixel is fully
        saturated color). Pass a fixed value if you want consistent
        brightness scaling across multiple snapshots/frames.
    mode : {"magnitude", "lightness", "direction_only"}
        "magnitude"      -- brightness (V in HSV) encodes |v|. Good default
                             for displacement/velocity/force fields where
                             the length varies and matters.
        "lightness"      -- like mumax3's magnetization plots: saturation
                             fixed at 1, hue = in-plane angle, and vz
                             (must be supplied) lightens/darkens via HSL.
                             Use this for near-unit-length fields.
        "direction_only" -- ignore magnitude entirely, full brightness
                             everywhere (pure direction map).

    Returns
    -------
    rgb : (ny, nx, 3) float array in [0, 1], ready for ax.imshow(rgb, ...).
    """
    vx = np.asarray(vx, dtype=float)
    vy = np.asarray(vy, dtype=float)

    theta = np.arctan2(vy, vx)                # [-pi, pi]
    hue = (theta / (2 * np.pi)) % 1.0          # [0, 1), 0=red matches mumax3

    mag = np.sqrt(vx**2 + vy**2 + (vz**2 if vz is not None else 0.0))

    if mode == "direction_only":
        sat = np.ones_like(hue)
        val = np.ones_like(hue)
        hsv = np.stack([hue, sat, val], axis=-1)
        return hsv_to_rgb(hsv)

    if mode == "lightness":
        if vz is None:
            raise ValueError("mode='lightness' requires vz")
        vz = np.asarray(vz, dtype=float)
        norm = np.where(mag > 0, mag, 1.0)
        vz_frac = np.clip(vz / norm, -1.0, 1.0)   # -1..1
        # HSL with L driven by vz_frac; S=1 in the "equatorial" plane and
        # tapering toward the poles (matches mumax3's magnetization color).
        sat = 1.0 - np.abs(vz_frac)
        light = 0.5 + 0.5 * vz_frac
        return _hsl_to_rgb(hue, sat, light)

    # mode == "magnitude" (default)
    if vmax is None:
        vmax = mag.max() if mag.max() > 0 else 1.0
    val = np.clip(mag / vmax, 0.0, 1.0)
    sat = np.ones_like(hue)
    hsv = np.stack([hue, sat, val], axis=-1)
    return hsv_to_rgb(hsv)


def _hsl_to_rgb(h, s, l):
    """Vectorized HSL->RGB, h/s/l in [0,1] arrays of identical shape."""
    c = (1 - np.abs(2 * l - 1)) * s
    hp = h * 6.0
    x = c * (1 - np.abs(np.mod(hp, 2) - 1))
    z = np.zeros_like(h)

    conds = [
        (hp >= 0) & (hp < 1), (hp >= 1) & (hp < 2), (hp >= 2) & (hp < 3),
        (hp >= 3) & (hp < 4), (hp >= 4) & (hp < 5), (hp >= 5) & (hp <= 6),
    ]
    r1 = np.select(conds, [c, x, z, z, x, c], default=z)
    g1 = np.select(conds, [x, c, c, x, z, z], default=z)
    b1 = np.select(conds, [z, z, x, c, c, x], default=z)

    m = l - c / 2
    rgb = np.stack([r1 + m, g1 + m, b1 + m], axis=-1)
    return np.clip(rgb, 0.0, 1.0)


def add_colorwheel_legend(fig, rect=(0.02, 0.02, 0.12, 0.12), n=200):
    """Add a small inset direction-legend wheel (hue vs angle, S=V=1),
    mimicking mumax3's on-plot color wheel. rect = (left, bottom, w, h)
    in figure-fraction coordinates.
    """
    ax = fig.add_axes(rect, projection=None)
    y, x = np.mgrid[-1:1:n * 1j, -1:1:n * 1j]
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    hue = (theta / (2 * np.pi)) % 1.0
    sat = np.ones_like(hue)
    val = np.ones_like(hue)
    hsv = np.stack([hue, sat, val], axis=-1)
    rgb = hsv_to_rgb(hsv)
    rgba = np.concatenate([rgb, (r <= 1)[..., None].astype(float)], axis=-1)
    ax.imshow(rgba, origin="lower", extent=(-1, 1, -1, 1))
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xlim(-1.15, 1.15)
    ax.set_ylim(-1.15, 1.15)
    return ax


def plot_vector_colorwheel(ax, vx, vy, dx, dy, extent, vz=None, mode="magnitude",
                            vmax=None, skip=8, quiver_color="white"):
    """Drop-in replacement for plot_displacement_magnitude /
    plot_normalized_displacement: colored background (direction=hue,
    magnitude=brightness) plus a light quiver overlay for local direction.
    Returns the RGB array actually plotted (useful if you want a shared
    vmax across multiple panels/snapshots).
    """
    rgb = vector_to_rgb(vx, vy, vz=vz, vmax=vmax, mode=mode)
    ax.imshow(rgb, origin="lower", extent=extent)

    ny_, nx_ = vx.shape
    Y, X = np.mgrid[0:ny_, 0:nx_]
    Xp = X[::skip, ::skip] * dx - 0.5 * dx
    Yp = Y[::skip, ::skip] * dy - 0.5 * dy
    ax.quiver(Xp, Yp, vx[::skip, ::skip], vy[::skip, ::skip],
              color=quiver_color, scale=None, alpha=0.85, width=0.003)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    return rgb