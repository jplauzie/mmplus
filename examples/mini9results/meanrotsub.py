"""
Subtract the best-fit rigid-body rotation from an OOMMF OVF 2.0 text-format
displacement field, to correct for a rotational drift that accumulates over
a long unconstrained magnetoelastic minimization/simulation but carries no
strain (a pure rotation, like a pure translation, doesn't affect the
physical state).

Usage:
    python ovf_subtract_rotation.py input.ovf [output.ovf]

If output.ovf is not given, defaults to <input>_derot.ovf

What it does:
    - Reuses the same "# Begin: Data Text" / "# End: Data Text" parsing as
      ovf_subtract_mean.py, and the same zero-triple masking heuristic for
      "outside geometry" cells (same caveat: a legitimate all-zero cell
      would be skipped).
    - Reconstructs each in-geometry cell's reference position r = (x, y, z)
      from the OVF header's grid geometry (xbase/xstepsize/xnodes, etc.),
      assuming standard OVF cell ordering (x fastest, then y, then z) and
      cell-center convention `pos_i = base_i + (index_i + 0.5) * step_i`.
    - Fits the single best-fit angular displacement vector omega (small-
      angle / infinitesimal rotation, valid since these are elastic
      displacements, not finite rotations) that minimizes
          sum_i || u_i - omega x (r_i - r_center) ||^2
      over the in-geometry cells, via ordinary least squares. r_center is
      the centroid of the in-geometry cell positions, so the fit isolates
      rotation from translation (translation should already have been
      removed by ovf_subtract_mean.py, or is now to run beforehand -- this
      script does not attempt to fit translation itself).
    - Subtracts omega x (r_i - r_center) from every in-geometry cell.
    - Writes the corrected file the same way as ovf_subtract_mean.py: only
      the data block is touched, header/footer copied through unmodified.
    - Prints the fitted omega vector and its magnitude for logging /
      sanity-checking.

Note: run ovf_subtract_mean.py first (or make sure the field is already
translation-free) -- this script does not jointly fit translation +
rotation, only rotation about the geometric centroid of the in-geometry
cells.

Note: like the companion script, this only handles OVF 2.0 "Data Text"
segments, not binary Data Binary 4/8 segments.
"""

import sys
import os
import re
import numpy as np


def parse_ovf(path):
    # See ovf_subtract_mean.py for why newline='' matters here.
    with open(path, "r", newline="") as f:
        text = f.read()

    header = {}
    for line in text.splitlines():
        m = re.match(r"#\s*(\w+):\s*(.+)", line)
        if m:
            key, val = m.group(1), m.group(2).strip()
            header[key] = val

    start_marker = "# Begin: Data Text"
    end_marker = "# End: Data Text"
    if start_marker not in text or end_marker not in text:
        raise ValueError(
            f"{path}: could not find '{start_marker}' / '{end_marker}' -- "
            "is this an OVF 2.0 text-format file?"
        )

    start = text.index(start_marker) + len(start_marker)
    end = text.index(end_marker)
    data_block = text[start:end]

    triples = []
    for line in data_block.splitlines():
        s = line.strip()
        if not s:
            continue
        parts = s.split()
        if len(parts) != 3:
            continue
        triples.append(tuple(float(p) for p in parts))

    return text, header, start, end, triples


def format_triple(t, fmt="{: .8e}"):
    return " ".join(fmt.format(v) for v in t)


def _grid_positions(header, ncells):
    """Reconstruct cell-center (x, y, z) reference positions from the OVF
    header, in standard OVF ordering (x fastest, then y, then z).
    """
    nx = int(float(header["xnodes"]))
    ny = int(float(header["ynodes"]))
    nz = int(float(header["znodes"]))
    if nx * ny * nz != ncells:
        raise ValueError(
            f"Header grid {nx}x{ny}x{nz} = {nx*ny*nz} cells does not match "
            f"{ncells} data triples -- check the file isn't truncated."
        )

    # xbase/ybase/zbase default to 0 if absent (some writers omit them).
    xbase = float(header.get("xbase", 0.0))
    ybase = float(header.get("ybase", 0.0))
    zbase = float(header.get("zbase", 0.0))
    dx = float(header["xstepsize"])
    dy = float(header["ystepsize"])
    dz = float(header["zstepsize"])

    iz, iy, ix = np.meshgrid(
        np.arange(nz), np.arange(ny), np.arange(nx), indexing="ij"
    )
    x = xbase + (ix.ravel() + 0.5) * dx
    y = ybase + (iy.ravel() + 0.5) * dy
    z = zbase + (iz.ravel() + 0.5) * dz
    return np.stack([x, y, z], axis=1)  # shape (ncells, 3)


def subtract_rotation(path, out_path=None, fmt="{: .8e}"):
    text, header, start, end, triples = parse_ovf(path)

    total_cells = len(triples)
    u = np.array(triples)  # shape (ncells, 3)
    positions = _grid_positions(header, total_cells)

    included_idx = np.array([
        i for i, t in enumerate(triples)
        if not (t[0] == 0.0 and t[1] == 0.0 and t[2] == 0.0)
    ])
    n = len(included_idx)

    if n == 0:
        raise ValueError(
            f"{path}: no in-geometry (nonzero) cells found -- cannot fit "
            "a rotation."
        )

    r = positions[included_idx]
    u_in = u[included_idx]

    r_center = r.mean(axis=0)
    r_rel = r - r_center

    # omega x r_rel = M(r_rel) @ omega, with M the skew-symmetric cross-
    # product matrix per cell. Stack into one big (3n, 3) system and solve
    # the least-squares problem for omega directly.
    M = np.zeros((3 * n, 3))
    rx, ry, rz = r_rel[:, 0], r_rel[:, 1], r_rel[:, 2]
    M[0::3] = np.stack([np.zeros(n), rz, -ry], axis=1)
    M[1::3] = np.stack([-rz, np.zeros(n), rx], axis=1)
    M[2::3] = np.stack([ry, -rx, np.zeros(n)], axis=1)

    u_flat = u_in.ravel()
    omega, *_ = np.linalg.lstsq(M, u_flat, rcond=None)

    print(f"=== {path} ===")
    print(
        f"Grid: {header.get('xnodes', '?')} x {header.get('ynodes', '?')} x "
        f"{header.get('znodes', '?')}  ({total_cells} total cells, "
        f"{n} in-geometry cells)"
    )
    print(
        f"Fitted rotation vector omega (rad): "
        f"x={omega[0]: .6e}  y={omega[1]: .6e}  z={omega[2]: .6e}"
    )
    print(f"|omega| = {np.linalg.norm(omega):.6e} rad")

    u_rot = M @ omega  # predicted rigid-rotation displacement, shape (3n,)
    u_rot = u_rot.reshape(n, 3)

    corrected = u.copy()
    corrected[included_idx] = u_in - u_rot

    new_data_lines = [format_triple(tuple(t), fmt) for t in corrected]
    new_data_block = "\n" + "\n".join(new_data_lines) + "\n"
    new_text = text[:start] + new_data_block + text[end:]

    if out_path is None:
        base, ext = os.path.splitext(path)
        out_path = f"{base}_derot{ext}"

    with open(out_path, "w", newline="") as f:
        f.write(new_text)

    print(f"Wrote rotation-corrected file: {out_path}\n")
    return out_path, omega


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python ovf_subtract_rotation.py input.ovf [output.ovf]")
        sys.exit(1)

    in_path = sys.argv[1]
    out_path = sys.argv[2] if len(sys.argv) > 2 else None
    subtract_rotation(in_path, out_path)