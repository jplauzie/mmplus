"""
Compute the volume-averaged and summed vector value from an OOMMF OVF 2.0
text-format file (e.g. effective_body_force.ovf, elastic_displacement.ovf).

Usage:
    python ovf_average.py path/to/file.ovf [path/to/other_file.ovf ...]

What it does:
    - Parses the header for xstepsize/ystepsize/zstepsize (cell volume) and
      xnodes/ynodes/znodes.
    - Reads all (x, y, z) data triples.
    - Treats any triple that is exactly (0, 0, 0) as "outside geometry" /
      masked-out (this matches how mumaxplus writes zero for out-of-geometry
      cells) and excludes those from the average. This is a heuristic --
      if you have real physical cells that could legitimately be exactly
      zero in all 3 components, this will incorrectly exclude them, but
      for force/displacement fields on a vortex/nonzero-bias state this is
      very unlikely.
    - Reports, per component (x, y, z) and total magnitude:
        - mean over included (in-geometry) cells
        - sum over included cells
        - sum * cell_volume (a physically meaningful "net force" in N,
          if the input is a force density in N/m^3; or "net volume-integrated
          displacement" if the input is a displacement field, which is less
          physically standard but still useful as a bulk-average check)
        - number of included cells vs total cells

Why this matters: for a traction-free elastic body, the elastic
(internal_body_force) contribution must sum to exactly zero by the
divergence theorem. If effective_body_force's mean is clearly nonzero,
that nonzero mean can only be coming from a term that is NOT structurally
guaranteed net-zero (e.g. magnetoelastic_force computed without a matching
boundary traction correction) -- which points to a net, unbalanceable
force on the body rather than ordinary discretization noise.
"""

import sys
import re


def parse_ovf(path):
    with open(path, "r") as f:
        text = f.read()

    header = {}
    for line in text.splitlines():
        m = re.match(r"#\s*(\w+):\s*(.+)", line)
        if m:
            key, val = m.group(1), m.group(2).strip()
            header[key] = val

    # Extract data block
    start = text.index("# Begin: Data Text") + len("# Begin: Data Text")
    end = text.index("# End: Data Text")
    data_block = text[start:end]

    triples = []
    for line in data_block.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 3:
            continue
        triples.append(tuple(float(p) for p in parts))

    return header, triples


def cell_volume(header):
    dx = float(header.get("xstepsize", 1.0))
    dy = float(header.get("ystepsize", 1.0))
    dz = float(header.get("zstepsize", 1.0))
    return dx * dy * dz


def analyze(path):
    header, triples = parse_ovf(path)
    vol = cell_volume(header)

    total_cells = len(triples)
    included = [t for t in triples if not (t[0] == 0.0 and t[1] == 0.0 and t[2] == 0.0)]
    n = len(included)

    print(f"\n=== {path} ===")
    print(f"Title: {header.get('Title', '?')}")
    print(f"Grid: {header.get('xnodes','?')} x {header.get('ynodes','?')} x {header.get('znodes','?')}"
          f"  ({total_cells} total cells, {n} in-geometry / nonzero cells)")
    print(f"Cell volume: {vol:.6e} m^3")

    if n == 0:
        print("No nonzero cells found -- check the file / masking heuristic.")
        return

    sums = [0.0, 0.0, 0.0]
    for t in included:
        for i in range(3):
            sums[i] += t[i]

    means = [s / n for s in sums]
    net = [s * vol for s in sums]  # sum * cell_volume -> physically integrated quantity

    labels = ["x", "y", "z"]
    for i in range(3):
        print(f"  {labels[i]}: mean={means[i]: .6e}   sum={sums[i]: .6e}   "
              f"sum*cellVolume={net[i]: .6e}")

    mag_mean = (means[0] ** 2 + means[1] ** 2 + means[2] ** 2) ** 0.5
    # also compute mean of the per-cell magnitude, for comparison
    per_cell_mag_mean = sum((t[0]**2 + t[1]**2 + t[2]**2) ** 0.5 for t in included) / n
    print(f"  |mean vector| = {mag_mean:.6e}")
    print(f"  mean of per-cell |vector| = {per_cell_mag_mean:.6e}")
    print(f"  ratio |mean vector| / mean |vector| = {mag_mean / per_cell_mag_mean:.4f}"
          f"   (near 0 => noise averages out; near 1 => strong net/bias)")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python ovf_average.py file1.ovf [file2.ovf ...]")
        sys.exit(1)

    for path in sys.argv[1:]:
        analyze(path)