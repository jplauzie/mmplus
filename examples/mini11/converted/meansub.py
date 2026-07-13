"""
Subtract the mean vector from an OOMMF OVF 2.0 text-format file to correct
for uniform drift (e.g. a rigid-body displacement offset that accumulates
over a long magnetoelastic simulation but doesn't affect internal dynamics
or the physical final state, since a uniform displacement carries no strain).

Usage:
    python ovf_subtract_mean.py input.ovf [output.ovf]

If output.ovf is not given, defaults to <input>_corrected.ovf

What it does:
    - Locates the "# Begin: Data Text" / "# End: Data Text" block, same as
      ovf_average.py.
    - Treats any triple that is exactly (0, 0, 0) as "outside geometry" and
      leaves those cells untouched (still exactly zero) in the output --
      same masking heuristic and same caveat as ovf_average.py (legitimate
      physical cells that are exactly zero in all 3 components would be
      incorrectly skipped, but this is very unlikely for a displacement or
      force field on a non-trivial state).
    - Computes the mean of the raw (x, y, z) components over the in-geometry
      cells and subtracts that mean vector from every in-geometry cell.
    - Writes a new OVF file that is byte-identical to the input EXCEPT for
      the data block: the header, footer, and any metadata (segment count,
      meshtype, bounds, valuedim, etc.) are copied through unmodified. This
      avoids having to reconstruct the full OVF header spec by hand, and
      guarantees the file stays valid regardless of which header fields
      your version of mumaxplus writes.
    - Prints the mean vector that was subtracted, for logging / sanity
      checking (e.g. comparing against effective_body_force's net mean).

Note: this script only handles OVF 2.0 "Data Text" segments (not binary
Data Binary 4/8 segments). If your files use binary format, let me know and
I can adapt this.
"""

import sys
import os
import re


def parse_ovf(path):
    # newline='' disables Python's universal-newline translation, so
    # whatever line endings the file already has (mumax typically writes
    # plain \n) are preserved exactly. mumax3's OVF header reader does a
    # strict string match on the first line and does not tolerate a
    # trailing \r, so round-tripping through default text mode (which can
    # rewrite \n -> \r\n on Windows) breaks loading.
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


def subtract_mean(path, out_path=None, fmt="{: .8e}"):
    text, header, start, end, triples = parse_ovf(path)

    total_cells = len(triples)
    included_idx = [
        i for i, t in enumerate(triples)
        if not (t[0] == 0.0 and t[1] == 0.0 and t[2] == 0.0)
    ]
    n = len(included_idx)

    if n == 0:
        raise ValueError(
            f"{path}: no in-geometry (nonzero) cells found -- cannot "
            "compute a mean to subtract."
        )

    sums = [0.0, 0.0, 0.0]
    for i in included_idx:
        t = triples[i]
        for k in range(3):
            sums[k] += t[k]
    mean = [s / n for s in sums]

    print(f"=== {path} ===")
    print(
        f"Grid: {header.get('xnodes', '?')} x {header.get('ynodes', '?')} x "
        f"{header.get('znodes', '?')}  ({total_cells} total cells, "
        f"{n} in-geometry cells)"
    )
    print(
        f"Mean vector subtracted from all in-geometry cells: "
        f"x={mean[0]: .6e}  y={mean[1]: .6e}  z={mean[2]: .6e}"
    )
    mag = (mean[0] ** 2 + mean[1] ** 2 + mean[2] ** 2) ** 0.5
    print(f"|mean vector| = {mag:.6e}")

    corrected = list(triples)
    for i in included_idx:
        t = triples[i]
        corrected[i] = (t[0] - mean[0], t[1] - mean[1], t[2] - mean[2])

    new_data_lines = [format_triple(t, fmt) for t in corrected]
    new_data_block = "\n" + "\n".join(new_data_lines) + "\n"

    new_text = text[:start] + new_data_block + text[end:]

    if out_path is None:
        base, ext = os.path.splitext(path)
        out_path = f"{base}_corrected{ext}"

    with open(out_path, "w", newline="") as f:
        f.write(new_text)

    print(f"Wrote drift-corrected file: {out_path}\n")
    return out_path, mean


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python ovf_subtract_mean.py input.ovf [output.ovf]")
        sys.exit(1)

    in_path = sys.argv[1]
    out_path = sys.argv[2] if len(sys.argv) > 2 else None
    subtract_mean(in_path, out_path)