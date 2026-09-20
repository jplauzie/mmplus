import sys, numpy as np
a, b = np.load(sys.argv[1]), np.load(sys.argv[2])
n = min(len(a["t"]), len(b["t"]))
print(f"{'chunk':>5s} {'t(ns)':>7s} {'rel diff KE':>12s} {'rel diff Eel':>13s} {'rel diff umax':>14s} {'max|dm|':>9s}")
for k in np.linspace(0, n - 1, 10).astype(int):
    rd = lambda x: abs(a[x][k] - b[x][k]) / max(abs(a[x][k]), abs(b[x][k]), 1e-300)
    print(f"{k:5d} {a['t'][k]*1e9:7.3f} {rd('KE'):12.2e} {rd('Eel'):13.2e} {rd('umax'):14.2e} "
          f"{np.max(np.abs(a['mavg'][k]-b['mavg'][k])):9.1e}")
print("mean dt:", a["dt"][:n].mean(), b["dt"][:n].mean(), "  total wall(s):", a["wall"].sum(), b["wall"].sum())