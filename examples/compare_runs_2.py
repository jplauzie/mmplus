import sys, numpy as np
from scipy.interpolate import CubicSpline
a, b = np.load(sys.argv[1]), np.load(sys.argv[2])
t0, t1 = max(a["t"][0], b["t"][0]), min(a["t"][-1], b["t"][-1])
tc = np.linspace(t0, t1, 3000)
print(f"end times (ns): {a['t'][-1]*1e9:.4f} {b['t'][-1]*1e9:.4f}   window {t0*1e9:.3f}..{t1*1e9:.3f} ns")
for k in ("ur", "KE", "Eel"):
    x, y = CubicSpline(a["t"], a[k])(tc), CubicSpline(b["t"], b[k])(tc)
    print(f"{k:4s}: max|diff|/max|value| = {np.max(np.abs(x-y))/np.max(np.abs(x)):.2e}"
          f"   rms diff/rms value = {np.sqrt(np.mean((x-y)**2))/np.sqrt(np.mean(x**2)):.2e}")