import sys, numpy as np
h = np.load(sys.argv[1]); E = h["KE"] + h["Eel"] + h["Eme"]
scale = np.max(np.abs(h["Eme"]))
print(f"Etot/max|Eme|: min {E.min()/scale:+.3e}  max {E.max()/scale:+.3e}")
for k in np.linspace(0, len(E) - 1, 10).astype(int):
    print(k, f"t={h['t'][k]*1e9:.3f}ns KE={h['KE'][k]:.3e} Eel={h['Eel'][k]:.3e} "
             f"Eme={h['Eme'][k]:.3e} Etot/scale={E[k]/scale:+.3e}")