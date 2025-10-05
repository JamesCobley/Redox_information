# --- build distributions & indices ---
pY, cY, qY = normalized_hist(y, EPS, ROUNDING)
pO, cO, qO = normalized_hist(o, EPS, ROUNDING)
M = len(pY)
x_pct = np.linspace(0, 100, M)

# Local symmetry components
Psi_m = np.minimum(pY, pO)           # local overlap ψε(m)
A_m   = 0.5 * np.abs(pY - pO)        # local asymmetry contribution Aε(m)

print(f"Check: global Ψ = {Psi_m.sum():.3f}, local asym sum = {A_m.sum():.3f} (should equal 1-Ψ)")

# --- Plot local asymmetry landscape ---
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(8, 3.5))
ax.plot(x_pct, A_m, lw=1.2)
ax.fill_between(x_pct, 0, A_m, alpha=0.35)
ax.set_xlabel("% oxidation (microstate)")
ax.set_ylabel("Local asymmetry  A\u03B5(m)")
ax.set_title("Redox Symmetry Landscape: where manifolds differ")
plt.tight_layout()
plt.savefig("/content/symmetry_landscape.png", dpi=300)
plt.show()

# --- (optional) Plot local overlap (where manifolds agree) ---
fig, ax = plt.subplots(figsize=(8, 3.5))
ax.plot(x_pct, Psi_m, lw=1.2)
ax.fill_between(x_pct, 0, Psi_m, alpha=0.35)
ax.set_xlabel("% oxidation (microstate)")
ax.set_ylabel("Local overlap  \u03C8\u03B5(m)")
ax.set_title("Local Redox Symmetry: where manifolds agree")
plt.tight_layout()
plt.savefig("/content/symmetry_overlap.png", dpi=300)
plt.show()
