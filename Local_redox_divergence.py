# --- Finite-safe pairwise mask BEFORE anything else ---
import numpy as np

y = np.asarray(y, float)
o = np.asarray(o, float)
pair_mask = np.isfinite(y) & np.isfinite(o)
y = y[pair_mask]
o = o[pair_mask]
print(f"Pairwise-finite sites: {y.size}")

# --- Robust helpers (finite-safe, zero-safe) ---
def quantize(vals, eps, mode="round"):
    M = int(np.floor(1.0/eps)) + 1          # microstates {0, eps, ..., 1}
    v = np.clip(np.asarray(vals, float), 0.0, 1.0) / eps
    if mode == "round": q = np.rint(v)
    elif mode == "floor": q = np.floor(v)
    elif mode == "ceil":  q = np.ceil(v)
    else: raise ValueError("ROUNDING must be 'round'|'floor'|'ceil'")
    q = np.clip(q.astype(int), 0, M-1)
    return q, M

def normalized_hist(vals, eps, mode="round"):
    """Finite-safe: drops NaN/Inf, zero-safe normalization."""
    M = int(np.floor(1.0/eps)) + 1
    v = np.asarray(vals, float)
    msk = np.isfinite(v)
    v = v[msk]
    if v.size == 0:
        return np.zeros(M, float), np.zeros(M, float), np.full(0, np.nan)
    q, _ = quantize(v, eps, mode)
    cnt = np.bincount(q, minlength=M).astype(float)
    total = cnt.sum()
    p = cnt / total if total > 0 else np.zeros_like(cnt)
    return p, cnt, q

# --- build distributions & indices ---
pY, cY, qY = normalized_hist(y, EPS, ROUNDING)
pO, cO, qO = normalized_hist(o, EPS, ROUNDING)
M = len(pY)
x_pct = np.linspace(0, 100, M)

# ===== A) Redox Symmetry Landscape =====
Psi_m = np.minimum(pY, pO)
A_m   = 0.5 * np.abs(pY - pO)
print(f"Check: global Ψ = {Psi_m.sum():.3f}, local asym sum = {A_m.sum():.3f} (should equal 1-Ψ)")

# Plot
import matplotlib.pyplot as plt
ax.plot(x_pct, A_m, lw=1.2)
ax.fill_between(x_pct, 0, A_m, alpha=0.4)
ax.set_xlabel("% oxidation (microstate)")
ax.set_ylabel("Local asymmetry Aε(m)")
ax.set_title("Redox Symmetry Landscape: where manifolds differ")
plt.tight_layout()
plt.savefig("/content/symmetry_landscape.png", dpi=300)
plt.show()


# ===== B) Redox Divergence Landscape =====
delta = y - o
SBF = np.full(M, np.nan); Dmag = np.full(M, np.nan); Bsgn = np.full(M, np.nan)
for m in range(M):
    sel = (qY == m) | (qO == m)
    if not np.any(sel): continue
    dsel = delta[sel]
    SBF[m] = np.mean((qY[sel] != qO[sel]).astype(float))
    Dmag[m] = np.median(np.abs(dsel))
    Bsgn[m] = np.median(dsel)

fig, axes = plt.subplots(3,1, figsize=(8,8), sharex=True)
axes[0].plot(x_pct, SBF); axes[0].set_ylabel("SBFε(m)")
axes[1].plot(x_pct, Dmag); axes[1].set_ylabel("Dε(m) = median |δ|")
axes[2].plot(x_pct, Bsgn); axes[2].axhline(0,color='k',lw=0.8,alpha=0.6)
axes[2].set_ylabel("Bε(m) = median δ"); axes[2].set_xlabel("% oxidation (microstate)")
plt.tight_layout(); plt.savefig("/content/divergence_landscape.png", dpi=300); plt.show()
print("Saved: /content/divergence_landscape.png")

# --- Top hotspots tables ---
import pandas as pd
k = 10
hot_sym = pd.DataFrame({"percent": x_pct, "A_m": A_m}).nlargest(k, "A_m")
hot_div = pd.DataFrame({"percent": x_pct, "SBF": SBF, "Dmag": Dmag, "Bsgn": Bsgn}) \
            .sort_values(["SBF","Dmag"], ascending=[False, False]).head(k)
print("\nTop symmetry-difference hotspots (by Aε(m)):")
print(hot_sym.to_string(index=False))
print("\nTop divergence hotspots (by SBF then |δ|):")
print(hot_div.to_string(index=False))
