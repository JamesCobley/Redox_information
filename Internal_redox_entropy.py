# === Internal Redox Entropy on the brain Oxi-Mouse data ===
# Definition used:
#   microstates at resolution ε (default 1%),
#   coverage Cε = (# occupied microstates) / (total microstates),
#   evenness Eε = 1 - ||p - u||_1 / 2  (u = uniform over microstates),
#   entropy Sε = Cε * Eε.
# No bootstrapping, no Shannon bins.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------
# User params
# -----------------------
CSV_PATH = "/content/site_all (3).csv"
Y_COL    = "oxi_percent_brain.young"
O_COL    = "oxi_percent_brain.old"
EPS      = 0.001   # instrument/analysis resolution (1% steps). Change if needed (e.g., 0.005).
ROUNDING = "round"  # 'round' | 'floor' | 'ceil' depending on your instrument export

# -----------------------
# Helpers
# -----------------------
def to_unit(x: pd.Series) -> pd.Series:
    """Convert % to [0,1] if needed and clip to [0,1]."""
    x = pd.to_numeric(x, errors="coerce")
    if x.dropna().size and x.dropna().max() > 1.5:
        x = x / 100.0
    return x.clip(0, 1)

def quantize(vals: np.ndarray, eps: float, mode: str = "round") -> np.ndarray:
    """Map [0,1] values to microstate indices 0..m-1 using the assay resolution."""
    m = int(np.floor(1.0/eps)) + 1  # microstates at {0, eps, ..., 1}
    x = np.clip(vals, 0.0, 1.0) / eps
    if mode == "round":
        q = np.rint(x)
    elif mode == "floor":
        q = np.floor(x)
    elif mode == "ceil":
        q = np.ceil(x)
    else:
        raise ValueError("ROUNDING must be 'round', 'floor', or 'ceil'")
    q = q.astype(int)
    q[q < 0]     = 0
    q[q > m - 1] = m - 1
    return q

def coverage_evenness_entropy(vals: np.ndarray, eps: float, mode: str = "round"):
    """Compute (Cε, Eε, Sε) from raw [0,1] values."""
    m = int(np.floor(1.0/eps)) + 1
    q = quantize(vals, eps, mode)  # 0..m-1
    counts = np.bincount(q, minlength=m).astype(float)
    N = counts.sum()
    p = counts / N  # frequency vector over microstates
    # Coverage
    C = (counts > 0).sum() / m
    # Evenness via L1 distance to uniform u
    u = np.full(m, 1.0/m)
    E = 1.0 - (np.abs(p - u).sum())/2.0
    # Entropy (order–disorder index)
    S = C * E
    return C, E, S, counts

def hist_plot(counts, eps, title, outfile=None):
    m = counts.size
    xs = np.arange(m) * eps * 100.0  # convert to %
    fig, ax = plt.subplots(figsize=(8,3.5))
    ax.bar(xs, counts, width=max(0.6, (100.0/m)*0.8))
    ax.set_xlabel("% oxidation (microstate centers)")
    ax.set_ylabel("# cysteines")
    ax.set_title(title)
    plt.tight_layout()
    if outfile:
        plt.savefig(outfile, dpi=300)
    plt.show()

# -----------------------
# Load and filter
# -----------------------
df = pd.read_csv(CSV_PATH, low_memory=False)
if not ({Y_COL, O_COL} <= set(df.columns)):
    raise ValueError(f"Required columns not found: {Y_COL}, {O_COL}")

y = to_unit(df[Y_COL])
o = to_unit(df[O_COL])

mask = y.notna() & o.notna()
y = y[mask].to_numpy()
o = o[mask].to_numpy()
N = y.size
print(f"Sites with measurements in BOTH (Young & Old): {N}")

# -----------------------
# Descriptives (for context)
# -----------------------
print("\n=== Means (unit interval and %) ===")
print(f"Young mean = {y.mean():.4f} ({100*y.mean():.2f}%)")
print(f"Old   mean = {o.mean():.4f} ({100*o.mean():.2f}%)")
print(f"Δ mean (Old−Young) = {o.mean()-y.mean():.4f} ({100*(o.mean()-y.mean()):.2f}%)")

# -----------------------
# Internal redox entropy
# -----------------------
CY, EY, SY, cntY = coverage_evenness_entropy(y, EPS, ROUNDING)
CO, EO, SO, cntO = coverage_evenness_entropy(o, EPS, ROUNDING)

print("\n=== Internal Redox Entropy (resolution-aware; no binning) ===")
print(f"Resolution ε = {EPS:.3f} → microstates m = {cntY.size}")
print(f"Young:  Coverage Cε = {CY:.3f}, Evenness Eε = {EY:.3f}, Entropy Sε = {SY:.3f}")
print(f"Old:    Coverage Cε = {CO:.3f}, Evenness Eε = {EO:.3f}, Entropy Sε = {SO:.3f}")
print(f"ΔSε (Old−Young) = {SO - SY:.3f}")

# Optional “deviation energy” from perfect evenness (geometric complement)
def deviation_energy(counts):
    m = counts.size
    N = counts.sum()
    return (np.abs(counts - (N/m)).sum())/N

VE_Y = deviation_energy(cntY)
VE_O = deviation_energy(cntO)
print(f"\nDeviation from uniform occupancy (smaller = flatter):")
print(f"Young: {VE_Y:.3f} | Old: {VE_O:.3f} | Δ = {VE_O-VE_Y:.3f}")

# -----------------------
# Quick visuals (optional)
# -----------------------
hist_plot(cntY, EPS, "Young — occupancy across microstates", "/content/young_microstates.png")
hist_plot(cntO, EPS, "Old — occupancy across microstates",   "/content/old_microstates.png")
