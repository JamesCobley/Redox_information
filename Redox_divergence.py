# === Redox Divergence (site-wise) on OxiMouse: brain young vs old ===
# Outputs:
#   - CSV with Site_ID, Young, Old, delta, psi_hard, psi_soft, microstates
#   - Scatter plot: delta (x) vs symmetry (y)
# Assumes the file `site_all (3).csv` contains columns:
#   "oxi_percent_brain.young" and "oxi_percent_brain.old"
# If Site_ID is missing, builds it from common protein/residue columns or falls back to row index.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------
# Params
# -----------------------
CSV_PATH = "/content/site_all (3).csv"
Y_COL    = "oxi_percent_brain.young"
O_COL    = "oxi_percent_brain.old"
EPS      = 0.001            # resolution ε (0.1% steps). Change if needed.
ROUNDING = "round"          # 'round' | 'floor' | 'ceil'
ID_COL   = "Site_ID"
OUT_CSV  = "/content/oximouse_brain_redox_divergence.csv"
OUT_PNG  = "/content/oximouse_brain_redox_divergence_scatter.png"

# Candidate columns to build a stable Site_ID if not present
PROT_CANDS = [
    "protein_id","Protein","ProteinID","Protein.Group","ProteinGroup",
    "Majority.protein.IDs","Leading.razor.protein","Accession","Uniprot","UniprotID"
]
RESI_CANDS = ["residue_index","ResidueIndex","Position","position","AApos","AminoAcidPosition","site","Site"]

# -----------------------
# Helpers
# -----------------------
def to_unit(x: pd.Series) -> pd.Series:
    x = pd.to_numeric(x, errors="coerce")
    if x.dropna().size and x.dropna().max() > 1.5:
        x = x / 100.0
    return x.clip(0, 1)

def quantize(vals: np.ndarray, eps: float, mode: str = "round") -> np.ndarray:
    m = int(np.floor(1.0/eps)) + 1  # microstates {0, eps, ..., 1}
    x = np.clip(vals, 0.0, 1.0) / eps
    if mode == "round":
        q = np.rint(x)
    elif mode == "floor":
        q = np.floor(x)
    elif mode == "ceil":
        q = np.ceil(x)
    else:
        raise ValueError("ROUNDING must be 'round' | 'floor' | 'ceil'")
    q = q.astype(int)
    q[q < 0]     = 0
    q[q > m - 1] = m - 1
    return q

# -----------------------
# Load and prepare
# -----------------------
df = pd.read_csv(CSV_PATH, low_memory=False)

if ID_COL not in df.columns:
    prot_col = next((c for c in PROT_CANDS if c in df.columns), None)
    resi_col = next((c for c in RESI_CANDS if c in df.columns), None)
    if prot_col and resi_col:
        df[ID_COL] = df[prot_col].astype(str) + "_C" + df[resi_col].astype(str)
    else:
        # Stable fallback: deterministic index
        df[ID_COL] = df.index.map(lambda i: f"row_{i}")

# Filter to sites quantified in BOTH conditions
if not ({Y_COL, O_COL} <= set(df.columns)):
    raise ValueError(f"Required columns not found: {Y_COL}, {O_COL}")

y = to_unit(df[Y_COL])
o = to_unit(df[O_COL])
mask = y.notna() & o.notna()
sub = df.loc[mask, [ID_COL, Y_COL, O_COL]].copy()
sub["Young"] = to_unit(sub[Y_COL])
sub["Old"]   = to_unit(sub[O_COL])
sub.drop(columns=[Y_COL, O_COL], inplace=True)

N = len(sub)
print(f"Sites with measurements in BOTH (Young & Old): {N}")

# -----------------------
# Per-site divergence and symmetry
# -----------------------
valsY = sub["Young"].to_numpy()
valsO = sub["Old"].to_numpy()

# Signed divergence (directional shift): delta_i = Young - Old
delta = valsY - valsO

# Microstate indices at resolution ε
qY = quantize(valsY, EPS, ROUNDING)
qO = quantize(valsO, EPS, ROUNDING)

# Symmetry indicator (hard): 1 if same microstate, else 0
psi_hard = (qY == qO).astype(float)

# Symmetry (soft fractional): 1 at equality, linearly decreasing to 0 at |Δ|=ε, 0 beyond
abs_diff = np.abs(valsY - valsO)
psi_soft = 1.0 - np.minimum(1.0, abs_diff / EPS)   # in [0,1]

# Assemble results
res = pd.DataFrame({
    "Site_ID": sub[ID_COL].values,
    "Young": valsY,
    "Old": valsO,
    "delta": delta,            # signed divergence (Young - Old)
    "psi_hard": psi_hard,      # 0/1 microstate match
    "psi_soft": psi_soft,      # fractional symmetry within ε
    "qY": qY,
    "qO": qO
})

# -----------------------
# Summary stats
# -----------------------
n_same_state = int(psi_hard.sum())
n_within_eps = int((abs_diff < EPS).sum())

print("\n=== Redox Divergence summary ===")
print(f"ε = {EPS:.3f} (microstates m = {int(np.floor(1.0/EPS))+1})")
print(f"Symmetric sites (same microstate): {n_same_state} / {N} ({100*n_same_state/N:.1f}%)")
print(f"Within ε (|Young-Old| < ε): {n_within_eps} / {N} ({100*n_within_eps/N:.1f}%)")
print(f"Median |delta| = {np.median(np.abs(delta)):.4f}, mean |delta| = {np.mean(np.abs(delta)):.4f}")

# -----------------------
# Scatter: delta (x) vs symmetry (y)
# -----------------------
fig, ax = plt.subplots(figsize=(8,5))
ax.scatter(delta, psi_soft, s=8, alpha=0.35)
ax.set_xlim(-1, 1)
ax.set_ylim(0, 1.02)
ax.set_xlabel("Signed divergence δ = Young − Old (unit interval)")
ax.set_ylabel("Symmetry ψ (soft, 1 at equality, →0 by ε)")
ax.set_title("Redox Divergence: site-wise map (OxiMouse brain)")
# helpful lines
ax.axvline(0, color='k', lw=0.7, alpha=0.6)
ax.axhline(1.0, color='k', lw=0.4, alpha=0.4)
ax.axhline(0.0, color='k', lw=0.4, alpha=0.4)
plt.tight_layout()
plt.savefig(OUT_PNG, dpi=300)
plt.show()
print(f"Saved scatter: {OUT_PNG}")

# Save results
res.to_csv(OUT_CSV, index=False)
print(f"Saved table: {OUT_CSV}")

# Optional: show top diverging sites for quick inspection
k = 10
top_pos = res.nlargest(k, "delta")[["Site_ID","Young","Old","delta","psi_hard","psi_soft"]]
top_neg = res.nsmallest(k, "delta")[["Site_ID","Young","Old","delta","psi_hard","psi_soft"]]
print("\nTop +Δ (more oxidized in Young):")
print(top_pos.to_string(index=False))
print("\nTop −Δ (more oxidized in Old):")
print(top_neg.to_string(index=False))
