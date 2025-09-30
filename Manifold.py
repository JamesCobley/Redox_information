# === Oxi-Shape: Young vs Old from original CSV (grid surface, viridis, 0–1) ===
# Bulletproof features:
# - Enforces biologically invariant IDs (Protein + Residue); no row-index fallback
# - Uses only sites measured in BOTH conditions
# - Converts % to 0–1 and clips to [0,1]
# - Builds a deterministic near-square grid (invariant x,y) by sorted Site_ID
# - Masks NaNs in the incomplete last grid row (no fake zeros)
# - Fixed color/z scale [0,1]; saves figures + grid mapping CSV

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ---------- Inputs ----------
CSV_PATH = "/content/site_all (3).csv"      # <-- adjust path
Y_COL    = "oxi_percent_brain.young"        # Young oxidation column
O_COL    = "oxi_percent_brain.old"          # Old oxidation column
PROT_CANDS = [
    "protein_id","Protein","ProteinID","Protein.Group","ProteinGroup",
    "Majority.protein.IDs","Leading.razor.protein","Accession","Uniprot","UniprotID"
]
RESI_CANDS = ["residue_index","ResidueIndex","Position","position","AApos","AminoAcidPosition","site","Site"]
ID_COL  = "Site_ID"                          # we will create/require this
OUT_STEM = "/content/oxi_shape_brain"        # output file stem
# -----------------------------

# --- Load ---
df = pd.read_csv(CSV_PATH, low_memory=False)

# --- Patch #1: Enforce biologically invariant Site_ID (Protein + Residue) ---
if ID_COL not in df.columns:
    prot_col = next((c for c in PROT_CANDS if c in df.columns), None)
    resi_col = next((c for c in RESI_CANDS if c in df.columns), None)
    if not (prot_col and resi_col):
        raise ValueError(
            "Cannot build invariant Site_ID: need a protein and residue column.\n"
            f"Checked proteins: {PROT_CANDS}\nChecked residues: {RESI_CANDS}\n"
            "Please provide Site_ID or add appropriate columns."
        )
    df[ID_COL] = df[prot_col].astype(str) + "_C" + df[resi_col].astype(str)

# --- Keep ONLY sites measured in BOTH conditions ---
for col in (Y_COL, O_COL):
    if col not in df.columns:
        raise ValueError(f"Required column '{col}' not found in CSV.")
mask = df[Y_COL].notna() & df[O_COL].notna()
df = df.loc[mask, [ID_COL, Y_COL, O_COL]].copy().reset_index(drop=True)
n_shared = len(df)
print(f"Sites measured in BOTH (Young & Old): {n_shared}")

# --- Convert to 0–1 (if given as %) and clip to [0,1] ---
def to_unit(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    # if values look like percentages, scale down
    if s.dropna().size and s.dropna().max() > 1.5:
        s = s / 100.0
    return s.clip(0, 1)

df["Young"] = to_unit(df[Y_COL])
df["Old"]   = to_unit(df[O_COL])

# --- Deterministic 2D manifold (x,y): near-square grid by sorted Site_ID ---
df = df.sort_values(ID_COL, kind="mergesort").reset_index(drop=True)  # stable sort
n = len(df)
W = int(np.ceil(np.sqrt(n)))    # grid width
H = int(np.ceil(n / W))         # grid height
df["idx"] = np.arange(n)
df["row"] = df["idx"] // W
df["col"] = df["idx"] % W

# --- Pack the two surfaces (Young, Old) on the same (x,y) ---
ZY = np.full((H, W), np.nan)    # Young surface (0–1)
ZO = np.full((H, W), np.nan)    # Old   surface (0–1)
for r, c, yv, ov in zip(df["row"], df["col"], df["Young"], df["Old"]):
    ZY[r, c] = yv
    ZO[r, c] = ov

# --- Plot helper (viridis, fixed [0,1], masked NaNs) ---
def plot_surface(Z, title, outfile):
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    xx, yy = np.meshgrid(np.arange(W), np.arange(H))
    Zm = np.ma.masked_invalid(Z)   # <-- Patch #2: mask NaNs, don't fill with zeros

    fig = plt.figure(figsize=(10, 7.5))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(xx, yy, Zm,
                           rstride=1, cstride=1,
                           linewidth=0, antialiased=True,
                           cmap="viridis", vmin=0, vmax=1)
    cb = fig.colorbar(surf, ax=ax, shrink=0.75, pad=0.1); cb.set_label("Oxidation (0–1)")
    ax.set_title(title)
    ax.set_xlabel("x (manifold)")
    ax.set_ylabel("y (manifold)")
    ax.set_zlabel("Oxidation (0–1)")
    ax.set_zlim(0, 1)
    plt.tight_layout()
    plt.savefig(outfile, dpi=300)
    plt.show()
    print(f"Saved: {outfile}")

# --- Render Oxi-Shapes ---
plot_surface(ZY, "Manifold Surface: YOUNG (brain)", f"{OUT_STEM}_young_surface.png")
plot_surface(ZO, "Manifold Surface: OLD (brain)",   f"{OUT_STEM}_old_surface.png")

# Optional Δ surface (Old − Young), still in 0–1 units (can be negative/positive)
ZD = ZO - ZY
xx, yy = np.meshgrid(np.arange(W), np.arange(H))
ZDm = np.ma.masked_invalid(ZD)
fig = plt.figure(figsize=(10, 7.5))
ax = fig.add_subplot(111, projection="3d")
surf = ax.plot_surface(xx, yy, ZDm,
                       rstride=1, cstride=1, linewidth=0, antialiased=True,
                       cmap="viridis", vmin=float(np.nanmin(ZD)), vmax=float(np.nanmax(ZD)))
cb = fig.colorbar(surf, ax=ax, shrink=0.75, pad=0.1); cb.set_label("Δ Oxidation (Old − Young)")
ax.set_title("Manifold Surface: Δ (Old − Young)")
ax.set_xlabel("x (manifold)"); ax.set_ylabel("y (manifold)"); ax.set_zlabel("Δ Oxidation")
plt.tight_layout()
plt.savefig(f"{OUT_STEM}_delta_surface.png", dpi=300)
plt.show()
print(f"Saved: {OUT_STEM}_delta_surface.png")

# --- Save the mapping for reproducibility ---
df[[ID_COL, "row", "col", "Young", "Old"]].to_csv(f"{OUT_STEM}_points_grid.csv", index=False)
print(f"Saved: {OUT_STEM}_points_grid.csv")
