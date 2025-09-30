# === Oxi-Shape + algebraic global state (AUC) and direction-aware change (OxUp/OxDown) ===
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ---------- Inputs ----------
CSV_PATH = "/content/site_all (3).csv"      # <-- adjust path if needed
Y_COL    = "oxi_percent_brain.young"
O_COL    = "oxi_percent_brain.old"
PROT_CANDS = [
    "protein_id","Protein","ProteinID","Protein.Group","ProteinGroup",
    "Majority.protein.IDs","Leading.razor.protein","Accession","Uniprot","UniprotID"
]
RESI_CANDS = ["residue_index","ResidueIndex","Position","position","AApos","AminoAcidPosition","site","Site"]
ID_COL   = "Site_ID"                        # will create if missing
OUT_STEM = "/content/oxi_shape_brain"
# --------------------------------

# --- Load ---
df = pd.read_csv(CSV_PATH, low_memory=False)

# --- Build invariant Site_ID from protein+residue (no row-index fallback) ---
if ID_COL not in df.columns:
    prot_col = next((c for c in PROT_CANDS if c in df.columns), None)
    resi_col = next((c for c in RESI_CANDS if c in df.columns), None)
    if not (prot_col and resi_col):
        raise ValueError("Need protein and residue columns (or precomputed Site_ID).")
    df[ID_COL] = df[prot_col].astype(str) + "_C" + df[resi_col].astype(str)

# --- Keep ONLY sites measured in BOTH Young & Old ---
for col in (Y_COL, O_COL):
    if col not in df.columns:
        raise ValueError(f"Required column '{col}' not found.")
mask = df[Y_COL].notna() & df[O_COL].notna()
df = df.loc[mask, [ID_COL, Y_COL, O_COL]].copy().reset_index(drop=True)
N = len(df)
print(f"Sites measured in BOTH (Young & Old): {N}")

# --- Convert to 0–1 and clip ---
def to_unit(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    if s.dropna().size and s.dropna().max() > 1.5:  # looks like %
        s = s / 100.0
    return s.clip(0, 1)

df["Young"] = to_unit(df[Y_COL])
df["Old"]   = to_unit(df[O_COL])

# --- Deterministic near-square grid (invariant x,y by sorted Site_ID) ---
df = df.sort_values(ID_COL, kind="mergesort").reset_index(drop=True)
W = int(np.ceil(np.sqrt(N)))
H = int(np.ceil(N / W))
df["idx"] = np.arange(N)
df["row"] = df["idx"] // W
df["col"] = df["idx"] % W

# Pack surfaces
ZY = np.full((H, W), np.nan)   # Young
ZO = np.full((H, W), np.nan)   # Old
for r, c, yv, ov in zip(df["row"], df["col"], df["Young"], df["Old"]):
    ZY[r, c] = yv
    ZO[r, c] = ov
ZD = ZO - ZY                    # Delta (Old - Young)

# --- Algebraic global state + direction-aware change ---
AUC_Y = float(np.nansum(ZY))
AUC_O = float(np.nansum(ZO))
dAUC  = AUC_O - AUC_Y

# Sitewise components from vectors (equivalent to surfaces on shared grid)
y = df["Young"].to_numpy()
o = df["Old"].to_numpy()
ox_up   = np.maximum(0.0, o - y)
ox_down = np.maximum(0.0, y - o)

OxUp    = float(ox_up.sum())
OxDown  = float(ox_down.sum())
f_up    = float((o > y).mean())
f_down  = float((y > o).mean())

print("\n=== Global state (AUC; 0–1 units) ===")
print(f"AUC(Young) = {AUC_Y:.3f}")
print(f"AUC(Old)   = {AUC_O:.3f}")
print(f"ΔAUC (Old−Young) = {dAUC:.3f}")

print("\n=== Direction-aware change (tropical semantics) ===")
print(f"OxUp   = Σ max(0, Old−Young)   = {OxUp:.3f}")
print(f"OxDown = Σ max(0, Young−Old)   = {OxDown:.3f}")
print(f"Fraction of sites ↑ = {f_up:.3f}, ↓ = {f_down:.3f}")

# --- Plot helpers ---
def plot_surface(Z, title, outfile, zlim=(0,1)):
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    xx, yy = np.meshgrid(np.arange(W), np.arange(H))
    Zm = np.ma.masked_invalid(Z)
    fig = plt.figure(figsize=(10, 7.5))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(xx, yy, Zm, rstride=1, cstride=1,
                           linewidth=0, antialiased=True,
                           cmap="viridis", vmin=zlim[0], vmax=zlim[1])
    cb = fig.colorbar(surf, ax=ax, shrink=0.75, pad=0.1)
    cb.set_label("Oxidation (0–1)" if zlim==(0,1) else "Δ Oxidation")
    ax.set_title(title)
    ax.set_xlabel("x (manifold)"); ax.set_ylabel("y (manifold)")
    ax.set_zlabel("Oxidation (0–1)" if zlim==(0,1) else "Δ Oxidation")
    if zlim is not None: ax.set_zlim(*zlim)
    plt.tight_layout(); plt.savefig(outfile, dpi=300); plt.show()
    print(f"Saved: {outfile}")

def bar_compare(values, labels, title, ylabel, outfile):
    fig, ax = plt.subplots(figsize=(6,4))
    x = np.arange(len(values))
    ax.bar(x, values)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=0)
    ax.set_title(title); ax.set_ylabel(ylabel)
    plt.tight_layout(); plt.savefig(outfile, dpi=300); plt.show()
    print(f"Saved: {outfile}")

# --- Render surfaces ---
plot_surface(ZY, "Oxi-Shape Surface: YOUNG (brain)", f"{OUT_STEM}_young_surface.png", zlim=(0,1))
plot_surface(ZO, "Oxi-Shape Surface: OLD (brain)",   f"{OUT_STEM}_old_surface.png",   zlim=(0,1))

# Optional Δ surface (can be negative/positive)
zmin, zmax = float(np.nanmin(ZD)), float(np.nanmax(ZD))
plot_surface(ZD, "Δ Surface: OLD − YOUNG (brain)",   f"{OUT_STEM}_delta_surface.png", zlim=(zmin, zmax))

# --- Compact algebra visuals ---
bar_compare([AUC_Y, AUC_O], ["Young", "Old"],
            "Global Redox State (AUC of Oxi-Shape)", "AUC (sum of 0–1 occupancies)",
            f"{OUT_STEM}_auc_bars.png")

bar_compare([OxUp, OxDown], ["OxUp", "OxDown"],
            "Direction-aware Global Change", "Sum of sitewise change (0–1 units)",
            f"{OUT_STEM}_oxup_oxdown_bars.png")

# Save mapping used
df[[ID_COL,"row","col","Young","Old"]].to_csv(f"{OUT_STEM}_points_grid.csv", index=False)
print(f"\nSaved mapping: {OUT_STEM}_points_grid.csv")
