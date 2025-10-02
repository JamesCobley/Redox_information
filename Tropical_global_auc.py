# === Oxi-Shape + tropical algebraic summaries (AUC, means, OxUp/OxDown) ===
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ---------- Inputs ----------
CSV_PATH = "/content/site_all (3).csv"      # <-- adjust path if needed
Y_COL    = "oxi_percent_brain.young"
O_COL    = "oxi_percent_brain.old"

# Columns that might hold protein & residue identifiers (we'll pick what exists)
PROT_CANDS = [
    "protein_id","Protein","ProteinID","Protein.Group","ProteinGroup",
    "Majority.protein.IDs","Leading.razor.protein","Accession","Uniprot","UniprotID"
]
RESI_CANDS = ["residue_index","ResidueIndex","Position","position","AApos",
              "AminoAcidPosition","site","Site","Residue"]

ID_COL   = "Site_ID"                        # will create if missing
OUT_STEM = "/content/oxi_shape_brain"
TOL_NULL = 1e-6                             # tolerance for "no change" at a site
# --------------------------------

# --- Load ---
df = pd.read_csv(CSV_PATH, low_memory=False)

# --- Build invariant Site_ID from protein+residue ---
if ID_COL not in df.columns:
    prot_col = next((c for c in PROT_CANDS if c in df.columns), None)
    resi_col = next((c for c in RESI_CANDS if c in df.columns), None)
    if not (prot_col and resi_col):
        raise ValueError("Need protein and residue columns (or precomputed Site_ID). "
                         f"Tried protein={PROT_CANDS}, residue={RESI_CANDS}")
    df[ID_COL] = df[prot_col].astype(str) + "_C" + df[resi_col].astype(str)

# --- Keep ONLY sites measured in BOTH Young & Old ---
for col in (Y_COL, O_COL):
    if col not in df.columns:
        raise ValueError(f"Required column '{col}' not found in file.")
mask = df[Y_COL].notna() & df[O_COL].notna()
df = df.loc[mask, [ID_COL, Y_COL, O_COL]].copy()

# --- (Optional) if duplicate Site_ID rows exist, aggregate by mean ---
if df.duplicated(ID_COL).any():
    df = (df
          .groupby(ID_COL, as_index=False)
          .agg({Y_COL: "mean", O_COL: "mean"}))

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
N = len(df)
print(f"Sites measured in BOTH (Young & Old): {N}")
W = int(np.ceil(np.sqrt(N)))
H = int(np.ceil(N / W))
df["idx"] = np.arange(N)
df["row"] = df["idx"] // W
df["col"] = df["idx"] % W

# --- Pack surfaces (Young/Old) on the same grid ---
ZY = np.full((H, W), np.nan)
ZO = np.full((H, W), np.nan)
for r, c, yv, ov in zip(df["row"], df["col"], df["Young"], df["Old"]):
    ZY[r, c] = yv
    ZO[r, c] = ov
ZD = ZO - ZY  # Old - Young

# --- Global algebraic state: AUC (sums) + means/medians/IQR ---
AUC_Y = float(np.nansum(ZY))
AUC_O = float(np.nansum(ZO))
dAUC  = AUC_O - AUC_Y

mean_y = AUC_Y / N
mean_o = AUC_O / N
delta_mean = mean_o - mean_y

med_y = float(np.nanmedian(df["Young"]))
med_o = float(np.nanmedian(df["Old"]))
iqr_y = float(np.nanpercentile(df["Young"], 75) - np.nanpercentile(df["Young"], 25))
iqr_o = float(np.nanpercentile(df["Old"],   75) - np.nanpercentile(df["Old"],   25))

print("\n=== Global state (AUC; 0–1 units) ===")
print(f"AUC(Young) = {AUC_Y:.3f}")
print(f"AUC(Old)   = {AUC_O:.3f}")
print(f"ΔAUC (Old−Young) = {dAUC:.3f}")

print("\n=== Mean cysteine oxidation (unit interval; % in parentheses) ===")
print(f"Young mean = {mean_y:.4f} ({100*mean_y:.2f}%) | median={med_y:.4f}, IQR={iqr_y:.4f}")
print(f"Old   mean = {mean_o:.4f} ({100*mean_o:.2f}%) | median={med_o:.4f}, IQR={iqr_o:.4f}")
print(f"Δ mean (Old − Young) = {delta_mean:.4f} ({100*delta_mean:.2f}%)")

# --- Direction-aware tropical change (sitewise; OxUp/OxDown) ---
y = df["Young"].to_numpy()
o = df["Old"].to_numpy()
ox_up   = np.maximum(0.0, o - y)  # oxidative morphisms
ox_down = np.maximum(0.0, y - o)  # reductive morphisms
OxUp    = float(ox_up.sum())
OxDown  = float(ox_down.sum())
f_up    = float((o > y).mean())
f_down  = float((y > o).mean())

print("\n=== Direction-aware change (tropical semantics) ===")
print(f"OxUp   = Σ max(0, Old−Young)   = {OxUp:.3f}")
print(f"OxDown = Σ max(0, Young−Old)   = {OxDown:.3f}")
print(f"Fraction of sites ↑ = {f_up:.3f}, ↓ = {f_down:.3f}")

# --- Per-site morphism classification (Young → Old) ---
df["Morphism"] = "Null"
df["Morphism_Value"] = df["Young"]  # initialize

# Reduction
mask_reduct = (df["Young"] - df["Old"]) > TOL_NULL
df.loc[mask_reduct, "Morphism"] = "Reduction"
df.loc[mask_reduct, "Morphism_Value"] = df.loc[mask_reduct, ["Young","Old"]].min(axis=1)  # tropical ⊕

# Oxidation
mask_oxid = (df["Old"] - df["Young"]) > TOL_NULL
df.loc[mask_oxid, "Morphism"] = "Oxidation"
df.loc[mask_oxid, "Morphism_Value"] = df.loc[mask_oxid, ["Young","Old"]].max(axis=1)      # tropical ⊗

# Null (idempotent within tolerance)
mask_null = (~mask_reduct) & (~mask_oxid)
df.loc[mask_null, "Morphism"] = "Null"
df.loc[mask_null, "Morphism_Value"] = df.loc[mask_null, "Young"]

counts = df["Morphism"].value_counts()
print("\n=== Morphism counts ===")
print(counts.to_string())

# Save per-site morphisms
morph_out = f"{OUT_STEM}_morphisms.csv"
df[[ID_COL,"row","col","Young","Old","Morphism","Morphism_Value"]].to_csv(morph_out, index=False)
print(f"Saved per-site morphisms: {morph_out}")

# ---------- Plot helpers ----------
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

# ---------- Render surfaces ----------
plot_surface(ZY, "Oxi-Shape Surface: YOUNG (brain)",
             f"{OUT_STEM}_young_surface.png", zlim=(0,1))

plot_surface(ZO, "Oxi-Shape Surface: OLD (brain)",
             f"{OUT_STEM}_old_surface.png",   zlim=(0,1))

# Δ surface (may span negative to positive)
zmin, zmax = float(np.nanmin(ZD)), float(np.nanmax(ZD))
plot_surface(ZD, "Δ Surface: OLD − YOUNG (brain)",
             f"{OUT_STEM}_delta_surface.png", zlim=(zmin, zmax))

# ---------- Compact algebra visuals ----------
bar_compare([AUC_Y, AUC_O], ["Young", "Old"],
            "Global Redox State (AUC of Oxi-Shape)",
            "AUC (sum of 0–1 occupancies)",
            f"{OUT_STEM}_auc_bars.png")

bar_compare([mean_y, mean_o], ["Young", "Old"],
            "Mean cysteine oxidation",
            "Mean occupancy (0–1)",
            f"{OUT_STEM}_mean_bars.png")

bar_compare([OxUp, OxDown], ["OxUp", "OxDown"],
            "Direction-aware Global Change",
            "Sum of sitewise change (0–1 units)",
            f"{OUT_STEM}_oxup_oxdown_bars.png")

# Save mapping used for surfaces
grid_out = f"{OUT_STEM}_points_grid.csv"
df[[ID_COL,"row","col","Young","Old"]].to_csv(grid_out, index=False)
print(f"Saved grid mapping: {grid_out}")
