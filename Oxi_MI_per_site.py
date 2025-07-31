import numpy as np
import pandas as pd
from sklearn.metrics import mutual_info_score

# --- Params ---
bins = 10        # number of bins for MI discretization
np.random.seed(42)

# --- Load & filter for complete brain sites ---
df = pd.read_csv("site_all (3).csv", low_memory=False)

sub = df[["site", 
          "oxi_percent_brain.young", "oxi_percent_brain.old"]].dropna()
sub = sub.rename(columns={
    "oxi_percent_brain.young": "mean_y",
    "oxi_percent_brain.old":   "mean_o"
})

# --- Compute delta and pick top 100 ---
sub["delta_mean"] = (sub["mean_o"] - sub["mean_y"]).abs()
top100 = sub.nlargest(100, "delta_mean").reset_index(drop=True)

# --- Mutual information helper ---
def compute_mi(x, y, bins):
    x_bin = pd.cut(x, bins=bins, labels=False)
    return mutual_info_score(x_bin, y)

# --- Build arrays for MI ---
# values: concatenated [mean_y, mean_o]
# labels: 0 for Young, 1 for Old
values = np.concatenate([top100["mean_y"], top100["mean_o"]])
labels = np.concatenate([
    np.zeros(len(top100), dtype=int),
    np.ones(len(top100),  dtype=int)
])

# --- Compute MI on the top-100 set ---
mi = compute_mi(values, labels, bins=bins)
print(f"Mutual Information (Top-100 Δ sites): {mi:.4f} bits")

# --- Optionally, per-site MI (treating each site as a two-point distribution) ---
per_site = []
for _, row in top100.iterrows():
    x = np.array([row["mean_y"], row["mean_o"]])
    lab = np.array([0, 1])
    mi_site = compute_mi(x, lab, bins=2)  # 2 bins suffices for two values
    per_site.append(mi_site)

top100["MI_site"] = per_site
print("\nTop 10 sites by per-site MI:")
print(top100[["site","delta_mean","MI_site"]]
      .sort_values("MI_site", ascending=False)
      .head(10)
      .to_string(index=False))
