import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy import arccos, clip
from matplotlib import cm

# --- Params ---
bins = np.linspace(0, 100, 21)  # 21 edges → 20 bins over 0–100%
np.random.seed(0)

# --- Load & filter brain data ---
df = pd.read_csv("site_all (3).csv", low_memory=False)
y = df["oxi_percent_brain.young"].dropna().values * 100
o = df["oxi_percent_brain.old"].dropna().values   * 100

# --- Compute histogram‐based P & Q ---
P, _ = np.histogram(y, bins=bins, density=True)
Q, _ = np.histogram(o, bins=bins, density=True)
P = P / P.sum()
Q = Q / Q.sum()

# --- 1) Global Fisher–Rao distance ---
BC   = np.sum(np.sqrt(P * Q))        # Bhattacharyya coefficient
BC   = clip(BC, 0, 1)                # numerical safety
dFR  = 2 * arccos(BC)
print(f"Global Fisher–Rao distance (Young‖Old): {dFR:.4f} (radians)")

# --- 2) Local per‐bin contributions δ_i ---
# δ_i = 2 arccos( sqrt(P_i * Q_i) )
delta = 2 * arccos(clip(np.sqrt(P*Q), 0, 1))

# --- 3) Plot heatmap of δ_i over bins ---
fig, ax = plt.subplots(figsize=(8,3))
# we make a 2×N array so pcolormesh shows a single row of heat
grid = np.vstack([delta, delta])

# ... [earlier code up through computing `delta`]

# Build 2×20 grid of local δ_i
grid = np.vstack([delta, delta])  # shape (2, 20)

# Define edges
y_edges = [0, 1, 2]  # M+1 = 3

# Plot
fig, ax = plt.subplots(figsize=(8, 3))
c = ax.pcolormesh(
    bins,        # length 21 = N+1
    y_edges,     # length 3  = M+1
    grid,        # shape (2, 20) = (M, N)
    cmap="viridis",
    edgecolors="k"
)
ax.set_yticks([0.5, 1.5])
ax.set_yticklabels(["Young vs Young", "Young vs Old"])
ax.set_xlabel("Mean Oxidation (%)")
ax.set_title("Local Fisher–Rao Contributions over Oxidation Bins")
fig.colorbar(c, ax=ax, label=r"$\delta_i = 2\arccos\sqrt{P_i Q_i}$")
plt.tight_layout()
plt.savefig("delta_heatmap.png", dpi=300)
plt.show()
