import numpy as np
import pandas as pd
from scipy.stats import entropy
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# === Step 1: Load data ===
df = pd.read_csv("site_all (3).csv", low_memory=False)

# Pull out the two matched groups:
x = df["oxi_percent_brain.young"].dropna().values
y = df["oxi_percent_brain.old"].dropna().values

# === Step 2: Define the FIM between two histograms ===
def compute_fim(x, y, bins=21):
    # histogram over [0,100]%
    hist_x, edges = np.histogram(x * 100, bins=bins, range=(0,100), density=True)
    hist_y, _     = np.histogram(y * 100, bins=bins, range=(0,100), density=True)
    # normalize to probabilities
    P = hist_x / hist_x.sum()
    Q = hist_y / hist_y.sum()
    # Fisher Information Metric (Hellinger‐based)
    fim = np.sum((np.sqrt(P) - np.sqrt(Q))**2)
    return fim, edges[:-1]  # return left‐bin edges as θ

# === Step 3: Compute FIM curve ===
# Compute FIM for sliding windows of widths, or simply for the full distribution
fim_val, thetas = compute_fim(x, y, bins=100)

print(f"Global Fisher Information Metric (Young vs Old): {fim_val:.4f}")

# If you want a *surface* over θ, you can compute the local contribution per‐bin:
#   local_fim[i] = (sqrt(P[i]) - sqrt(Q[i]))**2
hist_x, edges = np.histogram(x * 100, bins=100, range=(0,100), density=True)
hist_y, _     = np.histogram(y * 100, bins=100, range=(0,100), density=True)
P = hist_x / hist_x.sum()
Q = hist_y / hist_y.sum()
local_fim = (np.sqrt(P) - np.sqrt(Q))**2
thetas = edges[:-1]

# === Step 4: Plot as 3D surface over "θ" and condition axis ===
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection="3d")

# Create a fake "condition" axis with two traces: one for Young vs Old
cond = np.array([0,1])  # 0=Young→Young (zero), 1=Young→Old (our FIM)
Theta, Cond = np.meshgrid(thetas, cond)

# Build Z: zero line for cond=0; local_fim for cond=1
Z = np.zeros_like(Theta)
Z[1,:] = local_fim  # only second row carries information

ax.plot_surface(Theta, Cond, Z,
                rstride=2, cstride=2, cmap="viridis", edgecolor="k", alpha=0.8)

# Formatting
ax.set_xlabel("θ (% Oxidation)")
ax.set_ylabel("Comparison")
ax.set_yticks([0,1])
ax.set_yticklabels(["Young vs Young","Young vs Old"])
ax.set_zlabel("Fisher Information")
ax.set_title("Fisher Information Surface over Redox θ")

plt.tight_layout()
plt.savefig("fim_surface.png", dpi=300)
plt.show()
