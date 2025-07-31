import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import entropy
from matplotlib import cm

# --- Params ---
n_reps = 5
bins   = np.linspace(0, 1, 21)
np.random.seed(42)

# --- Load & filter brain data, keep site identifiers ---
df = pd.read_csv("site_all (3).csv", low_memory=False)
sub = df[["site",
          "oxi_percent_brain.young","se_brain.young",
          "oxi_percent_brain.old",  "se_brain.old"]].dropna()

sites    = sub["site"].values
means_y  = sub["oxi_percent_brain.young"].values
sems_y   = sub["se_brain.young"].values
means_o  = sub["oxi_percent_brain.old"].values
sems_o   = sub["se_brain.old"].values
n_sites  = len(sites)
print(f"Using {n_sites} brain sites with full data.\n")

# --- Helper: simulate per-site replicates ---
def simulate_reps(mu, sem):
    sd  = sem * np.sqrt(n_reps)
    sims = np.random.normal(loc=mu[:,None], scale=sd[:,None])
    return np.clip(sims, 0, 1)

# --- Compute per-site KL divergences once ---
kl_vals = []
for i in range(n_sites):
    sims_y = simulate_reps(means_y[[i]], sems_y[[i]]).ravel()
    sims_o = simulate_reps(means_o[[i]], sems_o[[i]]).ravel()
    p = np.histogram(sims_y, bins=bins, density=True)[0] + 1e-9
    q = np.histogram(sims_o, bins=bins, density=True)[0] + 1e-9
    p /= p.sum(); q /= q.sum()
    kl_vals.append(entropy(p, q, base=2))

kl_vals = np.array(kl_vals)

# --- Compute and print mean KL ---
mean_kl = kl_vals.mean()
print(f"Mean per-site KL divergence: {mean_kl:.4f} bits\n")

# --- Count divergent sites ---
eps = 1e-6
n_diverge = np.sum(kl_vals > eps)
print(f"Number of sites with KL > 0: {n_diverge} / {n_sites}\n")

# --- Print top 10 by KL ---
top_idx = np.argsort(kl_vals)[-10:][::-1]
print("Top 10 sites by per-site KL divergence:")
for rank, i in enumerate(top_idx, 1):
    print(f"{rank:2d}. {sites[i]:25s}  KL = {kl_vals[i]:.4f} bits")

# --- Plot histogram of the full distribution with mean line ---
plt.figure(figsize=(8,5))
plt.hist(kl_vals, bins=30, color=cm.viridis(0.6), edgecolor="black")
plt.axvline(mean_kl, color="gray", linestyle="--", linewidth=2,
            label=f"Mean KL = {mean_kl:.4f} bits")
plt.xlabel("Per-Site KL Divergence (Young || Old)")
plt.ylabel("Site Count")
plt.title("Distribution of Per-Site KL Divergences")
plt.legend()
plt.tight_layout()
plt.savefig("kl_hist_all.png", dpi=300)
plt.show()
