import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.stats import entropy

# Seed and simulated data
np.random.seed(42)
n = 100
control_probs = np.random.dirichlet(np.ones(4), size=n)
case_probs = control_probs + np.random.normal(0, 0.02, control_probs.shape)
case_probs = np.clip(case_probs, 1e-9, None)
case_probs /= case_probs.sum(axis=1, keepdims=True)

# Per-peptide KL divergence: D_KL(P || Q)
kl_divs = [entropy(p, q, base=2) for p, q in zip(control_probs, case_probs)]

# Plotting
plt.figure(figsize=(8, 5))
viridis_color = cm.viridis(0.6)
plt.hist(kl_divs, bins=20, color=viridis_color, edgecolor='black')

# Add vertical mean line
mean_kl = np.mean(kl_divs)
plt.axvline(mean_kl, color='gray', linestyle='--', label=f'Mean KL = {mean_kl:.4f}')

# Labels and title
plt.xlabel("KL Divergence (bits)")
plt.ylabel("Peptide Count")
plt.title("Per-Peptide KL Divergence\n(Control (P) || Case (Q))")
plt.legend()

# Optional: log scale toggle
use_log = False
if use_log:
    plt.yscale("log")

plt.tight_layout()
plt.savefig("per_peptide_kl_viridis.png", dpi=300)
plt.show()
