import numpy as np
import matplotlib.pyplot as plt

# Bin centers (midpoints)
bin_centers = np.array([0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95])

# Synthetic redox probabilities
control = np.array([0.156666667, 0.326666667, 0.386666667, 0.103333333, 0.023333333,
                    0.003333333, 0, 0, 0, 0])
case = np.array([0.023333333, 0.116666667, 0.37, 0.35, 0.116666667,
                 0.02, 0.003333333, 0, 0, 0])

# Fisher information approximation (symmetric KL divergence)
mask = (control + case) > 0  # Avoid divide-by-zero
fisher_info = ((control[mask] - case[mask]) ** 2) / ((control[mask] + case[mask]) / 2)
bin_centers_masked = bin_centers[mask]

# Plot
plt.figure(figsize=(6, 4))
plt.plot(bin_centers_masked, fisher_info, marker='o', color='blue', linewidth=2)
plt.title('Fisher Information Across Redox Bins')
plt.xlabel('Oxidation Bin Center')
plt.ylabel('Fisher Information')
plt.grid(True)
plt.tight_layout()
plt.savefig('fisher_info.png', dpi=300)
plt.show()
