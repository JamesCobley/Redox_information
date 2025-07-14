# --- Imports ---
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import cm
from matplotlib import colors as mcolors
from scipy.stats import entropy, ttest_ind

# --- Reproducibility ---
np.random.seed(42)

# --- Parameters ---
n_peptides = 100
n_replicates = 3
bins = np.linspace(0, 1, 11)  # 10 bins: 0–1 in 10% steps

# --- Synthetic Data Generator ---
def generate_redox_data(mean, std, n_peptides, n_replicates):
    return np.clip(np.random.normal(loc=mean, scale=std, size=(n_peptides, n_replicates)), 0, 1)

# Create synthetic data
control = generate_redox_data(0.2, 0.1, n_peptides, n_replicates)
case = generate_redox_data(0.5, 0.2, n_peptides, n_replicates)

# --- Compute Shannon Entropy ---
def compute_entropy(matrix, bins):
    entropies = []
    for peptide in matrix:
        hist, _ = np.histogram(peptide, bins=bins, density=True)
        ent = entropy(hist + 1e-10, base=2)
        entropies.append(ent)
    return np.array(entropies)

entropy_control = compute_entropy(control, bins)
entropy_case = compute_entropy(case, bins)

# --- DataFrame ---
df = pd.DataFrame({
    'Entropy': np.concatenate([entropy_control, entropy_case]),
    'Condition': ['Control'] * n_peptides + ['Case'] * n_peptides
})

# --- Color Palette ---
viridis = cm.get_cmap('viridis', 2)
colors = [mcolors.to_hex(viridis(i)) for i in range(2)]
palette = {'Control': colors[0], 'Case': colors[1]}

# --- Plotting ---
plt.figure(figsize=(8, 5))
sns.violinplot(data=df, x='Condition', y='Entropy', palette=palette, inner='box', cut=0)
sns.stripplot(data=df, x='Condition', y='Entropy', color='black', size=3, jitter=True, alpha=0.6)

plt.title("Shannon Entropy of Redox Signals\n(Synthetic Peptide Oxidation Data)", fontsize=14)
plt.ylabel("Shannon Entropy (bits)")
plt.xlabel("")
plt.tight_layout()
plt.savefig("shannon_entropy_violin.png", dpi=300)
plt.show()

# --- Optional: Statistical Test ---
t_stat, p_val = ttest_ind(entropy_case, entropy_control)
print(f"T-test p-value: {p_val:.4e}")
