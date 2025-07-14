# Install dependencies (if not already in your Colab)
!pip install numpy pandas matplotlib seaborn scikit-learn

# --- Imports ---
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mutual_info_score

# --- Parameters ---
np.random.seed(42)
n_peptides = 100
n_replicates = 3
bins = np.linspace(0, 1, 11)  # 10% oxidation bins

# --- Synthetic Data Generator ---
def generate_redox_data(mean, std, n_peptides, n_replicates):
    data = np.clip(np.random.normal(loc=mean, scale=std, size=(n_peptides, n_replicates)), 0, 1)
    return data

# --- Generate data ---
control = generate_redox_data(mean=0.2, std=0.1, n_peptides=n_peptides, n_replicates=n_replicates)
case    = generate_redox_data(mean=0.5, std=0.2, n_peptides=n_peptides, n_replicates=n_replicates)

# --- Combine data ---
all_data = np.vstack([control, case])
labels = np.array(['Control'] * n_peptides + ['Case'] * n_peptides)

# --- Compute MI per peptide ---
mi_scores = []
for i in range(n_peptides):
    peptide_data = np.concatenate([control[i], case[i]])
    condition_labels = ['Control'] * n_replicates + ['Case'] * n_replicates
    binned = np.digitize(peptide_data, bins)  # Bin the oxidation values
    mi = mutual_info_score(binned, condition_labels)
    mi_scores.append(mi)

# --- Plot ---
plt.figure(figsize=(8, 5))
sns.histplot(mi_scores, bins=20, kde=True, color='green')
plt.xlabel("Mutual Information (bits)")
plt.ylabel("Peptide Count")
plt.title("Mutual Information between Redox Bins and Condition\n(Synthetic Peptides)")
plt.tight_layout()
plt.savefig("mutual_information.png", dpi=300)
plt.show()

# --- Summary Table ---
mi_df = pd.DataFrame({
    'Peptide': [f"P{i+1}" for i in range(n_peptides)],
    'MI_bits': mi_scores
})
print(mi_df.head())

# Optional: Save to Excel for GitHub reproducibility
mi_df.to_excel("mutual_information_summary.xlsx", index=False)
