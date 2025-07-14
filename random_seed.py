# Install if not already available
!pip install openpyxl --quiet

import numpy as np
import pandas as pd

# Set random seed for reproducibility
np.random.seed(42)

# Parameters
n_peptides = 100
n_replicates = 3
bins = np.linspace(0, 1, 11)  # 10 bins (0-100% oxidation)

# Generate synthetic data for Control and Case
def generate_redox_data(mean, std, n_peptides, n_replicates):
    data = np.clip(np.random.normal(loc=mean, scale=std, size=(n_peptides, n_replicates)), 0, 1)
    return data

control = generate_redox_data(0.2, 0.1, n_peptides, n_replicates)
case    = generate_redox_data(0.3, 0.1, n_peptides, n_replicates)

# Discretize into bins and calculate normalized distributions
def compute_binned_distribution(data, bins):
    hist = np.histogram(data, bins=bins)[0].astype(float)
    return hist / hist.sum()

control_flat = control.flatten()
case_flat = case.flatten()

control_dist = compute_binned_distribution(control_flat, bins)
case_dist = compute_binned_distribution(case_flat, bins)

# Save to Excel
df = pd.DataFrame({
    'Bin Range': [f'{bins[i]:.1f}–{bins[i+1]:.1f}' for i in range(len(bins)-1)],
    'Control': control_dist,
    'Case': case_dist
})

excel_path = 'synthetic_redox_probabilities.xlsx'
df.to_excel(excel_path, index=False)

print(f"✅ Excel file saved: {excel_path}")
