import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data (manually inputting the bin data as an example)
data = {
    'Bin Range': ['0.0–0.1', '0.1–0.2', '0.2–0.3', '0.3–0.4', '0.4–0.5', '0.5–0.6', '0.6–0.7', '0.7–0.8', '0.8–0.9', '0.9–1.0'],
    'Control': [0.15667, 0.32667, 0.38667, 0.10333, 0.02333, 0.00333, 0.0, 0.0, 0.0, 0.0],
    'Case':    [0.02333, 0.11667, 0.37,    0.35,    0.11667, 0.02,    0.00333, 0.0, 0.0, 0.0]
}
df = pd.DataFrame(data)

# Small epsilon to avoid division by zero or log(0)
epsilon = 1e-10
control = np.array(df['Control']) + epsilon
case = np.array(df['Case']) + epsilon

# Compute Fisher Information Metric approximation
log_ratio = np.log(case / control)
fim_components = (log_ratio ** 2) * control
fisher_information = np.sum(fim_components)

print(f"Fisher Information Metric: {fisher_information:.6f}")

# Plot (viridis style)
plt.figure(figsize=(10, 5))
colors = sns.color_palette("viridis", len(df))
plt.bar(df['Bin Range'], fim_components, color=colors)
plt.title("Fisher Information Metric Contribution per Bin")
plt.xlabel("Oxidation Bin")
plt.ylabel("Fisher Information Contribution")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
