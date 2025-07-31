import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 1. Load & filter (point to wherever your CSV lives)
df = pd.read_csv("site_all (3).csv", low_memory=False)
my = df["oxi_percent_brain.young"]
mo = df["oxi_percent_brain.old"]
mask = my.notna() & mo.notna()
my, mo = my[mask], mo[mask]

# 2. Build flat arrays of values and age‐labels
vals = np.concatenate([my.values, mo.values])
ages = np.concatenate([np.zeros_like(my), np.ones_like(mo)])  # 0=Young, 1=Old

# 3. 2D histogram
#    - 20 bins across oxidation (0→1)
#    - 2 bins for age (one for Young, one for Old)
hist, xedges, yedges = np.histogram2d(
    vals,
    ages,
    bins=[np.linspace(0,1,21), [-0.5,0.5,1.5]],
    density=False
)

# 4. Plot as a heatmap
plt.figure(figsize=(6,4))
plt.imshow(
    hist.T,
    aspect="auto",
    origin="lower",
    extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]]
)
plt.colorbar(label="Count")
plt.yticks([0,1], ["Young","Old"])
plt.xlabel("Mean Oxidation (%)")
plt.ylabel("Age Group")
plt.title("Joint Histogram: Brain Oxidation vs. Age")
plt.tight_layout()
plt.savefig("brain_joint_hist.png", dpi=300)
plt.show()
