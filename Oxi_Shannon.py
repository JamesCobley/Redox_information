import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import entropy, ttest_ind

# --- Params ---
n_reps = 5
bins = np.linspace(0, 1, 21)  # 20 bins: 0–1 in 5% steps
np.random.seed(42)

# --- Load & extract brain columns ---
df = pd.read_csv("site_all (3).csv", low_memory=False)

mean_y = df["oxi_percent_brain.young"]
sem_y  = df["se_brain.young"]
mean_o = df["oxi_percent_brain.old"]
sem_o  = df["se_brain.old"]

# Drop any site missing data in either group
mask = mean_y.notna() & sem_y.notna() & mean_o.notna() & sem_o.notna()
mean_y, sem_y, mean_o, sem_o = mean_y[mask], sem_y[mask], mean_o[mask], sem_o[mask]
n_sites = len(mean_y)
print(f"Using {n_sites} brain sites with full data.\n")

# --- NEW: Mean redox state as percentage ---
mean_redox_y = mean_y.mean() 
mean_redox_o = mean_o.mean() 
print(f"Mean Brain Redox State:")
print(f"  Young: {mean_redox_y:.1f}%")
print(f"  Old:   {mean_redox_o:.1f}%\n")

# --- Simulate replicates per site ---
def simulate_group(means, sems):
    sd = sems * np.sqrt(n_reps)
    sims = np.random.normal(loc=means.values[:, None],
                             scale=sd.values[:, None])
    return np.clip(sims, 0, 1)

sims_y = simulate_group(mean_y, sem_y)
sims_o = simulate_group(mean_o, sem_o)

# --- Compute entropy per site ---
def per_site_entropy(sim_matrix):
    ent = []
    for row in sim_matrix:
        hist, _ = np.histogram(row, bins=bins, density=True)
        ent.append(entropy(hist + 1e-10, base=2))
    return np.array(ent)

ent_y = per_site_entropy(sims_y)
ent_o = per_site_entropy(sims_o)


# 1) Summary stats helper
def summary_stats(arr):
    m  = arr.mean()
    sd = arr.std(ddof=1)
    cv = sd / m if m else np.nan
    return m, sd, cv

m_y, sd_y, cv_y = summary_stats(ent_y)
m_o, sd_o, cv_o = summary_stats(ent_o)

# 2) Cohen's d (unpaired, pooled SD)
def cohen_d(x, y):
    nx, ny = len(x), len(y)
    vx, vy = x.var(ddof=1), y.var(ddof=1)
    s_pooled = np.sqrt(((nx-1)*vx + (ny-1)*vy) / (nx+ny-2))
    return (y.mean() - x.mean()) / s_pooled

d = cohen_d(ent_y, ent_o)

# 3) Print results
print(f"Brain entropy (per-site):")
print(f"  Young —  mean = {m_y:.3f} bits,  SD = {sd_y:.3f},  CV = {cv_y:.3f}")
print(f"  Old   —  mean = {m_o:.3f} bits,  SD = {sd_o:.3f},  CV = {cv_o:.3f}")
print(f"\nCohen’s d (Old vs Young) = {d:.3f}")


# --- Assemble for plotting & test ---
out = pd.DataFrame({
    "Entropy": np.concatenate([ent_y, ent_o]),
    "Age":      ["Young"]*n_sites + ["Old"]*n_sites
})

plt.figure(figsize=(6,4))
sns.violinplot(data=out, x="Age", y="Entropy", inner="box", cut=0)
sns.stripplot(data=out, x="Age", y="Entropy", color="k", size=3, jitter=True, alpha=0.6)
plt.title(f"Per-Site Shannon Entropy\nBrain (n sites={n_sites})")
plt.ylabel("Entropy (bits)")
plt.tight_layout()
plt.savefig("brain_entropy.png", dpi=300)
plt.show()

# --- Statistical comparison ---
t_stat, p_val = ttest_ind(ent_y, ent_o)
print(f"Brain entropy: Young mean={ent_y.mean():.3f}, Old mean={ent_o.mean():.3f}")
print(f"T-test p-value: {p_val:.3e}")
