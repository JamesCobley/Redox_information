import numpy as np
import pandas as pd
from scipy.stats import entropy
from tqdm.auto import trange

# --- Params ---
n_reps = 5
bins   = np.linspace(0, 1, 21)   # 20 bins: 0–1 in 5% steps
n_boot = 1000
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

# --- 1) Pooled entropy of the *true* means ---
def pooled_entropy(vals, bins):
    hist, _ = np.histogram(vals, bins=bins, density=True)
    return entropy(hist + 1e-10, base=2)

H_y_true = pooled_entropy(mean_y.values, bins)
H_o_true = pooled_entropy(mean_o.values, bins)
print(f"Pooled Shannon entropy (means):")
print(f"  Young: {H_y_true:.3f} bits")
print(f"  Old:   {H_o_true:.3f} bits\n")

# --- 2) Bootstrap per-site entropy from SEM-derived replicates ---
def simulate_entropy(means, sems):
    sd = sems * np.sqrt(n_reps)
    sims = np.random.normal(loc=means.values[:,None],
                             scale=sd.values[:,None])
    sims = np.clip(sims, 0, 1)
    # pool across sites & replicates
    flat = sims.ravel()
    hist, _ = np.histogram(flat, bins=bins, density=True)
    return entropy(hist + 1e-10, base=2)

boot_y, boot_o = [], []
for _ in trange(n_boot):
    boot_y.append(simulate_entropy(mean_y, sem_y))
    boot_o.append(simulate_entropy(mean_o, sem_o))

# --- Summaries of bootstrap distributions ---
def summarize_bootstrap(arr):
    arr = np.array(arr)
    m  = arr.mean()
    sd = arr.std(ddof=1)
    cv = sd / m if m else np.nan
    ci_low, ci_high = np.percentile(arr, [2.5, 97.5])
    return m, sd, cv, ci_low, ci_high

my_b, sd_yb, cv_yb, lo_y, hi_y = summarize_bootstrap(boot_y)
mo_b, sd_ob, cv_ob, lo_o, hi_o = summarize_bootstrap(boot_o)

# --- Cohen's d on the *pooled* entropy estimates ---
# (difference of bootstrapped distributions)
def cohen_d(x, y):
    nx, ny = len(x), len(y)
    vx, vy = x.var(ddof=1), y.var(ddof=1)
    s_p = np.sqrt(((nx-1)*vx + (ny-1)*vy) / (nx+ny-2))
    return (np.mean(y) - np.mean(x)) / s_p

d = cohen_d(np.array(boot_y), np.array(boot_o))

# --- Print everything ---
print("Bootstrap summary of pooled-entropy estimates:")
print(f" Young — mean={my_b:.3f}, SD={sd_yb:.3f}, CV={cv_yb:.3f}, 95% CI=({lo_y:.3f},{hi_y:.3f})")
print(f" Old   — mean={mo_b:.3f}, SD={sd_ob:.3f}, CV={cv_ob:.3f}, 95% CI=({lo_o:.3f},{hi_o:.3f})")
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
