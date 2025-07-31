import numpy as np
import pandas as pd
from scipy.stats import entropy
from sklearn.metrics import mutual_info_score
from tqdm.auto import trange

# --- Params ---
n_reps = 5
bins   = np.linspace(0, 1, 21)  # 20 bins → 5% steps
n_boot = 1000
np.random.seed(42)

# --- Load & filter data for brain ---
df = pd.read_csv("site_all (3).csv", low_memory=False)
my = df["oxi_percent_brain.young"]
sy = df["se_brain.young"]
mo = df["oxi_percent_brain.old"]
so = df["se_brain.old"]

mask = my.notna() & sy.notna() & mo.notna() & so.notna()
mean_y, sem_y, mean_o, sem_o = my[mask], sy[mask], mo[mask], so[mask]
n_sites = len(mean_y)
print(f"Using {n_sites} brain sites with full mean+SEM data.\n")

# --- Helpers ---

def simulate_group(means, sems):
    """Simulate n_reps per-site values from N(mean, sem*sqrt(n_reps)), clipped to [0,1]."""
    sd = sems * np.sqrt(n_reps)
    sims = np.random.normal(loc=means.values[:, None], scale=sd.values[:, None])
    return np.clip(sims, 0, 1)

def pooled_entropy(vals, bins):
    """Shannon entropy of a pooled array of values."""
    hist, _ = np.histogram(vals, bins=bins, density=True)
    return entropy(hist + 1e-10, base=2)

def compute_mi(x, y, bins=20):
    """Mutual information between two 1D arrays, discretized into `bins` equal-width bins."""
    x_bin = pd.cut(x, bins=bins, labels=False)
    y_bin = pd.cut(y, bins=bins, labels=False)
    return mutual_info_score(x_bin, y_bin)

def bootstrap_stat(func, *args):
    """Bootstrap a statistic by repeatedly calling func(*args) and returning array of results."""
    out = []
    for _ in trange(n_boot, desc=func.__name__):
        out.append(func(*args))
    return np.array(out)

def summarize(arr):
    m  = arr.mean()
    sd = arr.std(ddof=1)
    cv = sd / m if m else np.nan
    lo, hi = np.percentile(arr, [2.5, 97.5])
    return m, sd, cv, lo, hi

def cohen_d(x, y):
    nx, ny = len(x), len(y)
    vx, vy = x.var(ddof=1), y.var(ddof=1)
    s_p = np.sqrt(((nx-1)*vx + (ny-1)*vy) / (nx+ny-2))
    return (y.mean() - x.mean()) / s_p

# --- 1) Pooled entropy of the *true* means ---
H_y_true = pooled_entropy(mean_y.values, bins)
H_o_true = pooled_entropy(mean_o.values, bins)
print("Pooled Shannon entropy (means):")
print(f"  Young = {H_y_true:.3f} bits")
print(f"  Old   = {H_o_true:.3f} bits\n")

# --- 2) Bootstrap pooled-entropy via SEM simulations ---
def pooled_entropy_sim(means, sems):
    sims = simulate_group(means, sems).ravel()
    return pooled_entropy(sims, bins)

boot_Hy = bootstrap_stat(pooled_entropy_sim, mean_y, sem_y)
boot_Ho = bootstrap_stat(pooled_entropy_sim, mean_o, sem_o)

mHy, sdHy, cvHy, loHy, hiHy = summarize(boot_Hy)
mHo, sdHo, cvHo, loHo, hiHo = summarize(boot_Ho)
d_entropy = cohen_d(boot_Hy, boot_Ho)

print("Bootstrap summary of pooled-entropy:")
print(f" Young = {mHy:.3f} ± {sdHy:.3f} (CV={cvHy:.3f}), 95% CI=[{loHy:.3f},{hiHy:.3f}]")
print(f" Old   = {mHo:.3f} ± {sdHo:.3f} (CV={cvHo:.3f}), 95% CI=[{loHo:.3f},{hiHo:.3f}]")
print(f"Cohen’s d (entropy Old vs Young) = {d_entropy:.3f}\n")

# --- 3) Mutual information between site mean oxidation & age ---
# Build combined arrays
values = np.concatenate([mean_y.values, mean_o.values])
ages   = np.concatenate([np.zeros_like(mean_y), np.ones_like(mean_o)])

mi_true = compute_mi(values, ages, bins=len(bins)-1)
print(f"Mutual Information (Oxidation ↔ Age): {mi_true:.3f} bits")

# --- 4) Bootstrap MI via SEM simulations ---
from tqdm.auto import trange

# --- Combined MI simulation + bootstrap ---
def simulate_and_compute_mi(mean_y, sem_y, mean_o, sem_o, bins):
    # 1) simulate Young and Old replicates
    sims_y = simulate_group(mean_y, sem_y)  # shape (n_sites, n_reps)
    sims_o = simulate_group(mean_o, sem_o)
    # 2) flatten and build labels
    vals = np.concatenate([sims_y.ravel(), sims_o.ravel()])
    labs = np.concatenate([
        np.zeros(sims_y.size, dtype=int),
        np.ones(sims_o.size,  dtype=int),
    ])
    # 3) compute MI
    return compute_mi(vals, labs, bins=len(bins)-1)

# 4) Bootstrap MI
boot_mi = []
for _ in trange(n_boot, desc="Bootstrapping MI"):
    boot_mi.append(simulate_and_compute_mi(mean_y, sem_y, mean_o, sem_o, bins))

# 5) Summarize
mi_mean, mi_sd, mi_cv, mi_lo, mi_hi = summarize(np.array(boot_mi))
print(f"\nMutual Information (Oxidation ↔ Age): {compute_mi(values, ages, bins=len(bins)-1):.4f} bits")
print(f"Bootstrapped MI: {mi_mean:.4f} ± {mi_sd:.4f} (CV={mi_cv:.3f}), 95% CI=[{mi_lo:.4f},{mi_hi:.4f}] bits")
