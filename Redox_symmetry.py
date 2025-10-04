# === Redox Symmetry + Shift on OxiMouse (Young vs Old) ===
# Uses the same EPS, ROUNDING, and helper functions you already defined above.

import numpy as np
import matplotlib.pyplot as plt

def normalized_hist(vals: np.ndarray, eps: float, mode: str = "round"):
    """Return normalized microstate histogram p (length m) and raw counts."""
    m = int(np.floor(1.0/eps)) + 1
    q = quantize(vals, eps, mode)         # indices 0..m-1
    counts = np.bincount(q, minlength=m).astype(float)
    p = counts / counts.sum()
    return p, counts

def l1_overlap_similarity(pA: np.ndarray, pB: np.ndarray) -> float:
    """Psi = 1 - 0.5 * L1 distance; in [0,1]."""
    return 1.0 - 0.5 * np.abs(pA - pB).sum()

def wasserstein_1_on_unit(pA: np.ndarray, pB: np.ndarray, eps: float) -> float:
    """
    W1 for discrete distributions on a 1D grid with spacing eps.
    Here the support is {0, eps, ..., 1}; diameter = 1, so W1 ∈ [0,1].
    """
    cA = np.cumsum(pA)
    cB = np.cumsum(pB)
    return float(np.abs(cA - cB).sum() * eps)

def symmetry_and_shift(y: np.ndarray, o: np.ndarray, eps: float, mode: str="round"):
    pY, cntY = normalized_hist(y, eps, mode)
    pO, cntO = normalized_hist(o, eps, mode)

    # Similarity (overlap) and asymmetry
    Psi = l1_overlap_similarity(pY, pO)       # in [0,1]
    A   = 1.0 - Psi                            # asymmetry (0 best, 1 worst)

    # Shift sensitivity (earth mover)
    W1  = wasserstein_1_on_unit(pY, pO, eps)   # in [0,1]

    # Diagnostics explaining "why"
    suppY = cntY > 0
    suppO = cntO > 0
    m = cntY.size
    kY, kO = int(suppY.sum()), int(suppO.sum())
    k_overlap = int((suppY & suppO).sum())
    k_exclusive = int((suppY ^ suppO).sum())   # occupied in only one condition

    # microstates where frequencies differ beyond tiny tol
    tol = 1e-12
    k_diff = int((np.abs(pY - pO) > tol).sum())

    # units (cysteines) that "sit in different microstates"
    # measured as L1 mass needing reassignment (in counts)
    N = len(y)  # = len(o) from your filtering
    units_to_move = int(0.5 * np.abs(cntY/cntY.sum() - cntO/cntO.sum()).sum() * N)

    diag = {
        "m_microstates": m,
        "occupied_young": kY,
        "occupied_old": kO,
        "occupied_overlap": k_overlap,
        "occupied_exclusive": k_exclusive,
        "microstates_different": k_diff,
        "units_to_move": units_to_move
    }
    return A, W1, Psi, pY, pO, diag

A, D, Psi, pY, pO, diag = symmetry_and_shift(y, o, EPS, ROUNDING)

print("\n=== Redox Symmetry & Shift (site-agnostic; resolution-aware) ===")
print(f"Resolution ε = {EPS:.3f} → microstates m = {diag['m_microstates']}")
print(f"Similarity Ψε (overlap)   = {Psi:.3f}")
print(f"Asymmetry  Aε = 1-Ψε       = {A:.3f}   (0 = identical, 1 = disjoint)")
print(f"Shift      Dε = W1(pY,pO)  = {D:.3f}   (0 = aligned, 1 = max shift)")
print("\nDiagnostics (why):")
for k, v in diag.items():
    print(f"  {k}: {v}")

# --- Complex map: Asymmetry (x) vs Shift (y) in the unit square ---
fig, ax = plt.subplots(figsize=(4.6,4.6))
ax.scatter([A], [D], s=120)
ax.set_xlim(0,1); ax.set_ylim(0,1)
ax.set_xlabel("Asymmetry Aε = 1 − Ψε")
ax.set_ylabel("Shift Dε = W1(pY,pO)")
ax.set_title("Redox symmetry–shift map (Young vs Old)")
# reference grid & corners
ax.plot([0,1],[0,0], lw=0.5, color='k'); ax.plot([0,1],[1,1], lw=0.5, color='k')
ax.plot([0,0],[0,1], lw=0.5, color='k'); ax.plot([1,1],[0,1], lw=0.5, color='k')
ax.text(0.01,0.02,"perfect match", fontsize=9, ha='left', va='bottom')
ax.text(0.99,0.98,"max asymmetry & shift", fontsize=9, ha='right', va='top')
for spine in ax.spines.values():
    spine.set_linewidth(0.8)
plt.tight_layout()
plt.savefig("/content/redox_symmetry_shift_map.png", dpi=300)
plt.show()
print("Saved: /content/redox_symmetry_shift_map.png")
