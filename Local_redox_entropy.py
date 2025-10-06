# === Internal Redox Entropy per Protein (OxiMouse) — robust version ===
# Requires: df with columns Y_COL, O_COL, and a protein identifier column.

PROT_CANDS = ["Uniprot","UniprotID","Accession","Protein","ProteinID",
              "Majority.protein.IDs","Leading.razor.protein","Protein.Group"]
MIN_SITES  = 2         # <-- require at least 2 sites per protein
EPS        = 0.001     # keep in sync with your global analysis
ROUNDING   = "round"   # 'round' | 'floor' | 'ceil'

def find_prot_col(df, cands):
    for c in cands:
        if c in df.columns:
            return c
    raise ValueError(f"Could not find a protein column in {cands}")

def quantize(vals, eps, mode="round"):
    m = int(np.floor(1.0/eps)) + 1
    x = np.clip(vals, 0.0, 1.0) / eps
    if mode == "round": q = np.rint(x)
    elif mode == "floor": q = np.floor(x)
    elif mode == "ceil": q = np.ceil(x)
    else: raise ValueError("ROUNDING must be 'round'|'floor'|'ceil'")
    q = q.astype(int)
    q[q < 0] = 0; q[q > m-1] = m-1
    return q, m

def protein_entropy_metrics(vals, eps, mode="round"):
    """
    Returns (Cε, Eε, Sε) for a vector of [0,1] values from one protein.
    Coverage Cε: fraction of GLOBAL microstates touched.
    Evenness Eε: 1 - (1/2) * ||p_loc - u_loc||_1, where u_loc is uniform over the
                 microstates that THIS protein actually occupies (avoids penalizing
                 small proteins for not using the whole global grid).
    """
    q, M = quantize(vals, eps, mode)                 # 0..M-1 indices
    counts = np.bincount(q, minlength=M).astype(float)
    N      = counts.sum()
    occ    = counts > 0
    k      = int(occ.sum())                          # number of bins this protein uses
    # Coverage over global grid
    C = k / M if M else np.nan
    if k == 0:
        return np.nan, np.nan, np.nan               # no valid data
    # Local distribution (restricted to occupied bins)
    p_loc = counts[occ] / N                          # sums to 1 over k bins
    u_loc = np.full(k, 1.0 / k)                      # uniform over occupied bins
    E = 1.0 - 0.5 * np.abs(p_loc - u_loc).sum()      # ∈ [0,1]
    S = C * E
    return C, E, S

def per_protein_entropy(df, y_col, o_col, eps, mode, min_sites=2):
    prot_col = find_prot_col(df, PROT_CANDS)
    out = []
    skipped = {"missing":0, "lt_min":0}
    for prot, sub in df.groupby(prot_col):
        y = to_unit(sub[y_col]); o = to_unit(sub[o_col])
        mask = y.notna() & o.notna()
        yv = y[mask].to_numpy(); ov = o[mask].to_numpy()
        if yv.size < min_sites:
            skipped["lt_min"] += 1
            continue
        CY, EY, SY = protein_entropy_metrics(yv, eps, mode)
        CO, EO, SO = protein_entropy_metrics(ov, eps, mode)
        out.append({
            "Protein": str(prot),
            "n_sites": int(yv.size),
            "Cε_young": CY, "Eε_young": EY, "Sε_young": SY,
            "Cε_old":   CO, "Eε_old":   EO, "Sε_old":   SO,
            "ΔSε": SO - SY
        })
    res = pd.DataFrame(out)
    res.attrs["skipped_lt_min_sites"] = skipped["lt_min"]
    res.attrs["min_sites"] = min_sites
    return res

# --- run ---
protein_entropy = per_protein_entropy(df, Y_COL, O_COL, EPS, ROUNDING, MIN_SITES)

print(f"Proteins retained (≥{MIN_SITES} sites): {len(protein_entropy)}")
print(f"Proteins skipped (<{MIN_SITES} sites): {protein_entropy.attrs['skipped_lt_min_sites']}")

protein_entropy.sort_values("ΔSε").to_csv(
    "/content/protein_redox_entropy.csv", index=False)
print("Saved → /content/protein_redox_entropy.csv")

# Quick look at distribution
plt.figure(figsize=(6,4))
plt.hist(protein_entropy["ΔSε"].dropna(), bins=40, color="slateblue", alpha=0.85)
plt.axvline(0, color="k", ls="--")
plt.xlabel("ΔSε (Old − Young)"); plt.ylabel("# Proteins")
plt.title(f"Per-protein redox entropy shift (ΔSε), min_sites={MIN_SITES}")
plt.tight_layout(); plt.savefig("/content/protein_redox_entropy_hist.png", dpi=300); plt.show()
