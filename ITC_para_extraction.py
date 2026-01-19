import pandas as pd
import numpy as np
import re
filename = "E:/ITC/20260117_mRE-B&B-30aa_with_AMP-NAD+-NADH/Results Table.csv"


# -----------------------------
# Load ITC results table
# -----------------------------
df = pd.read_csv(filename)

# -----------------------------
# Rename columns
# -----------------------------
df = df.rename(columns={
    'KD (M) ': 'Kd',
    'KD Error (M)': 'Kd_err',
    '∆H (kcal/mol)': 'dH',
    '∆H Error (kcal/mol)': 'dH_err',
    'N (sites)': 'Nsites',
    'N Error (sites)': 'Nsites_err',
    '∆G (kcal/mol)': 'dG',
    '-T∆S (kcal/mol)': '-TdS'
})

# -----------------------------
# Format Kd
# -----------------------------
def format_kd(kd, kd_err):
    if pd.isna(kd):
        return np.nan

    if kd >= 1e-3:
        scale, unit = 1e3, "mM"
    elif kd >= 1e-6:
        scale, unit = 1e6, "μM"
    else:
        scale, unit = 1e9, "nM"

    value = kd * scale
    error = kd_err * scale

    return f"{value:.1f} ± {error:.1f} {unit}"

# -----------------------------
# Format ΔH (kcal/mol)
# -----------------------------
def format_dh(dh, dh_err):
    if pd.isna(dh):
        return np.nan
    return f"{dh:.1f} ± {dh_err:.1f} kcal/mol"

# -----------------------------
# Format Nsites
# -----------------------------
def format_nsites(n, n_err):
    if pd.isna(n):
        return np.nan
    return f"{n:.3f} ± {n_err:.3f}"

# -----------------------------
# Apply formatting
# -----------------------------
df['Kd'] = df.apply(lambda r: format_kd(r['Kd'], r['Kd_err']), axis=1)
df['ΔH'] = df.apply(lambda r: format_dh(r['dH'], r['dH_err']), axis=1)
df['Nsites'] = df.apply(lambda r: format_nsites(r['Nsites'], r['Nsites_err']), axis=1)

# -----------------------------
# Select final output
# -----------------------------
output = df[[
    'Filename ',
    'Nsites',
    'Kd',
    'ΔH',
    'dG',
    '-TdS'
]]
# -----------------------------
# Natural sort helper
# -----------------------------
def natural_key(text):
    return [int(t) if t.isdigit() else t.lower()
            for t in re.split(r'(\d+)', text)]

# -----------------------------
# Sort by filename (natural order)
# -----------------------------
df = df.sort_values(by='Filename ',
                    key=lambda col: col.map(natural_key))

# -----------------------------
# Save formatted table
# -----------------------------
output.to_csv("ITC_thermodynamics_formatted.csv", index=False)

print("Formatting complete. File saved as ITC_thermodynamics_formatted.csv")
