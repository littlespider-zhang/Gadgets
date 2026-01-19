import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# 1:1 Binding ITC Simulation
# -----------------------------

# User-defined parameters
M_tot = 50*10**(-6)           # Protein concentration in cell (M)
V_cell = 200*10**(-6)           # Cell volume (L), used to scale heat if needed
Kd = 5*10**(-6)              # Dissociation constant (M)
dH = -10.0             # Enthalpy per mole of binding (kcal/mol)
n_injections = 20      # Number of ligand injections
V_inj = 2*10**(-6)           # Ligand injection volume (L)
L_syringe = 500*10**(-6)        # Ligand concentration in syringe (M)

# -----------------------------
# Function: 1:1 binding model
# -----------------------------
def bound_complex(M, L, Kd):
    """Calculate bound protein [PL] using the quadratic solution"""
    return ((M + L + Kd) - np.sqrt((M + L + Kd)**2 - 4*M*L)) / 2

# -----------------------------
# Simulate injections
# -----------------------------
L_total_added = 0
heats = []

for i in range(n_injections):
    # Ligand added in this injection
    L_total_added += L_syringe * V_inj / V_cell  # total ligand concentration in cell
    # Calculate new bound complex
    PL = bound_complex(M_tot, L_total_added, Kd)
    # Calculate heat released in this injection
    if i == 0:
        q = dH * PL * V_cell       # first injection, all binding counts; ignore V_cell change during injection
        prev_PL = PL
    else:
        q = dH * (PL - prev_PL) * V_cell  # only new binding counts; ignore V_cell change during injection
        prev_PL = PL
    heats.append(q)

# -----------------------------
# Plotting
# -----------------------------
molar_ratio = [(L_syringe*V_inj/V_cell)*(i+1)/M_tot for i in range(n_injections)]

plt.figure(figsize=(8,5))
plt.plot(molar_ratio, heats, 'o-', color='tab:blue', label='Simulated ITC')
plt.xlabel('Ligand/Protein Molar Ratio')
plt.ylabel('Heat per injection (kcal/mol)')
plt.title('Simulated ITC Experiment (1:1 Binding)')
plt.legend()
plt.grid(True)
plt.show()
