import math

# ProtParam residue masses (average isotopic mass, Da)
aa_mass = {
'A': 71.0788, 'R': 156.1875, 'N': 114.1038, 'D': 115.0886,
'C': 103.1388, 'E': 129.1155, 'Q': 128.1307, 'G': 57.0519,
'H': 137.1411, 'I': 113.1594, 'L': 113.1594, 'K': 128.1741,
'M': 131.1926, 'F': 147.1766, 'P': 97.1167, 'S': 87.0782,
'T': 101.1051, 'W': 186.2132, 'Y': 163.1760, 'V': 99.1326
}

# Water mass added to full polypeptide
H2O = 18.01524


def read_sequence_from_txt(filepath):
    # 20个标准氨基酸
    valid_aa = set("ACDEFGHIKLMNPQRSTVWY")

    sequence = ""

    with open(filepath, "r") as f:
        for line in f:
            for char in line.upper():
                if char in valid_aa:
                    sequence += char

    return sequence

def protparam_analysis(seq, conc_uM=10, path_length_cm=1):

    seq = seq.replace("\n", "").replace(" ", "").upper()

    # ---------- molecular weight ----------
    mw = sum(aa_mass[aa] for aa in seq) + H2O

    # ---------- residue counts ----------
    nW = seq.count("W")
    nY = seq.count("Y")
    nC = seq.count("C")

    # ---------- extinction coefficients ----------
    epsilon_reduced = 5500*nW + 1490*nY
    epsilon_oxidized = 5500*nW + 1490*nY + 125*(nC//2)

    # ---------- Beer-Lambert ----------
    conc_M = conc_uM * 1e-6

    A280_reduced = epsilon_reduced * conc_M * path_length_cm
    A280_oxidized = epsilon_oxidized * conc_M * path_length_cm

    # ---------- output ----------
    print("Sequence length:", len(seq))
    print(f"Molecular weight: {mw:.2f} Da")
    print()

    print("Molar extinction coefficient (M^-1 cm^-1)")
    print("Reduced:", epsilon_reduced)
    print("Non-reduced:", epsilon_oxidized)
    print()

    print(f"A280 at {conc_uM} µM (1 cm path)")
    print("Reduced:", round(A280_reduced,4))
    print("Non-reduced:", round(A280_oxidized,4))

    print(f"Copy&Paste: Reduced coefficient (M^-1 cm^-1) & MW (kDa)\n{epsilon_reduced}\t{mw/1000:.5f}")


if __name__ == "__main__":
    sequence_file = "protein_sequence.txt"
    sequence = read_sequence_from_txt(sequence_file)
    protparam_analysis(sequence)
