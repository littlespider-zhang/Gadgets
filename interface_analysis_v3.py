from pymol import cmd, stored
import csv

# =========================
# 三字母 → 单字母
# =========================
aa_dict = {
    "ALA":"A","ARG":"R","ASN":"N","ASP":"D",
    "CYS":"C","GLN":"Q","GLU":"E","GLY":"G",
    "HIS":"H","ILE":"I","LEU":"L","LYS":"K",
    "MET":"M","PHE":"F","PRO":"P","SER":"S",
    "THR":"T","TRP":"W","TYR":"Y","VAL":"V"
}

# =========================
# sequence window（预计算 + 修复版）
# =========================
def get_interface_sequences(model, chainA, chainB, window=3):
    """
    一次性为两条链生成 {resi: window_seq} 字典
    - 使用 stored 避免命名空间问题
    - resi 保留 PyMOL 原生字符串（支持 insertion code 如 123A）
    - 按残基序号排序
    """
    def _get_seq_dict(chain):
        stored.reslist = []
        cmd.iterate(f"{model} and chain {chain} and name CA",
                    "stored.reslist.append((resi, resn))")

        # 排序：优先按数字部分排序（兼容 insertion code）
        def sort_key(item):
            resi_str = item[0]
            try:
                # 提取纯数字部分（如 "123A" → 123）
                num = int(''.join(filter(str.isdigit, resi_str))) if resi_str else 0
            except:
                num = 0
            return num

        reslist = sorted(stored.reslist, key=sort_key)

        seq_dict = {}
        for i, (resi, resn) in enumerate(reslist):
            sub = reslist[max(0, i - window): i + window + 1]
            seq = "".join([aa_dict.get(r[1], "X") for r in sub])
            seq_dict[resi] = seq                     # key 保持 PyMOL 原生 resi 字符串

        return seq_dict

    return _get_seq_dict(chainA), _get_seq_dict(chainB)


# =========================
# interaction classifier（保持原逻辑）
# =========================
def classify_interaction(resn1, resn2, atom1, atom2, dist):
    pos = ["ARG", "LYS", "HIS"]
    neg = ["ASP", "GLU"]
    aro = ["PHE", "TYR", "TRP"]
    hyd = ["VAL", "LEU", "ILE", "MET", "ALA", "PRO"] + aro

    if (resn1 in pos and resn2 in neg) or (resn2 in pos and resn1 in neg):
        # from wikepedia: salt bridge, < 4.0 A is required; note Tyrosine / Serione may also participate
        if dist <= 4.0:
            return "salt_bridge"

    if ("N" in atom1 or "O" in atom1) and ("N" in atom2 or "O" in atom2):
        if dist <= 3.5:
            return "hydrogen_bond"

    if (resn1 in pos and resn2 in aro) or (resn2 in pos and resn1 in aro):
        if dist <= 6.0:
            return "cation_pi"

    if resn1 in hyd and resn2 in hyd:
        # from Molecular Biology of the Gene
        # vdw radius:
        # -CH3 -> 2.0 A; Half thickness of aromatic molecule 1.7 A
        # S -> 1.85 A; P -> 1.9 A; O -> 1.4 A; N -> 1.5 A
        if dist <= 5.0:
            return "hydrophobic"

    return "vdw_contact"


# =========================
# main function（完整修复版）
# =========================
def analyze_interface(model, chainA, chainB, cutoff=4.0, outfile="interface.csv"):
    print(f"Analyzing model: {model}, {chainA} vs {chainB}")

    # interface selection
    cmd.select("A_int", f"byres (chain {chainA} within {cutoff} of chain {chainB})")
    cmd.select("B_int", f"byres (chain {chainB} within {cutoff} of chain {chainA})")

    # 预计算两条链的局部序列（大幅提升效率）
    seqA_dict, seqB_dict = get_interface_sequences(model, chainA, chainB)

    # 查找原子对（关键修复：使用 index 而不是 id）
    pairs = cmd.find_pairs(
        "A_int and not hydro",
        "B_int and not hydro",
        cutoff=cutoff
    )
    print(f"Found {len(pairs)} atom pairs")

    results = {}

    for (obj1, idx1), (obj2, idx2) in pairs:
        sel1 = f"{obj1} and index {idx1}"
        sel2 = f"{obj2} and index {idx2}"

        # 直接获取单个原子（get_model 返回列表，但单原子选择只需 [0]）
        atoms1 = cmd.get_model(sel1).atom
        atoms2 = cmd.get_model(sel2).atom
        if not atoms1 or not atoms2:
            continue

        a1 = atoms1[0]
        a2 = atoms2[0]

        dist = cmd.get_distance(sel1, sel2)

        resA = f"{aa_dict.get(a1.resn, 'X')}{a1.resi}_{chainA}"
        resB = f"{aa_dict.get(a2.resn, 'X')}{a2.resi}_{chainB}"

        mode = classify_interaction(
            a1.resn, a2.resn,
            a1.name, a2.name,
            dist
        )

        seqA = seqA_dict.get(a1.resi, "NA")
        seqB = seqB_dict.get(a2.resi, "NA")

        key = (resA, resB)

        priority = {
            "salt_bridge": 5,
            "hydrogen_bond": 4,
            "cation_pi": 3,
            "hydrophobic": 2,
            "vdw_contact": 1
        }

        # 保留优先级最高的相互作用类型
        if key not in results or priority.get(mode, 0) > priority.get(results[key][0], 0):
            results[key] = (mode, dist, seqA, seqB)

    # =========================
    # write CSV（列名更清晰）
    # =========================
    with open(outfile, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["distance", "interaction_mode", "Name_chainA", "Name_chainB"])

        for (rA, rB), (mode, dist, seqA, seqB) in results.items():
            writer.writerow([
                round(dist, 2),
                mode,
                f"{rA}|{seqA}",
                f"{rB}|{seqB}"
            ])

    print(f"Saved to {outfile} ({len(results)} residue pairs)")


# register command
cmd.extend("analyze_interface", analyze_interface)