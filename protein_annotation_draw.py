# -*- coding: utf-8 -*-
"""
蛋白质序列标注可视化脚本

输入:
    1. protein_sequence.txt —— 仅含蛋白质序列（可单行或多行，自动去除空白字符）
    2. anotation.txt        —— 第一行为蛋白质名称；
                               其余每行一条标注，格式: name, start_point, end_point, color
                               （start/end 为氨基酸编号，包含边界；color 可空缺）

输出:
    1. {蛋白质名称}_anotation.png —— 标注可视化图（dpi=300）
    2. {蛋白质名称}_anotation.txt —— anotation.txt 全部内容 + 蛋白质序列
"""

import os
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# ---------------------------------------------------------------------------
# 低饱和度色盘（参考 ColorBrewer Set2，常见于 Nature / Cell / Science 插图）
# 当 anotation 中 color 字段为空时，按标注顺序循环取用
# ---------------------------------------------------------------------------
DEFAULT_PALETTE = [
    "#66C2A5",  # 青绿
    "#FC8D62",  # 柔橙
    "#8DA0CB",  # 灰蓝紫
    "#E78AC3",  # 柔粉
    "#A6D854",  # 黄绿
    "#FFD92F",  # 柔黄
    "#E5C494",  # 米色
    "#B3B3B3",  # 浅灰
]

FONT_NAME = "Arial"          # 字体（系统无 Arial 时 matplotlib 自动回退）
LINE_COLOR = "#7F7F7F"       # 灰色线颜色
LINE_WIDTH_PT = 6            # 灰色线线宽（磅）
BOX_HEIGHT_RATIO = 1.5       # 标注方块高度 = 灰色线线宽 × 1.5
LABEL_FONTSIZE = 10                  # 两端数字标识字号
ANNO_NAME_FONTSIZE = LABEL_FONTSIZE         # 标注名称字号 = 数字标识字号
NAME_FONTSIZE = LABEL_FONTSIZE * 2          # 蛋白质名称字号 = 数字标识 × 2


# ---------------------------------------------------------------------------
# 输入读取
# ---------------------------------------------------------------------------
def read_sequence(path):
    """读取蛋白质序列，去除所有空白字符（支持多行 FASTA 之外的纯序列）。"""
    with open(path, "r", encoding="utf-8") as f:
        seq = "".join(f.read().split())
    if not seq:
        raise ValueError(f"序列文件为空: {path}")
    return seq.upper()


def read_annotations(path):
    """
    读取 anotation.txt。
    返回 (protein_name, annotations)，annotations 为字典列表:
        {"name": str, "start": int, "end": int, "color": str or None}
    """
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]

    if not lines:
        raise ValueError(f"标注文件为空: {path}")

    protein_name = lines[0]
    annotations = []
    for i, ln in enumerate(lines[1:], start=2):
        parts = [p.strip() for p in ln.split(",")]
        if len(parts) < 3:
            raise ValueError(f"第 {i} 行格式错误（至少需要 name, start, end）: {ln}")
        name = parts[0]
        start, end = int(parts[1]), int(parts[2])
        color = parts[3] if len(parts) >= 4 and parts[3] else None
        if start > end:
            start, end = end, start
        annotations.append({"name": name, "start": start, "end": end, "color": color})
    return protein_name, annotations


# ---------------------------------------------------------------------------
# 绘图
# ---------------------------------------------------------------------------
def _pt_to_data(ax, pt):
    """把磅（point）换算成当前坐标系的 y 数据单位高度。"""
    fig = ax.figure
    fig.canvas.draw()  # 确保坐标变换已建立
    inches = pt / 72.0
    inv = ax.transData.inverted()
    (_, y0), (_, y1) = inv.transform([(0, 0), (0, inches * fig.dpi)])
    return abs(y1 - y0)


def _spread_labels(labels, ax, fontsize):
    """
    数字标识沿 x 方向去重叠（y 保持不变，仍与灰色线两端数字水平对齐）。
    labels: [[x, text], ...]，返回调整后的 x 列表。
    """
    fig = ax.figure
    fig.canvas.draw()
    inv = ax.transData.inverted()
    (x0, _), (x1, _) = inv.transform([(0, 0), (1, 0)])
    data_per_px = abs(x1 - x0)
    # 估算每个数字标识的宽度（数据单位）：字宽 ≈ 0.62 em
    widths = [len(t) * 0.62 * fontsize / 72.0 * fig.dpi * data_per_px
              for _, t in labels]

    order = sorted(range(len(labels)), key=lambda i: labels[i][0])
    xs = [labels[i][0] for i in order]
    ws = [widths[i] for i in order]
    pad = 0.35 * fontsize / 72.0 * fig.dpi * data_per_px  # 标识间距

    for _ in range(20):  # 松弛迭代，把过近的标识向两侧推开
        moved = False
        for i in range(len(xs) - 1):
            gap = (ws[i] + ws[i + 1]) / 2.0 + pad
            if xs[i + 1] - xs[i] < gap:
                mid = (xs[i] + xs[i + 1]) / 2.0
                xs[i], xs[i + 1] = mid - gap / 2.0, mid + gap / 2.0
                moved = True
        if not moved:
            break

    new_x = [0.0] * len(labels)
    for rank, i in enumerate(order):
        new_x[i] = xs[rank]
    return new_x


def _assign_levels(labels, ax, fontsize):
    """
    为标注名称分配上下层级，避免相邻名称重叠。
    labels: [[center_x, text], ...]，返回每个名称的层级（0 为最贴近方块的层）。
    """
    fig = ax.figure
    fig.canvas.draw()
    inv = ax.transData.inverted()
    (x0, _), (x1, _) = inv.transform([(0, 0), (1, 0)])
    data_per_px = abs(x1 - x0)
    widths = [len(t) * 0.58 * fontsize / 72.0 * fig.dpi * data_per_px
              for _, t in labels]
    pad = 0.5 * fontsize / 72.0 * fig.dpi * data_per_px

    order = sorted(range(len(labels)), key=lambda i: labels[i][0])
    levels = [0] * len(labels)
    level_right = []  # 每层当前最右边界
    for i in order:
        cx, hw = labels[i][0], widths[i] / 2.0
        lv = 0
        while lv < len(level_right) and cx - hw < level_right[lv] + pad:
            lv += 1
        if lv == len(level_right):
            level_right.append(-1e18)
        level_right[lv] = cx + hw
        levels[i] = lv
    return levels


def plot_protein(protein_name, sequence, annotations, out_png):
    n = len(sequence)

    # 图宽随序列长度伸缩，保证长序列下数字不拥挤
    fig_w = max(8.0, min(20.0, n / 25.0))
    fig, ax = plt.subplots(figsize=(fig_w, 2.4))

    # 坐标范围：x 轴为氨基酸编号，左右各留 8% 余量放名称与数字
    margin = max(n * 0.08, 8)
    ax.set_xlim(1 - margin, n + margin)
    ax.set_ylim(-1.0, 1.0)
    ax.axis("off")

    y_line = 0.0

    # --- 3.1 灰色线：代表全部蛋白序列 ---
    ax.plot([1, n], [y_line, y_line],
            color=LINE_COLOR, linewidth=LINE_WIDTH_PT,
            solid_capstyle="butt", zorder=1)

    # 线宽（磅）→ 数据单位，用于计算方块高度（= 线宽 × 1.5）
    line_h = _pt_to_data(ax, LINE_WIDTH_PT)
    box_h = line_h * BOX_HEIGHT_RATIO

    # 数字标识的 y 位置（灰色线两端数字与标注边界数字共用，保证水平对齐）
    y_label = y_line - box_h / 2 - line_h * 1.2

    # 蛋白质名称：灰色线左端，Arial，距离灰色线更远（约原来的 4 倍）
    ax.text(1 - margin * 0.9, y_line, protein_name,
            ha="right", va="center",
            fontsize=NAME_FONTSIZE, fontfamily=FONT_NAME, zorder=3)

    # 收集所有数字标识（灰色线两端 + 各标注边界），统一去重叠后绘制，
    # 保证全部水平对齐、字体字号一致；相同位置的重复数字只画一次
    labels = [[1.0, "1"], [float(n), str(n)]]
    seen = {(1.0, "1"), (float(n), str(n))}

    def _add_label(x, text):
        key = (float(x), text)
        if key not in seen:
            seen.add(key)
            labels.append([float(x), text])

    # --- 3.2 / 3.3 标注方块 ---
    palette_idx = 0
    for anno in annotations:
        start = max(1, anno["start"])
        end = min(n, anno["end"])

        color = anno["color"]
        if not color:  # color 空缺 → 低饱和度色盘循环取色
            color = DEFAULT_PALETTE[palette_idx % len(DEFAULT_PALETTE)]
            palette_idx += 1

        rect = Rectangle((start, y_line - box_h / 2),
                         width=end - start + 1,  # 包含边界
                         height=box_h,
                         facecolor=color, edgecolor="none", zorder=2)
        ax.add_patch(rect)

        _add_label(anno["start"], str(anno["start"]))
        _add_label(anno["end"], str(anno["end"]))

    # --- 标注名称：方块上方居中，Arial，字号与数字标识一致；
    #     相邻名称重叠时自动分层错开 ---
    name_labels = [[(max(1, a["start"]) + min(n, a["end"])) / 2.0, a["name"]]
                   for a in annotations]
    if name_labels:
        levels = _assign_levels(name_labels, ax, ANNO_NAME_FONTSIZE)
        level_step = _pt_to_data(ax, ANNO_NAME_FONTSIZE * 1.6)
        y_name0 = y_line + box_h / 2 + line_h * 0.8
        for (cx, text), lv in zip(name_labels, levels):
            ax.text(cx, y_name0 + lv * level_step, text,
                    ha="center", va="bottom",
                    fontsize=ANNO_NAME_FONTSIZE, fontfamily=FONT_NAME, zorder=3)

    new_xs = _spread_labels(labels, ax, LABEL_FONTSIZE)
    for (x, text), nx in zip(labels, new_xs):
        ax.text(nx, y_label, text, ha="center", va="top",
                fontsize=LABEL_FONTSIZE, fontfamily=FONT_NAME,
                fontweight="bold", zorder=3)  # 数字标识加粗

    fig.savefig(out_png, dpi=300, bbox_inches="tight",
                facecolor="white", pad_inches=0.15)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 合并输出文件
# ---------------------------------------------------------------------------
def write_combined_file(protein_name, sequence, anno_path, out_dir):
    """生成 {蛋白质名称}_anotation.txt：anotation.txt 全部内容 + 蛋白质序列。"""
    with open(anno_path, "r", encoding="utf-8") as f:
        anno_content = f.read().rstrip("\n")

    out_path = os.path.join(out_dir, f"{protein_name}_anotation.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(anno_content + "\n\n")
        f.write(">sequence\n")
        f.write(sequence + "\n")
    return out_path


# ---------------------------------------------------------------------------
# 主函数
# ---------------------------------------------------------------------------
def main(seq_file="protein_sequence.txt", anno_file="anotation.txt", out_dir="."):
    sequence = read_sequence(seq_file)
    protein_name, annotations = read_annotations(anno_file)

    os.makedirs(out_dir, exist_ok=True)
    out_png = os.path.join(out_dir, f"{protein_name}_anotation.png")
    plot_protein(protein_name, sequence, annotations, out_png)
    out_txt = write_combined_file(protein_name, sequence, anno_file, out_dir)

    print(f"蛋白质: {protein_name}  长度: {len(sequence)} aa  标注数: {len(annotations)}")
    print(f"图像已保存: {out_png}")
    print(f"合并文件已保存: {out_txt}")


if __name__ == "__main__":
    main()
