# -*- coding: utf-8 -*-
"""
蛋白质序列标注可视化脚本

输入（唯一输入文件，默认 annotator.txt，可用命令行参数指定）:
    第一行为蛋白质名称；其余部分按 ">段落名" 划分，">" 之前为 feature 区:
        feature 区每行一条标注，格式:
            name, start_point, end_point, TYPE, color
            start/end 为氨基酸编号（包含边界）；
            TYPE 为标注类型标签（可空缺，默认 DOMAIN）：
                DOMAIN —— 在主链上绘制彩色方块（默认类型）
                REGION —— 在氨基酸编号下方绘制括号（开口朝上），标注一段区域
                POINT  —— 位点处绘制红色单氨基酸方块 + 黑色竖线段，
                          名称右对齐至 start point（start 与 end 相同）
            color 可空缺，空缺时自动从色盘分配颜色
        截短体定义行（可多行，与 feature 行并列书写），格式:
            >TRUNCATION, start_point, end_point
            start/end 以全长蛋白编号为基准；每个截短体
            按与全长蛋白完全相同的标准生成 3 张图；
            ">TRUNCATION, None, None" 表示不进行截短
        >features hided 段: 不需要显示的 feature（格式同上），绘图时直接跳过
        >sequence 段: 蛋白质序列（可单行或多行，自动去除空白字符）
    输出 txt 与输入文件格式完全统一（颜色已补全），可直接作为输入二次绘制。

输出（dpi=300，全部保存在 {蛋白质名称}_annotation/ 文件夹下）:
    全长蛋白:
    1. {蛋白质名称}_annotation_full.png        —— 完整版：所有 feature + full 数字标注
    2. {蛋白质名称}_annotation_default.png     —— 所有 feature + default 数字标注
    3. {蛋白质名称}_annotation_REGION_only.png —— 简略版：DOMAIN / POINT 方块 +
                                                 REGION 括号（无名称与线段）
    4. {蛋白质名称}_annotation_simple.png      —— 缩略图：省略所有文字（保留蛋白
                                                 名称），隐藏 POINT feature；
                                                 feature 段与间隔段按顺序依次排列，
                                                 各段宽度 = ln(实际氨基酸数)
                                                 （间隔段额外乘 SIMPLE_GAP_SCALE 压缩）
    5. {蛋白质名称}_annotation_{date}.txt      —— 与输入格式统一的数据文件
                                                 （date 为绘制日期，如 20260928）：
                                                 feature 区（颜色已按实际用色补全）
                                                 + ">TRUNCATION, start, end" 定义行
                                                 + ">features hided"
                                                 + ">sequence" 全长序列 + 每个截短体
                                                 的 ">truncated sequence start-end"
                                                 序列（仅便于阅读）
    每个截短体（start-end）:
    6. {蛋白质名称}_{start}-{end}_annotation_full.png / _default.png /
       _REGION_only.png                        —— 与全长蛋白相同的 3 种图，
                                                 编号保留全长蛋白坐标
                                                 （截短体序列不单独保存，见上方
                                                 合并 txt 文件）

数字标注模式:
    full    —— 显示所有 DOMAIN 的起止编号
    default —— 首尾编号始终显示；紧邻的 DOMAIN 只显示后一个的 start；
               长度 < 10 的 DOMAIN 只显示 start 并标注长度，如 229(10)
"""

import math
import os
import sys
from datetime import datetime

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.transforms import blended_transform_factory

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

FEATURE_TYPES = ("DOMAIN", "REGION", "POINT")

SCALE = 2.0                # 整体放大倍数（画布与所有字体、线宽同步放大）
POINT_BAR_COLOR = "#FF0000"   # POINT 位点方块颜色（红色）
POINT_SEG_COLOR = "black"     # POINT 竖线段颜色（黑色，无箭头）
POINT_BAR_WIDTH_AA = 1.0      # ★ POINT 红色方块横向宽度，单位：氨基酸个数
                              #   （1.0 = 恰好 1 个氨基酸的长度，方块占据 [x, x+宽度) 区间）
REGION_BRACKET_COLOR = "black"  # REGION 括号颜色（统一为黑色）

# --- 画布横向尺寸：与序列长度成正比，保证任意长度蛋白的每个氨基酸实际像素一致 ---
# 标定比例（沿用原 393 aa 版式）：画布 25 aa/英寸（未乘 SCALE），x 轴两侧
# 留白各为序列长度的 8%，折算后 1 个氨基酸单位 = SCALE / (25×1.16) 英寸，
# 300 dpi 下约 20.7 px，与蛋白长度无关。
CANVAS_MARGIN_X_RATIO = 0.08                              # x 轴每侧留白 = 8% × 序列长度
CANVAS_INCH_PER_AA = SCALE / (25.0 * (1 + 2 * CANVAS_MARGIN_X_RATIO))  # 每氨基酸单位的画布英寸数

# --- 缩略图（simple）专用参数 ---
# 缩略图横轴为“分段对数压缩”坐标：主链被拆成 feature 段（DOMAIN）与间隔段，
# 依次排列；每段显示宽度 = ln(实际氨基酸数)，间隔段在此基础上再乘压缩比例。
SIMPLE_GAP_SCALE = 0.5             # ★ 间隔区宽度的额外压缩比例：
                                   #   间隔显示宽度 = ln(间隔氨基酸数) × SIMPLE_GAP_SCALE
SIMPLE_MIN_SEG_LOG = math.log(2.0) # 长度为 1 个氨基酸的区段（ln(1)=0）的显示宽度


# ---------------------------------------------------------------------------
# 输入读取
# ---------------------------------------------------------------------------
def _parse_feature_line(ln, lineno, path):
    """解析一条 feature 行 "name, start, end, TYPE, color"，返回 annotation 字典。"""
    parts = [p.strip() for p in ln.split(",")]
    if len(parts) < 3:
        raise ValueError(f"{path} 第 {lineno} 行格式错误"
                         f"（至少需要 name, start, end）: {ln}")
    name = parts[0]
    if name.upper() == "TRUNCATION":  # 旧格式提示
        raise ValueError(f"{path} 第 {lineno} 行：截短体定义请使用 "
                         f'">TRUNCATION, start, end" 格式: {ln}')
    start, end = int(parts[1]), int(parts[2])
    if start > end:
        start, end = end, start
    ftype = parts[3].upper() if len(parts) >= 4 and parts[3] else "DOMAIN"
    if ftype not in FEATURE_TYPES:
        raise ValueError(f"{path} 第 {lineno} 行类型标签错误"
                         f"（应为 DOMAIN / REGION / POINT）: {ln}")
    color = parts[4] if len(parts) >= 5 and parts[4] else None  # 空缺时自动配色
    return {"name": name, "start": start, "end": end,
            "color": color, "type": ftype}


def read_input(path):
    """
    读取统一输入文件（默认 annotator.txt）。
    文件结构：第一行为蛋白质名称；">" 段落之前为 feature 区（feature 行）；
    截短体定义行为 ">TRUNCATION, start, end"（以全长蛋白编号为基准，可多行；
    ">TRUNCATION, None, None" 表示不进行截短）；
    ">features hided" 段为不需要显示的 feature（绘图时跳过）；
    ">sequence" 段为蛋白质序列；其余 ">" 段（如 ">truncated sequence x-y"
    截短体序列段）仅便于阅读，读取与绘图时忽略。

    返回 (protein_name, annotations, truncations, hidden_lines, sequence)：
        annotations 为字典列表:
            {"name": str, "start": int, "end": int,
             "color": str or None, "type": "DOMAIN" | "REGION" | "POINT"}
        truncations 为 (start, end) 列表（以全长蛋白编号为基准），可为空；
        hidden_lines 为 ">features hided" 段的原始行（用于原样写入输出文件）。
    """
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    if not lines:
        raise ValueError(f"输入文件为空: {path}")

    # 按 ">" 段落拆分：">" 之前为头部（蛋白质名称 + feature 区）；
    # 特例：">TRUNCATION, start, end" 是截短体定义行而非段落标记，
    #       ">TRUNCATION, None, None" 表示不进行截短（直接忽略）
    header, sections, truncations = [], {}, []
    cur = None
    for i, ln in enumerate(lines, start=1):
        if ln.startswith(">"):
            rest = ln[1:].strip()
            parts = [p.strip() for p in rest.split(",")]
            if parts[0].upper() == "TRUNCATION" and len(parts) >= 3:
                if parts[1].lower() == "none" or parts[2].lower() == "none":
                    continue                      # 不进行截短
                t_start, t_end = int(parts[1]), int(parts[2])
                if t_start > t_end:
                    t_start, t_end = t_end, t_start
                truncations.append((t_start, t_end))
                continue                          # 定义行，不开启新段落
            cur = rest.lower()
            sections.setdefault(cur, [])
        elif cur is None:
            header.append(ln)
        else:
            sections[cur].append(ln)
    if not header:
        raise ValueError(f"输入文件缺少蛋白质名称行: {path}")

    protein_name = header[0]
    annotations = []
    for i, ln in enumerate(header[1:], start=2):
        annotations.append(_parse_feature_line(ln, i, path))

    # >sequence：蛋白质序列（必需）
    if "sequence" not in sections:
        raise ValueError(f"输入文件缺少 >sequence 部分: {path}")
    sequence = "".join("".join(sections["sequence"]).split()).upper()
    if not sequence:
        raise ValueError(f"输入文件 >sequence 部分为空: {path}")

    # >features hided：不需要显示的 feature，绘图时直接跳过
    hidden_lines = sections.get("features hided",
                                sections.get("features hidden", []))
    hidden_keys = set()
    for ln in hidden_lines:
        anno = _parse_feature_line(ln, 0, path)
        hidden_keys.add((anno["name"], anno["start"],
                         anno["end"], anno["type"]))
    if hidden_keys:
        annotations = [a for a in annotations
                       if (a["name"], a["start"], a["end"], a["type"])
                       not in hidden_keys]

    return protein_name, annotations, truncations, hidden_lines, sequence


# ---------------------------------------------------------------------------
# 绘图辅助
# ---------------------------------------------------------------------------
def _data_per_px(ax):
    """x 方向 1 像素对应的数据单位数（用于估算文字宽度）。"""
    fig = ax.figure
    fig.canvas.draw()
    inv = ax.transData.inverted()
    (x0, _), (x1, _) = inv.transform([(0, 0), (1, 0)])
    return abs(x1 - x0)


def _text_widths(ax, texts, fontsize, em=0.6):
    """估算文字宽度（数据单位）。"""
    dpp = _data_per_px(ax)
    return [len(t) * em * fontsize / 72.0 * ax.figure.dpi * dpp for t in texts]


def _spread_labels(labels, ax, fontsize):
    """
    数字标识沿 x 方向去重叠（y 保持不变，仍与灰色线两端数字水平对齐）。
    labels: [[x, text], ...]，返回调整后的 x 列表。
    """
    dpp = _data_per_px(ax)
    widths = [len(t) * 0.62 * fontsize / 72.0 * ax.figure.dpi * dpp
              for _, t in labels]
    pad = 0.35 * fontsize / 72.0 * ax.figure.dpi * dpp

    order = sorted(range(len(labels)), key=lambda i: labels[i][0])
    xs = [labels[i][0] for i in order]
    ws = [widths[i] for i in order]

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


def _assign_name_levels(labels, ax, fontsize):
    """
    按文字宽度为名称分配上下层级，避免相邻名称重叠。
    labels: [[center_x, text], ...]，返回层级列表（0 为最贴近主链的层）。
    """
    widths = _text_widths(ax, [t for _, t in labels], fontsize, em=0.58)
    dpp = _data_per_px(ax)
    pad = 0.5 * fontsize / 72.0 * ax.figure.dpi * dpp

    order = sorted(range(len(labels)), key=lambda i: labels[i][0])
    levels = [0] * len(labels)
    level_right = []
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


def _dodge_obstacles(cx, half_w, obstacles, pad):
    """水平微调名称中心，避开 x = obstacles 处的竖向箭头。"""
    for ox in sorted(obstacles):
        lo, hi = cx - half_w - pad, cx + half_w + pad
        if lo < ox < hi:
            # 向较近的一侧避让
            cx = ox + half_w + pad if ox - lo <= hi - ox else ox - half_w - pad
    return cx


def _assign_interval_levels(features):
    """按区间重叠为 REGION 括号分配层级（同一层的括号互不重叠）。"""
    order = sorted(range(len(features)), key=lambda i: features[i]["start"])
    levels = [0] * len(features)
    level_end = []
    for i in order:
        lv = 0
        while lv < len(level_end) and features[i]["start"] <= level_end[lv]:
            lv += 1
        if lv == len(level_end):
            level_end.append(-1)
        level_end[lv] = features[i]["end"]
        levels[i] = lv
    return levels


# ---------------------------------------------------------------------------
# 绘图
# ---------------------------------------------------------------------------
def _fit_region_fontsize(ax, features, fs_max, fs_min):
    """
    根据各 REGION 的区域长度和名称长度，自动选择合适字号：
    取能让每个名称都放入其区域宽度的最大字号，并统一应用于所有 REGION 名称。
    """
    if not features:
        return fs_max
    dpp = _data_per_px(ax)
    per_char_pt = 0.58 / 72.0 * ax.figure.dpi * dpp  # 1 pt 字号下单字符宽（数据单位）
    fs = fs_max
    for a in features:
        region_len = a["end"] - a["start"] + 1
        fit = region_len / max(len(a["name"]) * per_char_pt, 1e-9)
        fs = min(fs, fit)
    return max(fs, fs_min)


def _default_number_labels(domains, n, offset=0):
    """
    default 标注模式的数字标识：
      1) 首尾氨基酸序号始终显示；
      2) 两个 DOMAIN 紧邻（前一个 end + 1 == 后一个 start）时，
         仅显示后一个的 start，隐藏前一个的 end；
      3) 长度 < 10 的 DOMAIN 仅显示 start，并标注序列长度，如 229(10)。
    offset: 绘图局部坐标 → 全长蛋白编号的偏移（截短体 = t_start - 1）。
    """
    first_txt, last_txt = str(1 + offset), str(n + offset)
    labels = [[1.0, first_txt], [float(n), last_txt]]
    seen = {(1.0, first_txt), (float(n), last_txt)}

    def _add(x, text):
        key = (float(x), text)
        if key not in seen:
            seen.add(key)
            labels.append([float(x), text])

    dom_sorted = sorted(domains, key=lambda a: a["start"])
    for i, a in enumerate(dom_sorted):
        length = a["end"] - a["start"] + 1
        if length < 10:
            if a["start"] == 1:  # 起始于第 1 位时，用 start(len) 替换单独的首编号
                labels = [l for l in labels if l != [1.0, first_txt]]
                seen.discard((1.0, first_txt))
            _add(a["start"], f"{a['start'] + offset}({length})")
            continue
        _add(a["start"], str(a["start"] + offset))
        nxt = dom_sorted[i + 1] if i + 1 < len(dom_sorted) else None
        adjacent = nxt is not None and nxt["start"] == a["end"] + 1
        if not adjacent:
            _add(a["end"], str(a["end"] + offset))
    return labels


def assign_colors(annotations):
    """
    为 color 空缺的 DOMAIN 自动分配色盘颜色（原地修改，幂等）。
    只有 DOMAIN 使用彩色方块（REGION 括号统一黑色、POINT 方块统一红色），
    因此只有 DOMAIN 占用色盘；显式指定的颜色会被跳过，不与自动色撞色。
    在任何类型过滤 / 截短之前对完整标注列表调用一次，即可保证全长图与
    所有截短体图的颜色一致。
    """
    used_colors = {a["color"].upper() for a in annotations if a["color"]}
    avail = [c for c in DEFAULT_PALETTE
             if c.upper() not in used_colors] or list(DEFAULT_PALETTE)
    palette_idx = 0
    for anno in annotations:
        if anno["type"] == "DOMAIN" and not anno["color"]:
            anno["color"] = avail[palette_idx % len(avail)]
            palette_idx += 1


def clip_annotations(annotations, t_start, t_end):
    """
    将标注裁剪到截短体区间 [t_start, t_end]（全长蛋白坐标，包含边界），
    并转换为截短体局部坐标（1 起）。区间外的标注丢弃，DOMAIN / REGION
    截断为与区间的交集，POINT 仅保留落在区间内的位点。
    """
    clipped = []
    for a in annotations:
        if a["type"] == "POINT":
            if t_start <= a["start"] <= t_end:
                p = a["start"] - t_start + 1
                clipped.append({**a, "start": p, "end": p})
        else:
            s, e = max(a["start"], t_start), min(a["end"], t_end)
            if s <= e:
                clipped.append({**a, "start": s - t_start + 1,
                                "end": e - t_start + 1})
    return clipped


def plot_protein(protein_name, sequence, annotations, out_png,
                 include_types=None, boxes_only=False, label_mode="full",
                 residue_offset=0, simple=False):
    """
    include_types:   需要绘制的标注类型集合，如 ("REGION",) 只画 REGION；
                     None 表示绘制全部类型。
    boxes_only:      True 时只绘制链上的 DOMAIN / POINT 方块和 REGION 括号，
                     不绘制 DOMAIN 名称、POINT 线段与名称（用于简略版图）。
    label_mode:      数字标识模式，"full" 显示所有 DOMAIN 边界；
                     "default" 按 default 规则精简（见 _default_number_labels）。
    residue_offset:  绘图局部坐标 → 全长蛋白编号的偏移（截短体 = t_start - 1，
                     全长蛋白为 0）；仅影响显示的数字，不影响几何布局。
    simple:          True 时生成缩略图：省略所有文字（仅保留左侧蛋白名称），
                     不画数字标识与名称，POINT feature 完全隐藏；
                     横轴为分段对数压缩坐标：feature 段（DOMAIN）与间隔段按
                     在主链上的先后顺序依次排列，各段显示宽度 = ln(实际氨基酸数)，
                     间隔段再乘 SIMPLE_GAP_SCALE 进一步压缩（段内按实际位置
                     线性插值）；方块形状大小、灰线线宽与其余图片完全一致，
                     画布按同一每氨基酸英寸数公式计算，分辨率一致（dpi=300）。
    """
    # --- 颜色分配（幂等；在类型筛选之前，保证各图颜色一致）---
    assign_colors(annotations)

    if include_types is not None:
        keep = set(include_types)
        annotations = [a for a in annotations if a["type"] in keep]

    n = len(sequence)
    sc = SCALE                            # 整体放大倍数
    lw = LINE_WIDTH_PT * sc               # 灰色线线宽（pt）
    box_h = lw * BOX_HEIGHT_RATIO         # 方块高度（pt）（缩略图与其余图一致）
    fs_label = LABEL_FONTSIZE * sc        # 数字标识字号
    fs_name = ANNO_NAME_FONTSIZE * sc     # POINT 名称字号
    fs_dom = fs_name * 1.25               # DOMAIN 名称字号（放大 1.25 倍并加粗）
    fs_title = NAME_FONTSIZE * sc         # 蛋白质名称字号
    name_step = fs_dom * 1.7              # 名称层高（pt）

    # --- 类型拆分 ---
    domains, region_feats, point_feats = [], [], []
    for anno in annotations:
        {"DOMAIN": domains, "REGION": region_feats,
         "POINT": point_feats}[anno["type"]].append(anno)

    # --- x 坐标映射 xm：实际氨基酸坐标 → 绘图坐标 ---
    # 普通图：线性坐标（xm 为恒等映射）。
    # 缩略图：分段对数压缩映射 —— 主链拆成 feature 段（DOMAIN）与间隔段，
    #   按先后顺序依次排列；每段显示宽度 = ln(实际氨基酸数)，间隔段额外乘
    #   SIMPLE_GAP_SCALE；段内按实际位置线性插值，因此各 feature 与间隔的
    #   排列顺序、相对位置与实际蛋白完全一致，仅长度被对数压缩。
    if simple:
        covered = sorted((max(1, a["start"]), min(n, a["end"]) + 1)
                         for a in domains)  # DOMAIN 占据 [start, end+1)
        segments = []          # (real_start, real_end_exclusive, kind)
        cur = 1
        for s, e in covered:
            if s > cur:
                segments.append((cur, s, "gap"))      # feature 之间的间隔段
            if e > cur:
                segments.append((max(s, cur), e, "feature"))
                cur = e
        if cur < n + 1:
            segments.append((cur, n + 1, "gap"))      # 末端间隔段
        seg_map = []           # (real_start, real_end, disp_start, disp_end)
        simple_total_x = 0.0   # 缩略图主链显示总宽（对数单位）
        for s, e, kind in segments:
            seg_len = e - s
            w = math.log(seg_len) if seg_len > 1 else SIMPLE_MIN_SEG_LOG
            if kind == "gap":
                w *= SIMPLE_GAP_SCALE                 # 间隔区按比例额外压缩
            seg_map.append((s, e, simple_total_x, simple_total_x + w))
            simple_total_x += w

        def xm(pos):           # 段内线性插值的分段映射
            pos = min(max(float(pos), 1.0), float(n + 1))
            for s, e, ds, de in seg_map:
                if pos <= e:
                    return ds + (pos - s) / (e - s) * (de - ds)
            return seg_map[-1][3]
    else:
        xm = float

    # --- 画布：x 用数据坐标，y 用图形比例坐标 ---
    # 画布宽 = x 数据范围 × 每氨基酸英寸数（CANVAS_INCH_PER_AA），
    # 因此不论蛋白长度，300 dpi 下每个 x 单位的实际像素宽度恒定（≈20.7 px）；
    # 缩略图以“对数压缩后的显示总宽”为 x 范围按同一公式计算，比例尺与其余图一致
    if simple:
        margin_x = CANVAS_MARGIN_X_RATIO * simple_total_x
        fig_w = (simple_total_x + 2.0 * margin_x) * CANVAS_INCH_PER_AA
    else:
        margin_x = CANVAS_MARGIN_X_RATIO * n
        fig_w = ((n - 1) + 2.0 * margin_x) * CANVAS_INCH_PER_AA
    fig = plt.figure(figsize=(fig_w, 3.0 * sc))
    ax = fig.add_axes([0, 0, 1, 1])       # 坐标轴占满整图，垂直布局完全可控
    if simple:
        ax.set_xlim(-margin_x, simple_total_x + margin_x)
    else:
        ax.set_xlim(1 - margin_x, n + margin_x)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")
    trans = blended_transform_factory(ax.transData, ax.transAxes)

    # --- 预先分配各元素层级（只与 x 有关）---
    if simple:  # 缩略图：省略所有文字，无需计算文字层级
        dom_labels, dom_levels, pt_levels = [], [], []
    else:
        # POINT 竖线（位于方块中心）的 x 位置作为障碍物，DOMAIN 名称水平避让
        point_xs = [float(min(n, max(1, a["start"]))) + POINT_BAR_WIDTH_AA / 2.0
                    for a in point_feats]
        dodge_pad = 2.0 * sc / 72.0 * fig.dpi * _data_per_px(ax)

        dom_labels = [[(max(1, a["start"]) + min(n, a["end"])) / 2.0, a["name"]]
                      for a in domains]
        if dom_labels:
            dom_widths = _text_widths(ax, [t for _, t in dom_labels],
                                      fs_dom, em=0.62)  # 粗体略宽
            for lbl, w in zip(dom_labels, dom_widths):
                lbl[0] = _dodge_obstacles(lbl[0], w / 2.0, point_xs, dodge_pad)
            dom_levels = _assign_name_levels(dom_labels, ax, fs_dom)
        else:
            dom_levels = []

        # POINT 名称右对齐至竖线（方块中心），即名称右缘在线 x 处，中心在 x - 半宽
        if point_feats:
            pt_widths = _text_widths(ax, [a["name"] for a in point_feats],
                                     fs_name, em=0.58)
            pt_labels = [[float(min(n, max(1, a["start"])))
                          + POINT_BAR_WIDTH_AA / 2.0 - w / 2.0, a["name"]]
                         for a, w in zip(point_feats, pt_widths)]
            pt_levels = _assign_name_levels(pt_labels, ax, fs_name)
        else:
            pt_levels = []

    region_levels = _assign_interval_levels(region_feats) if region_feats else []
    # REGION 名称字号：根据区域长度与名称长度自动选择，所有 REGION 统一
    fs_region = _fit_region_fontsize(ax, region_feats, fs_name, 4.0 * sc)

    if boxes_only:  # 简略版：不画 DOMAIN 名称、POINT 线段与名称
        dom_labels, dom_levels = [], []
        pt_levels = []

    # --- 垂直布局（单位 pt，以灰色线中心为 0，最后统一平移）---
    # 主链上方：DOMAIN 名称 → POINT 箭头
    dom_name_y = box_h / 2.0 + lw * 0.8               # DOMAIN 名称第 0 层
    top = dom_name_y
    if dom_labels:
        top = dom_name_y + (max(dom_levels) + 1) * name_step + lw * 0.8

    arrow_len = lw * 3.0                              # POINT 箭头长度
    arrow_step = name_step + lw * 1.0
    pt_tail_y = top + arrow_len                       # POINT 线顶第 0 层
    if pt_levels:
        top = pt_tail_y + max(pt_levels) * arrow_step + fs_name * 1.3
    top_edge = top + 12.0 * sc                        # 顶部留白

    # 主链下方：数字标识 → REGION 括号（名称在括号下方）
    if simple:  # 缩略图：无数字标识，括号直接贴近主链，垂直方向更紧凑
        y_label = 0.0
        labels_bottom = -(box_h / 2.0 + lw * 0.8)
    else:
        y_label = -(box_h / 2.0 + lw * 1.2)               # 数字标识锚点（va=top）
        labels_bottom = y_label - fs_label * 1.3
    bottom = labels_bottom
    tick_len = lw * 1.5                               # REGION 括号竖 tick 长度
    bracket_step = tick_len + lw * 0.4 + fs_region * 1.4 + 2.0 * sc
    region_bar_0 = labels_bottom - lw * 0.8 - tick_len  # REGION 括号第 0 层横杆
    if region_feats:
        bottom = (region_bar_0 - max(region_levels) * bracket_step
                  - lw * 0.4 - fs_region * 1.3)
    bottom_edge = bottom - 12.0 * sc                  # 底部留白

    # 平移使最低点为 0，并换算为图形比例坐标
    y_shift = -bottom_edge
    total_pt = top_edge + y_shift
    fig.set_size_inches(fig_w, total_pt / 72.0)
    fig.canvas.draw()
    F = lambda pt: (pt + y_shift) / total_pt          # pt → 图形比例坐标

    # --- 灰色线：代表全部蛋白序列 ---
    ax.plot([xm(1), xm(n)], [F(0.0)] * 2, transform=trans,
            color=LINE_COLOR, linewidth=lw, solid_capstyle="butt", zorder=1)

    # 蛋白质名称：灰色线左端，Arial，距离灰色线约 4 倍余量（缩略图也保留）
    ax.text(xm(1) - margin_x * 0.9, F(0.0), protein_name, transform=trans,
            ha="right", va="center",
            fontsize=fs_title, fontfamily=FONT_NAME, zorder=3)

    # --- DOMAIN：彩色方块（始终绘制）+ 上方居中名称（boxes_only/simple 时不画）---
    for anno in domains:
        start, end = max(1, anno["start"]), min(n, anno["end"])
        ax.add_patch(Rectangle((xm(start), F(-box_h / 2.0)),
                               width=xm(end + 1) - xm(start),  # 包含边界
                               height=F(box_h / 2.0) - F(-box_h / 2.0),
                               transform=trans,
                               facecolor=anno["color"], edgecolor="none", zorder=2))
    for anno, lv, (name_x, _) in zip(domains, dom_levels, dom_labels):
        ax.text(name_x, F(dom_name_y + lv * name_step),
                anno["name"], transform=trans, ha="center", va="bottom",
                fontsize=fs_dom, fontfamily=FONT_NAME,
                fontweight="bold", zorder=3)

    # --- REGION：数字标识下方括号（开口朝上：横杆在下，两端 tick 朝上）
    #           + 括号下方居中名称（simple 时不画名称）---
    for anno, lv in zip(region_feats, region_levels):
        start, end = max(1, anno["start"]), min(n, anno["end"])
        y_bar = region_bar_0 - lv * bracket_step
        ax.plot([xm(start), xm(start), xm(end), xm(end)],
                [F(y_bar + tick_len), F(y_bar), F(y_bar), F(y_bar + tick_len)],
                transform=trans, color=REGION_BRACKET_COLOR,
                linewidth=1.5 * sc, solid_capstyle="butt", zorder=2)
        if not simple:
            ax.text((xm(start) + xm(end)) / 2.0, F(y_bar - lw * 0.4),
                    anno["name"], transform=trans, ha="center", va="top",
                    fontsize=fs_region, fontfamily=FONT_NAME, zorder=3)

    # --- POINT：红色单氨基酸方块 + 黑色竖线段与名称 ---
    # （缩略图 simple 完全隐藏 POINT；boxes_only 时不画线段和名称）
    if not simple:
        for anno in point_feats:
            x = min(n, max(1, anno["start"]))
            # 红色方块：恰好覆盖位点氨基酸本身，区间 [x, x+POINT_BAR_WIDTH_AA)
            # 宽度由 POINT_BAR_WIDTH_AA 控制（见脚本顶部参数，默认 1.0 = 1 个氨基酸）
            ax.add_patch(Rectangle((xm(x), F(-box_h / 2.0)),
                                   width=xm(x + POINT_BAR_WIDTH_AA) - xm(x),
                                   height=F(box_h / 2.0) - F(-box_h / 2.0),
                                   transform=trans,
                                   facecolor=POINT_BAR_COLOR, edgecolor="none",
                                   zorder=2))
    for anno, lv in zip(point_feats, pt_levels):
        x = xm(min(n, max(1, anno["start"])) + POINT_BAR_WIDTH_AA / 2.0)  # 竖线在方块中心
        y_tail = pt_tail_y + lv * arrow_step
        # 黑色竖线段（无箭头），与红色方块之间留空隙
        ax.plot([x, x], [F(box_h / 2.0 + lw * 0.6), F(y_tail)], transform=trans,
                color=POINT_SEG_COLOR, linewidth=1.5 * sc,
                solid_capstyle="butt", zorder=2)
        # 名称右对齐至位点
        ax.text(x, F(y_tail + lw * 0.3), anno["name"], transform=trans,
                ha="right", va="bottom",
                fontsize=fs_name, fontfamily=FONT_NAME, zorder=3)

    # --- 数字标识：灰色线两端 + DOMAIN 边界，统一水平对齐、加粗 ---
    # 显示数字 = 局部坐标 + residue_offset（截短体保留全长蛋白编号）
    # （缩略图省略所有文字，不画数字标识）
    if not simple:
        if label_mode == "default":
            labels = _default_number_labels(domains, n, residue_offset)
        else:  # full：显示所有 DOMAIN 边界
            first_txt, last_txt = str(1 + residue_offset), str(n + residue_offset)
            labels = [[1.0, first_txt], [float(n), last_txt]]
            seen = {(1.0, first_txt), (float(n), last_txt)}
            for anno in domains:
                for x in (anno["start"], anno["end"]):
                    key = (float(x), str(x + residue_offset))
                    if key not in seen:
                        seen.add(key)
                        labels.append([float(x), str(x + residue_offset)])

        new_xs = _spread_labels(labels, ax, fs_label)
        for (x, text), nx in zip(labels, new_xs):
            ax.text(nx, F(y_label), text, transform=trans, ha="center", va="top",
                    fontsize=fs_label, fontfamily=FONT_NAME,
                    fontweight="bold", zorder=3)

    fig.savefig(out_png, dpi=300, bbox_inches="tight",
                facecolor="white", pad_inches=0.15)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 合并输出文件
# ---------------------------------------------------------------------------
def write_output_file(protein_name, sequence, annotations, truncations,
                      hidden_lines, out_dir):
    """
    生成 {蛋白质名称}_annotation_{date}.txt（date 为绘制日期，如 20260928）。
    文件格式与输入文件完全统一，可直接作为输入二次绘制：
        第一行：蛋白质名称
        feature 区：每条 feature 一行，颜色按生成 figure 实际用色补全
            （DOMAIN 为其实际填充色——显式指定的保持原值、空缺时列出自动
            分配结果；REGION 括号统一为 REGION_BRACKET_COLOR；POINT 方块
            统一为 POINT_BAR_COLOR），以及 ">TRUNCATION, start, end" 截短体
            定义行（无截短体时写入 ">TRUNCATION, None, None" 占位行）
        ">features hided" 段：输入中不需要显示的 feature（原样保留）
        ">sequence" 段：全长蛋白质序列
        ">truncated sequence {start}-{end}" 段：每个截短体的序列
            （仅便于阅读；读取输入与绘图时忽略该段）
    """
    # 各 feature 在图中实际使用的颜色（annotations 已在 main 中完成颜色分配）
    actual_color = {"DOMAIN": lambda a: a["color"],
                    "REGION": lambda a: REGION_BRACKET_COLOR,
                    "POINT": lambda a: POINT_BAR_COLOR}

    date = datetime.now().strftime("%Y%m%d")
    out_path = os.path.join(out_dir, f"{protein_name}_annotation_{date}.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(protein_name + "\n")
        for a in annotations:
            f.write(f"{a['name']}, {a['start']}, {a['end']}, {a['type']}, "
                    f"{actual_color[a['type']](a)}\n")
        if truncations:
            for t_start, t_end in truncations:
                f.write(f">TRUNCATION, {t_start}, {t_end}\n")
        else:
            f.write(">TRUNCATION, None, None\n")   # 无截短体时的占位行
        if hidden_lines:
            f.write("\n>features hided\n")
            for ln in hidden_lines:
                f.write(ln + "\n")
        f.write("\n>sequence\n")
        f.write(sequence + "\n")
        for t_start, t_end in truncations:
            # 仅便于阅读的截短体序列段（读取输入与绘图时忽略）
            f.write(f"\n>truncated sequence {t_start}-{t_end}\n")
            f.write(sequence[t_start - 1:t_end] + "\n")
    return out_path


# ---------------------------------------------------------------------------
# 主函数
# ---------------------------------------------------------------------------
def main(input_file="annotator.txt"):
    # 唯一的输入文件：feature 区 + >features hided + >sequence
    protein_name, annotations, truncations, hidden_lines, sequence = \
        read_input(input_file)
    n_full = len(sequence)

    # 所有输出文件保存在 {蛋白质名称}_annotation/ 文件夹下
    out_dir = f"{protein_name}_annotation"
    os.makedirs(out_dir, exist_ok=True)

    # 颜色在任何截短 / 类型过滤之前统一分配，保证全长图与截短体图颜色一致
    assign_colors(annotations)

    # 三种输出模式：完整版 / default 数字标注版 / 简略版（REGION_only）
    modes = [
        # 完整版：所有 feature + full 数字标注
        ("full", dict(label_mode="full")),
        # default 版：所有 feature + default 数字标注
        ("default", dict(label_mode="default")),
        # 简略版：DOMAIN / POINT 方块 + REGION 括号，无名称与线段
        ("REGION_only", dict(boxes_only=True, label_mode="default")),
    ]

    # --- 全长蛋白：3 张图 + 缩略图 + 合并 txt ---
    for tag, kwargs in modes:
        out_png = os.path.join(out_dir, f"{protein_name}_annotation_{tag}.png")
        plot_protein(protein_name, sequence, annotations, out_png, **kwargs)
        print(f"图像已保存: {out_png}")
    # 缩略图：省略所有文字（保留蛋白名称），隐藏 POINT，横轴分段对数压缩
    out_png = os.path.join(out_dir, f"{protein_name}_annotation_simple.png")
    plot_protein(protein_name, sequence, annotations, out_png, simple=True)
    print(f"图像已保存: {out_png}")
    out_txt = write_output_file(protein_name, sequence, annotations,
                                truncations, hidden_lines, out_dir)
    print(f"蛋白质: {protein_name}  长度: {n_full} aa  标注数: {len(annotations)}"
          f"  隐藏 feature 数: {len(hidden_lines)}")
    print(f"数据文件已保存: {out_txt}")

    # --- 截短体：每个 TRUNCATION 按与全长蛋白相同的标准输出 3 张图 ---
    # （截短体序列已随全长序列一并写入上方的数据文件，不单独输出）
    for t_start, t_end in truncations:
        if not (1 <= t_start <= t_end <= n_full):
            raise ValueError(
                f"TRUNCATION 范围超出全长蛋白 (1-{n_full}): {t_start}-{t_end}")
        sub_seq = sequence[t_start - 1:t_end]             # 截短体序列
        sub_annos = clip_annotations(annotations, t_start, t_end)
        offset = t_start - 1        # 局部坐标 → 全长编号的偏移（保留原编号显示）
        base = f"{protein_name}_{t_start}-{t_end}"
        for tag, kwargs in modes:
            out_png = os.path.join(out_dir, f"{base}_annotation_{tag}.png")
            plot_protein(protein_name, sub_seq, sub_annos, out_png,
                         residue_offset=offset, **kwargs)
            print(f"图像已保存: {out_png}")
        print(f"截短体 {base}: {len(sub_seq)} aa")


if __name__ == "__main__":
    # 用法: python3 protein_annotation_plot.py [输入文件，默认 annotator.txt]
    main(sys.argv[1] if len(sys.argv) > 1 else "annotator.txt")
