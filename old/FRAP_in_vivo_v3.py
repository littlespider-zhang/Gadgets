"""
FRAP_in_vivo.py
FRAP Analysis Tool with Dynamic Droplet Tracking
"""

import os
import glob
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
from PIL import Image, ImageDraw, ImageTk
import csv
from datetime import datetime

# ── 可选依赖：CZI 支持 ──────────────────────────────────────
try:
    import czifile
    _CZI_AVAILABLE = True
except ImportError:
    _CZI_AVAILABLE = False

# tifffile 用于多页 TIFF 的可靠读取（Pillow 作为后备）
try:
    import tifffile
    _TIFF_AVAILABLE = True
except ImportError:
    _TIFF_AVAILABLE = False

# ── 可选依赖：matplotlib（结果图表面板）──────────────────────
try:
    import matplotlib
    matplotlib.use("TkAgg")
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_tkagg import (
        FigureCanvasTkAgg, NavigationToolbar2Tk)
    _MPL_AVAILABLE = True
except ImportError:
    _MPL_AVAILABLE = False


# ─────────────────────────────────────────────
#  Part 0: Multi-format image I/O helpers
# ─────────────────────────────────────────────

# 所有支持的扩展名（小写）
SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".czi"}


def _ext(filepath: str) -> str:
    """返回小写扩展名，含点号。"""
    return os.path.splitext(filepath)[1].lower()


def _load_czi_as_rgb(filepath: str) -> Image.Image:
    """
    读取 CZI 文件并返回 RGB PIL Image。
    取第一个时间点、第一个 Z 层、所有通道做最大投影或单通道转灰度。
    """
    if not _CZI_AVAILABLE:
        raise ImportError(
            "读取 .czi 文件需要安装 czifile：pip install czifile")
    arr = czifile.imread(filepath)  # 典型维度：T Z C Y X (S)
    # 压缩所有大小为 1 的轴
    arr = np.squeeze(arr)
    # 归一化到 uint8
    arr = arr.astype(np.float32)
    arr -= arr.min()
    if arr.max() > 0:
        arr /= arr.max()
    arr = (arr * 255).astype(np.uint8)
    # 处理维度
    if arr.ndim == 2:                        # 单通道灰度
        return Image.fromarray(arr, "L").convert("RGB")
    if arr.ndim == 3:
        if arr.shape[0] <= 4:               # (C, H, W) → 通道在前
            arr = np.max(arr, axis=0)       # 最大投影
            return Image.fromarray(arr, "L").convert("RGB")
        else:                               # (H, W, C)
            if arr.shape[2] == 1:
                return Image.fromarray(arr[:, :, 0], "L").convert("RGB")
            return Image.fromarray(arr[:, :, :3], "RGB")
    # 4D 及以上：取第一帧第一 Z 层
    if arr.ndim >= 4:
        frame = arr.reshape(-1, arr.shape[-2], arr.shape[-1])[0]
        return Image.fromarray(frame, "L").convert("RGB")
    raise ValueError(f"无法解析 CZI 数组维度：{arr.shape}")


def _load_tif_as_rgb(filepath: str) -> Image.Image:
    """读取 TIF/TIFF（含多页/16-bit），返回 RGB PIL Image。"""
    if _TIFF_AVAILABLE:
        arr = tifffile.imread(filepath)
        arr = np.squeeze(arr)
        # 取第一帧（若为时间序列）
        if arr.ndim > 3:
            arr = arr[0]
        arr = arr.astype(np.float32)
        arr -= arr.min()
        if arr.max() > 0:
            arr /= arr.max()
        arr = (arr * 255).astype(np.uint8)
        if arr.ndim == 2:
            return Image.fromarray(arr, "L").convert("RGB")
        if arr.ndim == 3:
            if arr.shape[0] <= 4:            # (C, H, W)
                arr = np.max(arr, axis=0)
                return Image.fromarray(arr, "L").convert("RGB")
            return Image.fromarray(arr[:, :, :3])
    # 后备：直接用 Pillow
    img = Image.open(filepath)
    img.load()
    return img.convert("RGB")


def open_as_rgb(filepath: str) -> Image.Image:
    """
    统一入口：根据扩展名选择读取方式，始终返回 RGB PIL Image。
    支持 .jpg/.jpeg / .png / .tif/.tiff / .czi
    """
    ext = _ext(filepath)
    if ext == ".czi":
        return _load_czi_as_rgb(filepath)
    if ext in (".tif", ".tiff"):
        return _load_tif_as_rgb(filepath)
    # PNG / JPEG / 其他 Pillow 支持的格式
    return Image.open(filepath).convert("RGB")




# ─────────────────────────────────────────────
#  Part 0b: CZI Stack → JPEG extraction
# ─────────────────────────────────────────────

def _normalize_to_uint8(arr: np.ndarray) -> np.ndarray:
    """将任意数值类型的数组线性归一化到 uint8 [0, 255]。"""
    a = arr.astype(np.float32)
    lo, hi = a.min(), a.max()
    if hi > lo:
        a = (a - lo) / (hi - lo) * 255.0
    else:
        a = np.zeros_like(a)
    return a.astype(np.uint8)


def _czi_extract_frames(arr: np.ndarray) -> list:
    """
    将 CZI imread（squeeze=True）结果拆分为单帧灰度 uint8 列表。

    ★ 全栈统一拉伸：用整个 stack 的全局 min/max 做一次线性映射到 0-255，
      所有帧共用同一个拉伸系数。这样帧间真实强度变化得以保留，
      不会因为逐帧独立 min-max 把整帧灰度系统性抖动注入到数据里。
      （FRAP 分析的关键前提就是各帧强度可比。）

    CZI 标准维度顺序（squeeze 后）：...T Z C Y X
    最后两轴始终视为 (H, W)。
    其余轴中大小 <= 4 的视为通道轴（C），做最大投影；
    其余轴（T、Z）全部展开为独立帧。

    特殊处理：若 3D 且最后轴 <= 4 且远小于前两轴，
    则判定为 PIL 风格 (H, W, C)，做最后轴投影返回 1 帧。

    支持的典型情形（squeeze 后）：
      2D  (H, W)               → 1 帧
      3D  (T, H, W)            → T 帧（FRAP 最常见）
      3D  (C, H, W)  C ≤ 4    → 1 帧（多通道最大投影）
      3D  (H, W, C)  C ≤ 4    → 1 帧（PIL 风格）
      4D  (T, C, H, W)         → T 帧
      4D  (T, Z, H, W)  Z > 4  → T×Z 帧
      5D  (T, Z, C, H, W)      → T×Z 帧（C 轴投影）
    """
    arr = np.squeeze(arr)
    ndim = arr.ndim

    # ── 1. 先把数据整理成 (n_frames, H, W) 的浮点 stack ──
    if ndim == 2:
        stack = arr[np.newaxis, :, :]                    # (1, H, W)

    elif ndim == 3:
        s0, s1, s2 = arr.shape
        # 3D PIL 风格 (H, W, C)，最后轴很小且远小于前两轴 → 1 帧
        if s2 <= 4 and s2 < s0 and s2 < s1:
            stack = arr.max(axis=2)[np.newaxis, :, :]    # (1, H, W)
        else:
            # 通用：最后两轴 = (H, W)，前面是帧/通道轴
            prefix = arr.shape[:-2]
            channel_axes = [i for i, s in enumerate(prefix) if s <= 4]
            reduced = arr
            for ax in sorted(channel_axes, reverse=True):
                reduced = reduced.max(axis=ax)
            # reduced 形状现在是 (?, H, W) 或 (H, W)
            if reduced.ndim == 2:
                stack = reduced[np.newaxis, :, :]
            else:
                n_frames = int(np.prod(reduced.shape[:-2]))
                stack = reduced.reshape(n_frames, reduced.shape[-2],
                                        reduced.shape[-1])
    else:
        # 4D+：最后两轴 (H, W)，前面按大小拆 channel / frame 轴
        prefix = arr.shape[:-2]
        channel_axes = [i for i, s in enumerate(prefix) if s <= 4]
        reduced = arr
        for ax in sorted(channel_axes, reverse=True):
            reduced = reduced.max(axis=ax)
        if reduced.ndim == 2:
            stack = reduced[np.newaxis, :, :]
        else:
            n_frames = int(np.prod(reduced.shape[:-2]))
            stack = reduced.reshape(n_frames, reduced.shape[-2],
                                    reduced.shape[-1])

    # ── 2. 全栈统一拉伸到 uint8 ─────────────────────────────
    stack_f = stack.astype(np.float32)
    g_lo = float(stack_f.min())
    g_hi = float(stack_f.max())
    if g_hi > g_lo:
        scale = 255.0 / (g_hi - g_lo)
        stack_u8 = np.clip((stack_f - g_lo) * scale, 0, 255).astype(np.uint8)
    else:
        stack_u8 = np.zeros_like(stack_f, dtype=np.uint8)

    return [stack_u8[i] for i in range(stack_u8.shape[0])]


def extract_czi_stack(
        czi_path: str,
        out_dir: str,
        prefix: str = "frame",
        jpeg_quality: int = 95,
        progress_cb=None) -> list:
    """
    将单个 CZI stack 文件解压为逐帧 JPEG。

    参数
    ----
    czi_path      : CZI 文件路径
    out_dir       : 输出目录（自动创建）
    prefix        : 输出文件名前缀，默认 "frame"
    jpeg_quality  : JPEG 质量 1-95，默认 95
    progress_cb   : 可选回调 progress_cb(i, total, filename)

    返回
    ----
    list[str]：所有已保存 JPEG 的完整路径，按帧顺序排列
    """
    if not _CZI_AVAILABLE:
        raise ImportError("请先安装 czifile：pip install czifile")

    arr = czifile.imread(czi_path, squeeze=True)
    frames = _czi_extract_frames(arr)

    os.makedirs(out_dir, exist_ok=True)
    total = len(frames)
    saved = []
    pad = len(str(total))   # 零填充位数，保证文件名排序正确

    for i, frame_arr in enumerate(frames):
        fname = f"{prefix}_{str(i + 1).zfill(pad)}.jpg"
        fpath = os.path.join(out_dir, fname)
        img = Image.fromarray(frame_arr, mode="L").convert("RGB")
        img.save(fpath, "JPEG", quality=jpeg_quality)
        saved.append(fpath)
        if progress_cb:
            progress_cb(i + 1, total, fname)

    return saved

# ─────────────────────────────────────────────
#  Part 1: Core Functions
# ─────────────────────────────────────────────

def load_image_as_gray(filepath: str) -> np.ndarray:
    """格式转换：将图片（jpg/png/tif/czi）转为灰度 numpy 数组。"""
    img = open_as_rgb(filepath).convert("L")
    return np.array(img, dtype=np.float64)


def get_roi(gray: np.ndarray, cx1: int, cy1: int, cx2: int, cy2: int) -> np.ndarray:
    """ROI 裁切：返回 ROI 区域的灰度数组（行=y，列=x）。"""
    return gray[cy1:cy2, cx1:cx2]


# 性能优化：按形状缓存 np.mgrid 网格，避免每帧重复构建。
# 这些网格只读不改，可安全跨帧共享；不改变任何计算结果。
_MGRID_CACHE = {}


def _cached_mgrid(h: int, w: int):
    """返回 (ys, xs) = np.mgrid[0:h, 0:w]，按 (h, w) 缓存复用。"""
    key = (h, w)
    grid = _MGRID_CACHE.get(key)
    if grid is None:
        grid = np.mgrid[0:h, 0:w]
        _MGRID_CACHE[key] = grid
    return grid


def calc_centroid(roi: np.ndarray):
    """
    Droplet 质心计算（亮度加权重心）。
    返回 (droplet_center_x, droplet_center_y)，坐标参考系为 ROI 区域。
    """
    total = roi.sum()
    if total == 0:
        h, w = roi.shape
        return w / 2.0, h / 2.0
    # 与原实现完全等价：网格按形状缓存复用，算式不变。
    ys, xs = _cached_mgrid(roi.shape[0], roi.shape[1])
    cx = (xs * roi).sum() / total
    cy = (ys * roi).sum() / total
    return float(cx), float(cy)


def calc_offset(frap_x: float, frap_y: float,
                center_x: float, center_y: float):
    """
    偏移量计算：FRAP 圆心相对于 droplet 质心的偏移。
    offset = FRAP_pos - centroid_pos（均在 ROI 坐标系下）
    """
    return frap_x - center_x, frap_y - center_y


def _circle_intensity(gray: np.ndarray,
                      fx_global: float, fy_global: float,
                      frap_radius: float) -> tuple:
    """
    亚像素加权圆内强度（抗锯齿圆）。
    边界像素按其落在圆内的覆盖比例（0~1）加权，
    圆心亚像素移动时权重平滑变化，避免整像素纳入/排除导致的锯齿。

    返回 (intensity, weight_sum)。
    """
    h, w = gray.shape
    pad = 1  # 多留 1px 给边界过渡带
    x0 = max(0, int(np.floor(fx_global - frap_radius - pad)))
    x1 = min(w, int(np.ceil(fx_global + frap_radius + pad)) + 1)
    y0 = max(0, int(np.floor(fy_global - frap_radius - pad)))
    y1 = min(h, int(np.ceil(fy_global + frap_radius + pad)) + 1)

    if x1 <= x0 or y1 <= y0:
        return 0.0, 0.0

    sub = gray[y0:y1, x0:x1]
    ys, xs = np.mgrid[y0:y1, x0:x1]
    dist = np.sqrt((xs - fx_global) ** 2 + (ys - fy_global) ** 2)
    weight = np.clip(frap_radius + 0.5 - dist, 0.0, 1.0)
    weight_sum = float(weight.sum())
    if weight_sum > 0:
        intensity = float((sub * weight).sum() / weight_sum)
    else:
        intensity = 0.0
    return intensity, weight_sum


def calc_frap_intensity(gray: np.ndarray,
                        cx1: int, cy1: int, cx2: int, cy2: int,
                        offset_x: float, offset_y: float,
                        frap_radius: float) -> dict:
    """
    荧光强度计算：
      1. 计算 ROI 内的质心
      2. 加上偏移量得到本帧的 FRAP 圆心（ROI 坐标系）
      3. 转为全图坐标后，在 frap_radius 范围内做「亚像素加权均值」
         （抗锯齿圆：边界像素按覆盖比例加权，消除 pixel_count 跳动引起的锯齿）
    返回包含所有中间量的字典。
    """
    roi = get_roi(gray, cx1, cy1, cx2, cy2)
    dcx, dcy = calc_centroid(roi)

    # FRAP 圆心（ROI 坐标系）
    fx_roi = dcx + offset_x
    fy_roi = dcy + offset_y

    # 转换为全图坐标
    fx_global = fx_roi + cx1
    fy_global = fy_roi + cy1

    intensity, weight_sum = _circle_intensity(gray, fx_global, fy_global,
                                              frap_radius)

    return {
        "droplet_center_x": dcx,
        "droplet_center_y": dcy,
        "frap_x_roi": fx_roi,
        "frap_y_roi": fy_roi,
        "frap_x_global": fx_global,
        "frap_y_global": fy_global,
        "FRAP_intensity": intensity,
        # pixel_count 改为「有效加权面积」（≈ π·r²，逐帧近似恒定，可作平滑性自检）
        "pixel_count": weight_sum,
    }


def calc_bg_intensity(gray: np.ndarray,
                      bg_x1: int, bg_y1: int,
                      frap_x_roi: float, frap_y_roi: float,
                      frap_radius: float) -> dict:
    """
    背景区 FRAP 圆强度（位置固定，不做质心追踪）。
      背景圆心（全图坐标）= 背景 ROI 左上角 + 主 ROI 内的 FRAP 圆心相对坐标
      用与主 FRAP 圆相同的亚像素加权方式计算均值。

    返回 dict 含 bg_frap_x_global / bg_frap_y_global / FRAP_bg_intensity /
                  bg_pixel_count
    """
    fx_global = bg_x1 + frap_x_roi
    fy_global = bg_y1 + frap_y_roi
    intensity, weight_sum = _circle_intensity(gray, fx_global, fy_global,
                                              frap_radius)
    return {
        "bg_frap_x_global": fx_global,
        "bg_frap_y_global": fy_global,
        "FRAP_bg_intensity": intensity,
        "bg_pixel_count": weight_sum,
    }


def normalize_intensities(results: list,
                          baseline_frames: int = 2,
                          bleach_frame: int = None,
                          has_bg: bool = False) -> dict:
    """
    FRAP 双点归一化（two-point normalization）。

    以前 baseline_frames 帧的 FRAP_intensity 均值作为 100%（pre-bleach baseline），
    以 bleach_frame（漂白后第一帧）的 FRAP_intensity 作为 0%，
    对所有帧做线性归一化：

        norm = (I - I_bleach) / (baseline - I_bleach) * 100

    使得 baseline → 100%，bleach 帧 → 0%，恢复曲线自 0% 回升。
    结果原地写入每个 row 的 "FRAP_intensity_norm"。

    若 has_bg=True 且每个 row 含有 "FRAP_bg_intensity"，则同步生成两组校正：
      · 减法（加性噪声）：FRAP_intensity_corrected = FRAP - bg，
                        归一化列 FRAP_intensity_corrected_norm
      · 除法（乘性噪声）：FRAP_intensity_ratio    = FRAP / bg，
                        归一化列 FRAP_intensity_ratio_norm
    乘性噪声（激光功率抖动、相机增益、JPEG 逐帧拉伸）减法压不掉，须用除法。

    参数
    ----
    results         : batch_analysis 返回的逐帧字典列表
    baseline_frames : 前多少帧的均值作为 baseline（100%），默认 2
    bleach_frame    : 漂白后第一帧的帧号（1-based，即第 1 帧记为 1）。
                      None 或 <=0 时退化为旧式单点归一化
                      （baseline → 100%，I_bleach 取 0）。
    has_bg          : 是否同步处理扣背景列。

    返回
    ----
    dict 包含：
        baseline, i_bleach                    ：主 FRAP 通道
        baseline_corr, i_bleach_corr          ：减法扣背景通道
        baseline_ratio, i_bleach_ratio        ：除法比值通道
    """
    out = {"baseline": 0.0, "i_bleach": 0.0,
           "baseline_corr": 0.0, "i_bleach_corr": 0.0,
           "baseline_ratio": 0.0, "i_bleach_ratio": 0.0}
    if not results:
        return out

    n = max(1, min(int(baseline_frames), len(results)))

    # ── 主 FRAP 通道 ─────────────────────────────────────────
    base_vals = [r["FRAP_intensity"] for r in results[:n]]
    baseline = sum(base_vals) / len(base_vals) if base_vals else 0.0

    if bleach_frame is not None and bleach_frame >= 1:
        idx = min(int(bleach_frame) - 1, len(results) - 1)
        i_bleach = results[idx]["FRAP_intensity"]
    else:
        i_bleach = 0.0

    denom = baseline - i_bleach
    for r in results:
        if denom != 0:
            r["FRAP_intensity_norm"] = (r["FRAP_intensity"] - i_bleach) / denom * 100.0
        else:
            r["FRAP_intensity_norm"] = 0.0

    out["baseline"] = baseline
    out["i_bleach"] = i_bleach

    # ── 扣背景通道（减法：消除加性噪声）─────────────────────
    if has_bg:
        # 先算每帧的 corrected = FRAP - bg
        for r in results:
            r["FRAP_intensity_corrected"] = (r.get("FRAP_intensity", 0.0)
                                             - r.get("FRAP_bg_intensity", 0.0))

        base_vals_c = [r["FRAP_intensity_corrected"] for r in results[:n]]
        baseline_c = sum(base_vals_c) / len(base_vals_c) if base_vals_c else 0.0

        if bleach_frame is not None and bleach_frame >= 1:
            idx = min(int(bleach_frame) - 1, len(results) - 1)
            i_bleach_c = results[idx]["FRAP_intensity_corrected"]
        else:
            i_bleach_c = 0.0

        denom_c = baseline_c - i_bleach_c
        for r in results:
            if denom_c != 0:
                r["FRAP_intensity_corrected_norm"] = (
                    (r["FRAP_intensity_corrected"] - i_bleach_c) / denom_c * 100.0)
            else:
                r["FRAP_intensity_corrected_norm"] = 0.0

        out["baseline_corr"] = baseline_c
        out["i_bleach_corr"] = i_bleach_c

        # ── 比值通道（除法：消除乘性噪声，如激光抖动/增益变化/逐帧拉伸）──
        # 每帧 FRAP_ratio = FRAP_intensity / FRAP_bg_intensity，再对该比值序列
        # 做相同的双点归一化（baseline → 100%，bleach → 0%）。
        for r in results:
            bg_v = r.get("FRAP_bg_intensity", 0.0)
            if bg_v != 0:
                r["FRAP_intensity_ratio"] = r.get("FRAP_intensity", 0.0) / bg_v
            else:
                r["FRAP_intensity_ratio"] = 0.0

        base_vals_r = [r["FRAP_intensity_ratio"] for r in results[:n]]
        baseline_r = sum(base_vals_r) / len(base_vals_r) if base_vals_r else 0.0

        if bleach_frame is not None and bleach_frame >= 1:
            idx = min(int(bleach_frame) - 1, len(results) - 1)
            i_bleach_r = results[idx]["FRAP_intensity_ratio"]
        else:
            i_bleach_r = 0.0

        denom_r = baseline_r - i_bleach_r
        for r in results:
            if denom_r != 0:
                r["FRAP_intensity_ratio_norm"] = (
                    (r["FRAP_intensity_ratio"] - i_bleach_r) / denom_r * 100.0)
            else:
                r["FRAP_intensity_ratio_norm"] = 0.0

        out["baseline_ratio"] = baseline_r
        out["i_bleach_ratio"] = i_bleach_r

    return out


def load_frap_results(csv_path: str) -> dict:
    """
    读取 FRAP_results.csv，返回各列的 numpy 数组（供绘图使用）。
    若 CSV 缺少 FRAP_intensity_norm 列（旧版结果），则自动用前两帧
    FRAP_intensity 均值为 100% 重新计算归一化（单点，作为兜底）。
    若存在 FRAP_intensity_corrected_norm 列，一并读入。
    """
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError("CSV 文件为空或格式不正确。")

    def col(name, cast=float):
        out = []
        for r in rows:
            v = r.get(name, "")
            try:
                out.append(cast(v))
            except (TypeError, ValueError):
                out.append(float("nan"))
        return np.array(out, dtype=float)

    frame_index = col("frame_index", lambda x: int(float(x)))
    intensity = col("FRAP_intensity")
    cx = col("droplet_center_x")
    cy = col("droplet_center_y")

    # 归一化列：优先用 CSV 里已有的，否则重新计算
    has_norm = ("FRAP_intensity_norm" in rows[0]
                and rows[0].get("FRAP_intensity_norm", "") != "")
    if has_norm:
        norm = col("FRAP_intensity_norm")
    else:
        valid = intensity[:2]
        valid = valid[~np.isnan(valid)]
        baseline = float(valid.mean()) if valid.size > 0 else 0.0
        norm = (intensity / baseline * 100.0) if baseline > 0 else np.zeros_like(intensity)

    # 扣背景列（可选）
    has_corr = ("FRAP_intensity_corrected_norm" in rows[0]
                and rows[0].get("FRAP_intensity_corrected_norm", "") != "")
    corr_norm = col("FRAP_intensity_corrected_norm") if has_corr else None

    has_ratio = ("FRAP_intensity_ratio_norm" in rows[0]
                 and rows[0].get("FRAP_intensity_ratio_norm", "") != "")
    ratio_norm = col("FRAP_intensity_ratio_norm") if has_ratio else None

    return {
        "frame_index": frame_index,
        "FRAP_intensity": intensity,
        "FRAP_intensity_norm": norm,
        "FRAP_intensity_corrected_norm": corr_norm,   # None 或 np.ndarray
        "FRAP_intensity_ratio_norm": ratio_norm,      # None 或 np.ndarray
        "droplet_center_x": cx,
        "droplet_center_y": cy,
    }


def annotate_frame(filepath: str,
                   cx1: int, cy1: int, cx2: int, cy2: int,
                   frap_global_x: float, frap_global_y: float,
                   frap_radius: float,
                   dcx_roi: float, dcy_roi: float,
                   bg_box: tuple = None,
                   bg_frap_global: tuple = None) -> Image.Image:
    """
    在原始彩色图片上绘制标注并返回 PIL Image：
      · ROI         : 黄色虚线矩形
      · Droplet 质心: 黄色十字（ROI 坐标 → 全图坐标）
      · FRAP 圆     : 蓝色实线圆 + 圆心点
      · 背景 ROI    : 绿色虚线矩形（若提供 bg_box）
      · 背景 FRAP 圆: 绿色实线圆（若提供 bg_frap_global）

    bg_box           : (bx1, by1, bx2, by2) 全图坐标
    bg_frap_global   : (fx, fy) 背景区 FRAP 圆全图坐标
    """
    img = open_as_rgb(filepath)
    draw = ImageDraw.Draw(img)

    # ROI 黄色虚线矩形
    dash = 10
    for x in range(cx1, cx2, dash * 2):
        draw.line([(x, cy1), (min(x + dash, cx2), cy1)], fill="#f0e040", width=2)
        draw.line([(x, cy2), (min(x + dash, cx2), cy2)], fill="#f0e040", width=2)
    for y in range(cy1, cy2, dash * 2):
        draw.line([(cx1, y), (cx1, min(y + dash, cy2))], fill="#f0e040", width=2)
        draw.line([(cx2, y), (cx2, min(y + dash, cy2))], fill="#f0e040", width=2)

    # Droplet 质心：黄色十字（ROI → 全图坐标）
    gx = dcx_roi + cx1
    gy = dcy_roi + cy1
    arm = 10
    draw.line([(gx - arm, gy), (gx + arm, gy)], fill="#f0e040", width=1)
    draw.line([(gx, gy - arm), (gx, gy + arm)], fill="#f0e040", width=1)

    # FRAP 圆：蓝色实线
    fx, fy, r = frap_global_x, frap_global_y, frap_radius
    draw.ellipse([(fx - r, fy - r), (fx + r, fy + r)],
                 outline="#4fc3f7", width=2)
    draw.ellipse([(fx - 2, fy - 2), (fx + 2, fy + 2)], fill="#4fc3f7")

    # 背景 ROI：绿色虚线矩形
    if bg_box is not None:
        bx1, by1, bx2, by2 = bg_box
        for x in range(bx1, bx2, dash * 2):
            draw.line([(x, by1), (min(x + dash, bx2), by1)],
                      fill="#2ecc71", width=2)
            draw.line([(x, by2), (min(x + dash, bx2), by2)],
                      fill="#2ecc71", width=2)
        for y in range(by1, by2, dash * 2):
            draw.line([(bx1, y), (bx1, min(y + dash, by2))],
                      fill="#2ecc71", width=2)
            draw.line([(bx2, y), (bx2, min(y + dash, by2))],
                      fill="#2ecc71", width=2)

    # 背景 FRAP 圆：绿色实线
    if bg_frap_global is not None:
        bfx, bfy = bg_frap_global
        draw.ellipse([(bfx - r, bfy - r), (bfx + r, bfy + r)],
                     outline="#2ecc71", width=2)
        draw.ellipse([(bfx - 2, bfy - 2), (bfx + 2, bfy + 2)], fill="#2ecc71")

    return img


def batch_analysis(folder: str,
                   cx1: int, cy1: int, cx2: int, cy2: int,
                   offset_x: float, offset_y: float,
                   frap_radius: float,
                   marked_dir: str = None,
                   progress_cb=None,
                   frame_cb=None,
                   bg_corner1: tuple = None) -> list:
    """
    批量数据分析：对文件夹内所有支持格式（jpg/png/tif/czi）按文件名排序后逐帧计算。
    marked_dir 不为 None 时，将标注后的图片（ROI虚线框、质心、FRAP圆）保存到该目录。
    progress_cb(i, total, filename) 可用于更新进度。
    frame_cb(row) 每帧完成后立即回调。

    bg_corner1 : (bg_x1, bg_y1) 背景 ROI 左上角全图坐标；为 None 时不计算背景。
                 背景 ROI 尺寸与主 ROI 一致 (cx2-cx1, cy2-cy1)。
                 背景 FRAP 圆位置固定（不做质心追踪），等于主 ROI 中
                 frap_x_roi/frap_y_roi 平移到背景 ROI 起点。

    返回列表，每项为字典（含 filename + calc_frap_intensity 返回的字段；
    若启用背景，还含 FRAP_bg_intensity 等）。
    """
    patterns = ["*.jpg", "*.jpeg", "*.JPG", "*.JPEG",
                "*.png", "*.PNG",
                "*.tif", "*.tiff", "*.TIF", "*.TIFF",
                "*.czi", "*.CZI"]
    files = []
    for p in patterns:
        files.extend(glob.glob(os.path.join(folder, p)))
    files = sorted(set(files))

    if marked_dir:
        os.makedirs(marked_dir, exist_ok=True)

    roi_w = cx2 - cx1
    roi_h = cy2 - cy1
    bg_box = None
    if bg_corner1 is not None:
        bg_x1, bg_y1 = bg_corner1
        bg_box = (bg_x1, bg_y1, bg_x1 + roi_w, bg_y1 + roi_h)

    results = []
    total = len(files)
    for i, fp in enumerate(files):
        gray = load_image_as_gray(fp)
        row = calc_frap_intensity(gray, cx1, cy1, cx2, cy2,
                                  offset_x, offset_y, frap_radius)
        row["filename"] = os.path.basename(fp)
        row["frame_index"] = i

        # 背景计算（若启用）
        bg_frap_xy_global = None
        if bg_corner1 is not None:
            bg_row = calc_bg_intensity(gray, bg_corner1[0], bg_corner1[1],
                                       row["frap_x_roi"], row["frap_y_roi"],
                                       frap_radius)
            row.update(bg_row)
            bg_frap_xy_global = (bg_row["bg_frap_x_global"],
                                 bg_row["bg_frap_y_global"])

        results.append(row)

        # 保存标注图片
        if marked_dir:
            marked_img = annotate_frame(
                fp, cx1, cy1, cx2, cy2,
                row["frap_x_global"], row["frap_y_global"],
                frap_radius,
                row["droplet_center_x"], row["droplet_center_y"],
                bg_box=bg_box,
                bg_frap_global=bg_frap_xy_global,
            )
            base, _ = os.path.splitext(os.path.basename(fp))
            marked_img.save(os.path.join(marked_dir, f"{base}_marked.jpg"),
                            "JPEG", quality=95)

        if frame_cb:
            frame_cb(row)
        if progress_cb:
            progress_cb(i + 1, total, os.path.basename(fp))
    return results

def _mean_filter3(a: np.ndarray) -> np.ndarray:
    """3×3 均值滤波（edge padding），用于检测前轻度降噪。"""
    p = np.pad(a, 1, mode="edge")
    s = (p[0:-2, 0:-2] + p[0:-2, 1:-1] + p[0:-2, 2:] +
         p[1:-1, 0:-2] + p[1:-1, 1:-1] + p[1:-1, 2:] +
         p[2:,   0:-2] + p[2:,   1:-1] + p[2:,   2:])
    return s / 9.0


def _otsu_threshold(values: np.ndarray) -> float:
    """
    Otsu 自动阈值（纯 numpy 实现）。
    返回的阈值用于区分暗区（<= t，FRAP 淬灭）与亮区（> t，液滴背景）。
    """
    v = values.ravel().astype(np.float64)
    vmin, vmax = float(v.min()), float(v.max())
    if vmax <= vmin:
        return vmin
    hist, edges = np.histogram(v, bins=256, range=(vmin, vmax))
    total = v.size
    prob = hist / total
    omega = np.cumsum(prob)                         # 累积前景占比
    bin_centers = (edges[:-1] + edges[1:]) / 2.0
    mu = np.cumsum(prob * bin_centers)              # 累积均值
    mu_t = mu[-1]
    denom = omega * (1.0 - omega)
    denom[denom == 0] = 1e-12
    sigma_b2 = (mu_t * omega - mu) ** 2 / denom     # 类间方差
    idx = int(np.argmax(sigma_b2))
    return float(bin_centers[idx])


def _label_components(mask: np.ndarray):
    """
    8-连通域标记（纯 Python 栈式 flood-fill，针对小区域足够快）。
    返回 (labels, n_components)，labels 为 int32 数组，0 表示背景。
    """
    labels = np.zeros(mask.shape, dtype=np.int32)
    h, w = mask.shape
    current = 0
    coords = np.argwhere(mask)
    for (sy, sx) in coords:
        if labels[sy, sx] != 0:
            continue
        current += 1
        stack = [(int(sy), int(sx))]
        labels[sy, sx] = current
        while stack:
            cy, cx = stack.pop()
            y0, y1 = max(0, cy - 1), min(h, cy + 2)
            x0, x1 = max(0, cx - 1), min(w, cx + 2)
            for ny in range(y0, y1):
                for nx in range(x0, x1):
                    if mask[ny, nx] and labels[ny, nx] == 0:
                        labels[ny, nx] = current
                        stack.append((ny, nx))
    return labels, current


def detect_frap_circle(roi_gray: np.ndarray,
                       bx1: float, by1: float, bx2: float, by2: float,
                       min_area: int = 4) -> dict:
    """
    自动检测 FRAP 淬灭圆。

    思路：FRAP 淬灭区 = “强度低（暗）” + “近似圆形”。
      1. 在 ROI 灰度图的大致范围 (bx1,by1)-(bx2,by2) 内取子图；
      2. 3×3 均值降噪后用 Otsu 阈值分出暗区（淬灭）/亮区（液滴）；
      3. 连通域标记，选「离框中心最近且面积达标」的暗块（避开液滴边缘等干扰）；
      4. 质心即圆心；半径取「等面积圆半径」与「二阶矩半径」的均值，更稳健。

    参数 / 范围坐标均为 ROI 坐标系。
    返回 dict: frap_x_roi, frap_y_roi, radius, area, threshold, n_components
    """
    h_roi, w_roi = roi_gray.shape
    bx1 = max(0, int(round(bx1))); by1 = max(0, int(round(by1)))
    bx2 = min(w_roi, int(round(bx2))); by2 = min(h_roi, int(round(by2)))
    if bx2 - bx1 < 2 or by2 - by1 < 2:
        raise ValueError("框选的大致范围太小，请重新框选。")

    sub = roi_gray[by1:by2, bx1:bx2].astype(np.float64)
    smooth = _mean_filter3(sub)

    # Otsu 分割：暗区（淬灭）为 <= t
    t = _otsu_threshold(smooth)
    dark_mask = smooth <= t
    if not dark_mask.any():
        raise ValueError("未检测到暗区，请调整框选范围。")

    labels, n = _label_components(dark_mask)
    sub_h, sub_w = sub.shape
    cx_box, cy_box = sub_w / 2.0, sub_h / 2.0

    # 选「面积达标 + 离框中心最近」的连通域
    best = None
    best_score = None
    for lab in range(1, n + 1):
        ys, xs = np.where(labels == lab)
        area = xs.size
        if area < min_area:
            continue
        cx, cy = xs.mean(), ys.mean()
        dist2 = (cx - cx_box) ** 2 + (cy - cy_box) ** 2
        if best_score is None or dist2 < best_score:
            best_score = dist2
            best = (cx, cy, area, xs, ys)

    if best is None:
        # 全部太小，退而取最大暗块
        sizes = sorted(((np.count_nonzero(labels == l), l)
                        for l in range(1, n + 1)), reverse=True)
        if not sizes:
            raise ValueError("未检测到有效暗区，请重新框选。")
        lab = sizes[0][1]
        ys, xs = np.where(labels == lab)
        cx, cy, area = xs.mean(), ys.mean(), xs.size

    cx, cy, area, xs, ys = best if best is not None else (cx, cy, area, xs, ys)

    # 半径：等面积圆 + 二阶矩 取均值
    r_area = np.sqrt(area / np.pi)
    var = (((xs - cx) ** 2 + (ys - cy) ** 2).mean())
    r_moment = np.sqrt(2.0 * var) if var > 0 else r_area
    radius = float(max(1.0, (r_area + r_moment) / 2.0))

    return {
        "frap_x_roi": float(cx + bx1),
        "frap_y_roi": float(cy + by1),
        "radius": radius,
        "area": int(area),
        "threshold": float(t),
        "n_components": int(n),
    }


# ─────────────────────────────────────────────
#  Part 2 & 3: GUI
# ─────────────────────────────────────────────

class AutoFRAPApp(tk.Tk):
    """主应用窗口"""

    # Default / sentinel value for unset parameters
    _UNSET = "—"

    def __init__(self):
        super().__init__()
        self.title("FRAP_in_vivo.py")
        self.resizable(True, True)
        self.configure(bg="#1a1a2e")

        # ── Internal state ──────────────────────────────────────
        self.folder_path = tk.StringVar(value="")
        self.frap_radius = tk.DoubleVar(value=8.0)
        # 归一化参数（用户可在 GUI 设置）
        self.baseline_frames = tk.IntVar(value=2)   # 前几帧均值作为 baseline(100%)
        self.bleach_frame = tk.IntVar(value=3)      # bleach 后第一帧（1-based 帧号）
        # 背景区开关
        self.use_bg = tk.BooleanVar(value=False)

        # ROI corners (full-image pixel coords)
        self.corner1_x = None
        self.corner1_y = None
        self.corner2_x = None
        self.corner2_y = None

        # Background ROI: 用户只点左上角，右下角由主 ROI 尺寸推导
        self.bg_corner1_x = None
        self.bg_corner1_y = None

        # FRAP centre (ROI coords)
        self.frap_x = None
        self.frap_y = None

        # Centroid of reference frame (ROI coords)
        self.droplet_center_x = None
        self.droplet_center_y = None

        # Offsets
        self.offset_x = None
        self.offset_y = None

        self._build_ui()

    # ── 工具方法 ────────────────────────────────────────────

    def bg_corner2(self) -> tuple:
        """根据主 ROI 尺寸推导背景 ROI 右下角坐标，未启用时返回 (None, None)。"""
        if (self.bg_corner1_x is None or self.bg_corner1_y is None or
            None in (self.corner1_x, self.corner1_y,
                     self.corner2_x, self.corner2_y)):
            return None, None
        w = self.corner2_x - self.corner1_x
        h = self.corner2_y - self.corner1_y
        return self.bg_corner1_x + w, self.bg_corner1_y + h

    # ── UI Builder ───────────────────────────────────────────────

    def _build_ui(self):
        PAD = 10
        BG = "#1a1a2e"
        CARD = "#16213e"
        ACCENT = "#0f3460"
        HIGHLIGHT = "#e94560"
        TEXT = "#eaeaea"
        MUTED = "#8892a4"
        FONT_TITLE = ("Courier New", 18, "bold")
        FONT_LABEL = ("Courier New", 10)
        FONT_VAL = ("Courier New", 10, "bold")
        FONT_BTN = ("Courier New", 10, "bold")

        # ── Title ────────────────────────────────────────────────
        title_frame = tk.Frame(self, bg=BG, pady=10)
        title_frame.pack(fill="x", padx=PAD)
        tk.Label(title_frame, text="◈  FRAP_In_Vivo  ◈",
                 font=FONT_TITLE, fg=HIGHLIGHT, bg=BG).pack()
        tk.Label(title_frame,
                 text="Dynamic Droplet Tracking · FRAP Analysis",
                 font=("Courier New", 9), fg=MUTED, bg=BG).pack()

        sep = tk.Frame(self, bg=HIGHLIGHT, height=1)
        sep.pack(fill="x", padx=PAD, pady=(0, 6))

        # ── Row 1: Folder selection ──────────────────────────────
        folder_frame = tk.Frame(self, bg=CARD, padx=8, pady=6)
        folder_frame.pack(fill="x", padx=PAD, pady=3)
        tk.Label(folder_frame, text="DATA FOLDER", font=FONT_LABEL,
                 fg=MUTED, bg=CARD, width=14, anchor="w").pack(side="left")
        tk.Entry(folder_frame, textvariable=self.folder_path,
                 font=FONT_LABEL, bg=ACCENT, fg=TEXT,
                 insertbackground=TEXT, relief="flat", width=52).pack(side="left", padx=6)
        tk.Button(folder_frame, text="Browse", font=FONT_BTN,
                  bg=HIGHLIGHT, fg="white", activebackground="#c73652",
                  relief="flat", padx=8,
                  command=self._browse_folder).pack(side="left")

        # FRAP radius + 归一化参数 sub-row（同一行）
        rad_frame = tk.Frame(self, bg=CARD, padx=8, pady=4)
        rad_frame.pack(fill="x", padx=PAD, pady=(0, 3))

        tk.Label(rad_frame, text="FRAP RADIUS (px)", font=FONT_LABEL,
                 fg=MUTED, bg=CARD, anchor="w").pack(side="left")
        tk.Entry(rad_frame, textvariable=self.frap_radius,
                 font=FONT_LABEL, bg=ACCENT, fg=TEXT,
                 insertbackground=TEXT, relief="flat", width=7).pack(side="left", padx=(6, 0))

        # 右侧：归一化用的 baseline 帧数 与 bleach 帧号（1-based）
        tk.Label(rad_frame, text="BASELINE FRAMES", font=FONT_LABEL,
                 fg=MUTED, bg=CARD, anchor="w").pack(side="left", padx=(20, 0))
        tk.Entry(rad_frame, textvariable=self.baseline_frames,
                 font=FONT_LABEL, bg=ACCENT, fg=TEXT,
                 insertbackground=TEXT, relief="flat", width=5).pack(side="left", padx=(6, 0))

        tk.Label(rad_frame, text="BLEACH FRAME", font=FONT_LABEL,
                 fg=MUTED, bg=CARD, anchor="w").pack(side="left", padx=(20, 0))
        tk.Entry(rad_frame, textvariable=self.bleach_frame,
                 font=FONT_LABEL, bg=ACCENT, fg=TEXT,
                 insertbackground=TEXT, relief="flat", width=5).pack(side="left", padx=(6, 0))

        # 背景区开关复选框
        tk.Checkbutton(rad_frame, text="USE BACKGROUND",
                       variable=self.use_bg,
                       font=FONT_LABEL, fg=MUTED, bg=CARD,
                       activebackground=CARD, activeforeground=TEXT,
                       selectcolor=ACCENT, relief="flat",
                       bd=0, highlightthickness=0).pack(side="left", padx=(20, 0))

        # ── Row 2: Action buttons ────────────────────────────────
        btn_frame = tk.Frame(self, bg=BG, pady=6)
        btn_frame.pack(fill="x", padx=PAD)

        btn_style = dict(font=FONT_BTN, relief="flat", padx=18, pady=8,
                         cursor="hand2", activeforeground="white")
        tk.Button(btn_frame, text="⊕  偏移量-手动",
                  bg=ACCENT, fg=TEXT, activebackground="#1a4a80",
                  command=self._run_offset_calc, **btn_style).pack(side="left", padx=4)
        tk.Button(btn_frame, text="⊙  偏移量-自动",
                  bg=ACCENT, fg=TEXT, activebackground="#1a4a80",
                  command=self._run_auto_detect, **btn_style).pack(side="left", padx=4)
        tk.Button(btn_frame, text="◎  图像预览",
                  bg=ACCENT, fg=TEXT, activebackground="#1a4a80",
                  command=self._run_preview, **btn_style).pack(side="left", padx=4)
        tk.Button(btn_frame, text="▶  全数据分析",
                  bg=HIGHLIGHT, fg="white", activebackground="#c73652",
                  command=self._run_batch, **btn_style).pack(side="left", padx=4)
        tk.Button(btn_frame, text="◫  结果图表",
                  bg="#5c3a7a", fg="white", activebackground="#7a4ea0",
                  command=self._run_show_panel, **btn_style).pack(side="left", padx=4)
        tk.Button(btn_frame, text="⬡  CZI 解压",
                  bg="#1a5c3a", fg="white", activebackground="#237a50",
                  command=self._run_czi_extract, **btn_style).pack(side="left", padx=4)

        sep2 = tk.Frame(self, bg=ACCENT, height=1)
        sep2.pack(fill="x", padx=PAD, pady=(2, 6))

        # ── Rows 3-5: Parameter display ──────────────────────────
        param_outer = tk.Frame(self, bg=CARD, padx=10, pady=8)
        param_outer.pack(fill="x", padx=PAD, pady=2)
        tk.Label(param_outer, text="PARAMETERS", font=("Courier New", 8, "bold"),
                 fg=HIGHLIGHT, bg=CARD).grid(row=0, column=0, columnspan=8,
                                             sticky="w", pady=(0, 4))

        params = [
            ("corner1_x", "corner1_y", "corner2_x", "corner2_y"),
            ("FRAP_x", "FRAP_y", "droplet_center_x", "droplet_center_y"),
            ("offset_x", "offset_y", "bg_corner1_x", "bg_corner1_y"),
        ]
        self._param_vars = {}
        for r, row_params in enumerate(params):
            for c, name in enumerate(row_params):
                if not name:
                    continue
                tk.Label(param_outer, text=name + ":", font=FONT_LABEL,
                         fg=MUTED, bg=CARD).grid(row=r + 1, column=c * 2,
                                                  sticky="e", padx=(8, 2), pady=2)
                var = tk.StringVar(value=self._UNSET)
                self._param_vars[name] = var
                tk.Label(param_outer, textvariable=var, font=FONT_VAL,
                         fg=TEXT, bg=ACCENT, width=10,
                         anchor="center", relief="flat").grid(
                    row=r + 1, column=c * 2 + 1, sticky="w", padx=(0, 6), pady=2)

        # ── Log area ─────────────────────────────────────────────
        log_frame = tk.Frame(self, bg=BG, pady=4)
        log_frame.pack(fill="both", expand=True, padx=PAD, pady=(4, PAD))
        tk.Label(log_frame, text="LOG", font=("Courier New", 8, "bold"),
                 fg=HIGHLIGHT, bg=BG, anchor="w").pack(fill="x")
        self.log_box = tk.Text(log_frame, height=8, font=("Courier New", 9),
                               bg="#0d0d1a", fg="#7ec8e3",
                               insertbackground=TEXT, relief="flat",
                               state="disabled", wrap="word")
        self.log_box.pack(fill="both", expand=True)
        sb = tk.Scrollbar(log_frame, command=self.log_box.yview)
        self.log_box.configure(yscrollcommand=sb.set)

        self.log("FRAP_In_Vivo 已启动。请选择数据文件夹，然后执行偏移量计算。")

    # ── Helpers ──────────────────────────────────────────────────

    def log(self, msg: str):
        ts = datetime.now().strftime("%H:%M:%S")
        self.log_box.configure(state="normal")
        self.log_box.insert("end", f"[{ts}] {msg}\n")
        self.log_box.see("end")
        self.log_box.configure(state="disabled")

    def _set_param(self, name: str, value):
        if value is None:
            self._param_vars[name].set(self._UNSET)
        else:
            self._param_vars[name].set(f"{value:.2f}" if isinstance(value, float) else str(value))

    def _refresh_params(self):
        mapping = {
            "corner1_x": self.corner1_x,
            "corner1_y": self.corner1_y,
            "corner2_x": self.corner2_x,
            "corner2_y": self.corner2_y,
            "FRAP_x": self.frap_x,
            "FRAP_y": self.frap_y,
            "droplet_center_x": self.droplet_center_x,
            "droplet_center_y": self.droplet_center_y,
            "offset_x": self.offset_x,
            "offset_y": self.offset_y,
            "bg_corner1_x": self.bg_corner1_x,
            "bg_corner1_y": self.bg_corner1_y,
        }
        for k, v in mapping.items():
            self._set_param(k, v)

    def _browse_folder(self):
        path = filedialog.askdirectory(title="选择数据文件夹")
        if path:
            self.folder_path.set(path)
            self.log(f"数据文件夹：{path}")

    def _pick_frame(self, title="选择一帧图片") -> str:
        folder = self.folder_path.get()
        init = folder if folder else "/"
        fp = filedialog.askopenfilename(
            title=title,
            initialdir=init,
            filetypes=[
                ("支持的图像格式",
                 "*.jpg *.jpeg *.JPG *.JPEG *.png *.PNG "
                 "*.tif *.tiff *.TIF *.TIFF *.czi *.CZI"),
                ("JPEG", "*.jpg *.jpeg *.JPG *.JPEG"),
                ("PNG",  "*.png *.PNG"),
                ("TIFF", "*.tif *.tiff *.TIF *.TIFF"),
                ("CZI",  "*.czi *.CZI"),
                ("所有文件", "*.*"),
            ])
        return fp

    def _validate_roi(self) -> bool:
        if None in (self.corner1_x, self.corner1_y,
                    self.corner2_x, self.corner2_y):
            messagebox.showwarning("参数缺失", "请先完成偏移量计算以设置 ROI。")
            return False
        return True

    def _validate_all(self) -> bool:
        if not self._validate_roi():
            return False
        if None in (self.frap_x, self.frap_y, self.offset_x, self.offset_y):
            messagebox.showwarning("参数缺失", "请先完成偏移量计算以设置 FRAP 参数。")
            return False
        return True

    # ── Feature 1: Offset Calculation ───────────────────────────

    def _run_offset_calc(self):
        fp = self._pick_frame("选择参考帧（用于偏移量计算）")
        if not fp:
            return
        self.log(f"偏移量计算 — 参考帧：{os.path.basename(fp)}"
                 f"{'  [+background]' if self.use_bg.get() else ''}")
        # 每次重选时清空背景坐标，避免与新 ROI 不匹配
        self.bg_corner1_x = self.bg_corner1_y = None
        self._refresh_params()
        OffsetWindow(self, fp)

    # ── Feature 1b: Auto-detect FRAP circle ─────────────────────

    def _run_auto_detect(self):
        """
        自动检测流程：选参考帧 → 复用 OffsetWindow 选 ROI →
        下一步走 AutoDetectWindow（框选大致范围 + 自动找圆心/半径）。
        """
        fp = self._pick_frame("选择参考帧（用于自动检测 FRAP 圆）")
        if not fp:
            return
        self.log(f"自动检测 — 参考帧：{os.path.basename(fp)}"
                 f"{'  [+background]' if self.use_bg.get() else ''}")
        self.bg_corner1_x = self.bg_corner1_y = None
        self._refresh_params()
        OffsetWindow(self, fp, next_step=AutoDetectWindow)

    # ── Feature 2: Image Preview ─────────────────────────────────

    def _run_preview(self):
        if not self._validate_all():
            return
        fp = self._pick_frame("选择预览帧")
        if not fp:
            return
        self.log(f"图像预览：{os.path.basename(fp)}")
        PreviewWindow(self, fp)

    # ── Feature 3: Batch Analysis ────────────────────────────────

    def _run_batch(self):
        if not self._validate_all():
            return
        folder = self.folder_path.get()
        if not folder:
            messagebox.showwarning("未选择文件夹", "请先选择数据文件夹。")
            return

        # 背景启用且坐标完整 → 传入
        use_bg = self.use_bg.get() and (self.bg_corner1_x is not None
                                        and self.bg_corner1_y is not None)
        bg_corner1 = (self.bg_corner1_x, self.bg_corner1_y) if use_bg else None
        if self.use_bg.get() and not use_bg:
            self.log("  注意：勾选了 USE BACKGROUND 但未设置背景 ROI 坐标，将按无背景处理。")

        radius = self.frap_radius.get()
        try:
            baseline_frames = int(self.baseline_frames.get())
        except (tk.TclError, ValueError):
            baseline_frames = 2
        try:
            bleach_frame = int(self.bleach_frame.get())
        except (tk.TclError, ValueError):
            bleach_frame = 0
        self.log(f"全数据分析开始 — 文件夹：{folder}，FRAP半径：{radius} px，"
                 f"baseline帧数={baseline_frames}，bleach帧#={bleach_frame}，"
                 f"background={'on' if use_bg else 'off'}")

        # 用一个可变容器暂存每帧结果，供 progress_cb 读取
        _partial: list = []

        def _progress(i, t, fn):
            if _partial:
                last = _partial[-1]
                if "FRAP_bg_intensity" in last:
                    self.log(f"  [{i}/{t}] {fn}  I={last['FRAP_intensity']:.4f}  "
                             f"bg={last['FRAP_bg_intensity']:.4f}")
                else:
                    self.log(f"  [{i}/{t}] {fn}  I={last['FRAP_intensity']:.4f}")
            else:
                self.log(f"  [{i}/{t}] {fn}")
            self.update_idletasks()  # 让 GUI 实时刷新

        marked_dir = os.path.join(folder, "marked_files")

        try:
            results = batch_analysis(
                folder,
                self.corner1_x, self.corner1_y,
                self.corner2_x, self.corner2_y,
                self.offset_x, self.offset_y,
                radius,
                marked_dir=marked_dir,
                progress_cb=_progress,
                frame_cb=lambda row: _partial.append(row),
                bg_corner1=bg_corner1,
            )
        except Exception as e:
            messagebox.showerror("分析错误", str(e))
            self.log(f"错误：{e}")
            return

        # ── 功能1：双点归一化（baseline → 100%，bleach 帧 → 0%）──
        norm_out = normalize_intensities(
            results,
            baseline_frames=baseline_frames,
            bleach_frame=bleach_frame,
            has_bg=use_bg,
        )
        baseline = norm_out["baseline"]
        i_bleach = norm_out["i_bleach"]
        if baseline - i_bleach != 0:
            self.log(f"  双点归一化：baseline(前{baseline_frames}帧均值)={baseline:.4f} → 100%；"
                     f"bleach帧#{bleach_frame} 强度={i_bleach:.4f} → 0%")
        else:
            self.log("  归一化：baseline 与 bleach 强度相等或无效，归一化列全部置 0。")

        if use_bg:
            bc = norm_out["baseline_corr"]
            ic = norm_out["i_bleach_corr"]
            if bc - ic != 0:
                self.log(f"  减法扣背景归一化：baseline_corr={bc:.4f} → 100%；"
                         f"bleach 强度_corr={ic:.4f} → 0%")
            else:
                self.log("  减法扣背景归一化：基线无效，corrected_norm 列全部置 0。")

            br = norm_out["baseline_ratio"]
            ir = norm_out["i_bleach_ratio"]
            if br - ir != 0:
                self.log(f"  比值归一化（除法）：baseline_ratio={br:.4f} → 100%；"
                         f"bleach 比值={ir:.4f} → 0%")
            else:
                self.log("  比值归一化：基线无效，ratio_norm 列全部置 0。")

        # Save CSV
        out_path = os.path.join(folder, "FRAP_results.csv")
        fields = ["frame_index", "filename",
                  "droplet_center_x", "droplet_center_y",
                  "frap_x_roi", "frap_y_roi",
                  "frap_x_global", "frap_y_global",
                  "FRAP_intensity", "FRAP_intensity_norm",
                  "pixel_count"]
        if use_bg:
            fields += [
                "bg_frap_x_global", "bg_frap_y_global",
                "FRAP_bg_intensity", "bg_pixel_count",
                "FRAP_intensity_corrected",
                "FRAP_intensity_corrected_norm",
                "FRAP_intensity_ratio",
                "FRAP_intensity_ratio_norm",
            ]
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for r in results:
                writer.writerow({k: r.get(k, "") for k in fields})

        self.log(f"分析完成！共处理 {len(results)} 帧。")
        self.log(f"  CSV  → {out_path}")
        self.log(f"  标注图 → {marked_dir}/")
        messagebox.showinfo(
            "完成",
            f"分析完成！共 {len(results)} 帧\n\n"
            f"CSV 结果：{out_path}\n"
            f"标注图片：{marked_dir}/\n\n"
            + ("（含扣背景列 FRAP_intensity_corrected[_norm]）\n\n"
               if use_bg else "")
            + "可点击「◫ 结果图表」查看曲线。"
        )

    # ── Feature 3b: Result Panel (figures) ──────────────────────

    def _run_show_panel(self):
        """
        功能2：读取 FRAP_results.csv，弹出图表面板。
        默认定位到数据文件夹下的 FRAP_results.csv，也可手动选择其他 CSV。
        """
        if not _MPL_AVAILABLE:
            messagebox.showerror(
                "缺少依赖",
                "结果图表功能需要安装 matplotlib：\n\n  pip install matplotlib")
            return

        folder = self.folder_path.get()
        init = folder if folder else "/"
        default_csv = os.path.join(folder, "FRAP_results.csv") if folder else ""
        csv_path = filedialog.askopenfilename(
            title="选择 FRAP_results.csv",
            initialdir=init,
            initialfile=(os.path.basename(default_csv)
                         if default_csv and os.path.exists(default_csv) else ""),
            filetypes=[("CSV 文件", "*.csv *.CSV"), ("所有文件", "*.*")])
        if not csv_path:
            return

        self.log(f"结果图表：{os.path.basename(csv_path)}")
        try:
            ResultPanelWindow(self, csv_path)
        except Exception as e:
            messagebox.showerror("绘图错误", str(e))
            self.log(f"错误：{e}")

    # ── Feature 4: CZI Stack Extraction ─────────────────────────

    def _run_czi_extract(self):
        """选择 CZI 文件，解压为逐帧 JPEG，保存到同目录下的子文件夹。"""
        if not _CZI_AVAILABLE:
            messagebox.showerror(
                "缺少依赖",
                "读取 CZI 文件需要安装 czifile：\n\n  pip install czifile"
            )
            return

        czi_path = filedialog.askopenfilename(
            title="选择 CZI Stack 文件",
            filetypes=[
                ("Zeiss CZI 文件", "*.czi *.CZI"),
                ("所有文件", "*.*"),
            ],
        )
        if not czi_path:
            return

        # 输出目录：CZI 同级目录下，以文件名（不含扩展）命名的文件夹
        base_name = os.path.splitext(os.path.basename(czi_path))[0]
        out_dir = os.path.join(os.path.dirname(czi_path), base_name + "_frames")

        self.log(f"CZI 解压开始：{os.path.basename(czi_path)}")
        self.log(f"  输出目录：{out_dir}")
        self.update_idletasks()

        def _progress(i, total, fname):
            self.log(f"  [{i}/{total}] 已保存 {fname}")
            self.update_idletasks()

        try:
            saved = extract_czi_stack(
                czi_path,
                out_dir,
                prefix=base_name,
                jpeg_quality=95,
                progress_cb=_progress,
            )
        except Exception as e:
            messagebox.showerror("解压错误", str(e))
            self.log(f"错误：{e}")
            return

        self.log(f"CZI 解压完成！共 {len(saved)} 帧 → {out_dir}/")

        # 询问是否自动将输出目录填入数据文件夹
        if messagebox.askyesno(
            "解压完成",
            f"共解压 {len(saved)} 帧。\n\n"
            f"输出目录：\n{out_dir}\n\n"
            "是否将该目录自动设为当前数据文件夹？",
        ):
            self.folder_path.set(out_dir)
            self.log(f"  数据文件夹已更新为：{out_dir}")


# ─────────────────────────────────────────────
#  Window 1 & 2: Offset Calculation Workflow
# ─────────────────────────────────────────────

class OffsetWindow(tk.Toplevel):
    """
    窗口1：展示整帧图片，用户依次点击两个角确定 ROI。
    若主程序勾选了 USE BACKGROUND，则再点第 3 个点作为背景 ROI 左上角；
    背景 ROI 尺寸与主 ROI 一致（程序自动推导右下角）。

    next_step：Continue 之后要打开的窗口类（构造签名须为 (app, filepath, parent_win)）。
               None → 默认手动选点窗口 FRAPPickerWindow；
               传入 AutoDetectWindow → 自动检测流程。
    """
    _CANVAS_MAX = 700  # canvas 最大显示尺寸

    def __init__(self, app: AutoFRAPApp, filepath: str, next_step=None):
        super().__init__(app)
        self.app = app
        self.filepath = filepath
        self.next_step = next_step   # 下一步窗口类；None → 默认手动选点流程
        self.use_bg = bool(app.use_bg.get())
        self.title("Step 1 — 选择 ROI"
                   + ("（+ 背景 ROI 左上角）" if self.use_bg else ""))
        self.configure(bg="#1a1a2e")
        self.resizable(False, False)

        # Load image
        self.pil_img = open_as_rgb(filepath)
        self.orig_w, self.orig_h = self.pil_img.size
        self.scale = min(self._CANVAS_MAX / self.orig_w,
                         self._CANVAS_MAX / self.orig_h, 1.0)
        dw = int(self.orig_w * self.scale)
        dh = int(self.orig_h * self.scale)
        self.display_img = self.pil_img.resize((dw, dh), Image.LANCZOS)

        self._click_pts = []  # up to 2 ROI canvas pts (+ 1 bg pt)
        self._rect_id = None
        self._bg_rect_id = None
        self._bg_marker_id = None

        self._build()

    def _expected_clicks(self) -> int:
        return 3 if self.use_bg else 2

    def _build(self):
        BG = "#1a1a2e"
        CARD = "#16213e"
        ACCENT = "#0f3460"
        HIGHLIGHT = "#e94560"
        TEXT = "#eaeaea"
        MUTED = "#8892a4"

        hint = "左键单击依次选定 ROI 左上角、右下角"
        if self.use_bg:
            hint += "，再点第 3 个点作为背景 ROI 左上角（尺寸与主 ROI 一致）"
        tk.Label(self, text=hint,
                 font=("Courier New", 9), fg=MUTED, bg=BG).pack(pady=(8, 2))

        # Canvas
        self.canvas = tk.Canvas(self,
                                width=self.display_img.width,
                                height=self.display_img.height,
                                bg=CARD, relief="flat", bd=0,
                                cursor="crosshair")
        self.canvas.pack(padx=10)
        self._tk_img = ImageTk.PhotoImage(self.display_img)
        self.canvas.create_image(0, 0, anchor="nw", image=self._tk_img)
        self.canvas.bind("<Button-1>", self._on_click)

        # Status label
        self.status_var = tk.StringVar(value="点 #1：ROI 左上角")
        tk.Label(self, textvariable=self.status_var,
                 font=("Courier New", 9, "bold"), fg=HIGHLIGHT, bg=BG).pack(pady=4)

        # Buttons
        btn_row = tk.Frame(self, bg=BG)
        btn_row.pack(pady=(4, 10))
        tk.Button(btn_row, text="Reset", font=("Courier New", 10, "bold"),
                  bg=ACCENT, fg=TEXT, relief="flat", padx=14, pady=6,
                  command=self._reset).pack(side="left", padx=8)
        tk.Button(btn_row, text="Continue →", font=("Courier New", 10, "bold"),
                  bg="#e94560", fg="white", relief="flat", padx=14, pady=6,
                  command=self._continue).pack(side="left", padx=8)

    def _on_click(self, event):
        if len(self._click_pts) >= self._expected_clicks():
            return
        self._click_pts.append((event.x, event.y))

        # Convert canvas -> original image coords
        ox = int(event.x / self.scale)
        oy = int(event.y / self.scale)

        if len(self._click_pts) == 1:
            self.app.corner1_x = ox
            self.app.corner1_y = oy
            self.status_var.set("点 #2：ROI 右下角")
            self.canvas.create_oval(event.x - 4, event.y - 4,
                                    event.x + 4, event.y + 4,
                                    fill="#e94560", outline="white")
        elif len(self._click_pts) == 2:
            self.app.corner2_x = ox
            self.app.corner2_y = oy
            if self.use_bg:
                self.status_var.set("点 #3：背景 ROI 左上角")
            else:
                self.status_var.set("ROI 已选定，单击 Continue 继续。")
            x1c, y1c = self._click_pts[0]
            x2c, y2c = self._click_pts[1]
            if self._rect_id:
                self.canvas.delete(self._rect_id)
            self._rect_id = self.canvas.create_rectangle(
                x1c, y1c, x2c, y2c,
                outline="#f0e040", dash=(4, 4), width=2)
        else:
            # 第 3 个点：背景 ROI 左上角
            self.app.bg_corner1_x = ox
            self.app.bg_corner1_y = oy
            # 计算背景 ROI 在 canvas 上的覆盖框（用主 ROI 尺寸）
            roi_w_canvas = abs(self._click_pts[1][0] - self._click_pts[0][0])
            roi_h_canvas = abs(self._click_pts[1][1] - self._click_pts[0][1])
            bx1c, by1c = event.x, event.y
            bx2c, by2c = bx1c + roi_w_canvas, by1c + roi_h_canvas
            if self._bg_rect_id:
                self.canvas.delete(self._bg_rect_id)
            self._bg_rect_id = self.canvas.create_rectangle(
                bx1c, by1c, bx2c, by2c,
                outline="#2ecc71", dash=(4, 4), width=2)
            if self._bg_marker_id:
                self.canvas.delete(self._bg_marker_id)
            self._bg_marker_id = self.canvas.create_oval(
                event.x - 4, event.y - 4, event.x + 4, event.y + 4,
                fill="#2ecc71", outline="white")
            self.status_var.set("ROI + 背景 ROI 已选定，单击 Continue 继续。")

        self.app._refresh_params()
        self.app.log(f"  点 #{len(self._click_pts)} 选定：原图 ({ox}, {oy})")

    def _reset(self):
        self._click_pts.clear()
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=self._tk_img)
        self._rect_id = None
        self._bg_rect_id = None
        self._bg_marker_id = None
        self.app.corner1_x = self.app.corner1_y = None
        self.app.corner2_x = self.app.corner2_y = None
        self.app.bg_corner1_x = self.app.bg_corner1_y = None
        self.app._refresh_params()
        self.status_var.set("点 #1：ROI 左上角")
        self.app.log("  ROI/背景坐标已重置。")

    def _continue(self):
        if None in (self.app.corner1_x, self.app.corner1_y,
                    self.app.corner2_x, self.app.corner2_y):
            messagebox.showwarning("未完成", "请先选定 ROI 的两个角点。")
            return
        if self.use_bg and (self.app.bg_corner1_x is None
                            or self.app.bg_corner1_y is None):
            messagebox.showwarning("未完成", "请先选定背景 ROI 左上角（第 3 个点）。")
            return

        # Normalise so corner1 is always top-left
        x1 = min(self.app.corner1_x, self.app.corner2_x)
        y1 = min(self.app.corner1_y, self.app.corner2_y)
        x2 = max(self.app.corner1_x, self.app.corner2_x)
        y2 = max(self.app.corner1_y, self.app.corner2_y)
        self.app.corner1_x, self.app.corner1_y = x1, y1
        self.app.corner2_x, self.app.corner2_y = x2, y2

        # 校验背景 ROI 不越界
        if self.use_bg:
            roi_w, roi_h = x2 - x1, y2 - y1
            bx1, by1 = self.app.bg_corner1_x, self.app.bg_corner1_y
            if (bx1 < 0 or by1 < 0
                    or bx1 + roi_w > self.orig_w
                    or by1 + roi_h > self.orig_h):
                messagebox.showwarning(
                    "背景 ROI 越界",
                    f"背景 ROI ({bx1},{by1}) → ({bx1+roi_w},{by1+roi_h}) "
                    f"超出图像范围 ({self.orig_w}×{self.orig_h})，请重选。")
                return

        self.app._refresh_params()

        # 根据 next_step 决定下一步窗口（默认手动选点 FRAPPickerWindow）
        next_cls = self.next_step if self.next_step is not None else FRAPPickerWindow
        next_cls(self.app, self.filepath, self)


class FRAPPickerWindow(tk.Toplevel):
    """
    窗口2：展示 ROI 区域（4× 放大），用户点击选定 FRAP 圆心。
    坐标参考系为 ROI 内坐标。
    若启用了背景 ROI，圆心选定后自动在画布上叠加显示对应背景区位置示意。
    （背景圆位置不需要用户再点；它在分析时直接固定平移到背景 ROI 起点。）
    """
    ZOOM = 4

    def __init__(self, app: AutoFRAPApp, filepath: str, parent_win: OffsetWindow):
        super().__init__(app)
        self.app = app
        self.parent_win = parent_win
        self.title("Step 2 — 选择 FRAP 圆心（ROI 坐标系）")
        self.configure(bg="#1a1a2e")
        self.resizable(False, False)

        # Crop ROI from original image
        pil = open_as_rgb(filepath)
        roi_pil = pil.crop((app.corner1_x, app.corner1_y,
                             app.corner2_x, app.corner2_y))
        roi_w, roi_h = roi_pil.size
        zoom_w = roi_w * self.ZOOM
        zoom_h = roi_h * self.ZOOM
        self.roi_zoom = roi_pil.resize((zoom_w, zoom_h), Image.NEAREST)
        self.roi_w = roi_w
        self.roi_h = roi_h

        # Compute centroid of ROI in reference frame
        gray = load_image_as_gray(filepath)
        roi_gray = get_roi(gray,
                           app.corner1_x, app.corner1_y,
                           app.corner2_x, app.corner2_y)
        dcx, dcy = calc_centroid(roi_gray)
        self.app.droplet_center_x = dcx
        self.app.droplet_center_y = dcy
        self.app._refresh_params()
        self.app.log(f"  参考帧质心（ROI 坐标）：({dcx:.2f}, {dcy:.2f})")

        self._frap_marker = None
        self._bg_marker = None
        self._build()

    def _build(self):
        BG = "#1a1a2e"
        CARD = "#16213e"
        ACCENT = "#0f3460"
        TEXT = "#eaeaea"
        MUTED = "#8892a4"

        tk.Label(self, text=f"ROI 区域（{self.ZOOM}× 放大）— 左键单击选定 FRAP 圆心"
                            + ("，背景 FRAP 圆将以绿色显示在相同 ROI 坐标"
                               if self.app.bg_corner1_x is not None else ""),
                 font=("Courier New", 9), fg=MUTED, bg=BG).pack(pady=(8, 2))

        self.canvas = tk.Canvas(self,
                                width=self.roi_zoom.width,
                                height=self.roi_zoom.height,
                                bg=CARD, relief="flat", bd=0,
                                cursor="crosshair")
        self.canvas.pack(padx=10)
        self._tk_img = ImageTk.PhotoImage(self.roi_zoom)
        self.canvas.create_image(0, 0, anchor="nw", image=self._tk_img)

        # Draw centroid marker
        cx_c = self.app.droplet_center_x * self.ZOOM
        cy_c = self.app.droplet_center_y * self.ZOOM
        r = 5
        self.canvas.create_oval(cx_c - r, cy_c - r, cx_c + r, cy_c + r,
                                 outline="#f0e040", width=2)
        self.canvas.create_line(cx_c - 8, cy_c, cx_c + 8, cy_c,
                                 fill="#f0e040", width=1)
        self.canvas.create_line(cx_c, cy_c - 8, cx_c, cy_c + 8,
                                 fill="#f0e040", width=1)
        self.canvas.create_text(cx_c + 10, cy_c - 10,
                                 text="centroid", fill="#f0e040",
                                 font=("Courier New", 7))

        self.canvas.bind("<Button-1>", self._on_click)

        self.status_var = tk.StringVar(value="请单击选定 FRAP 圆心")
        tk.Label(self, textvariable=self.status_var,
                 font=("Courier New", 9, "bold"), fg="#e94560", bg=BG).pack(pady=4)

        btn_row = tk.Frame(self, bg=BG)
        btn_row.pack(pady=(4, 10))
        tk.Button(btn_row, text="Reset", font=("Courier New", 10, "bold"),
                  bg=ACCENT, fg=TEXT, relief="flat", padx=14, pady=6,
                  command=self._reset).pack(side="left", padx=8)
        tk.Button(btn_row, text="Finish Setting ✓",
                  font=("Courier New", 10, "bold"),
                  bg="#e94560", fg="white", relief="flat", padx=14, pady=6,
                  command=self._finish).pack(side="left", padx=8)

    def _on_click(self, event):
        # Canvas coords → ROI coords
        roi_x = event.x / self.ZOOM
        roi_y = event.y / self.ZOOM

        self.app.frap_x = roi_x
        self.app.frap_y = roi_y

        # Recompute offset
        ox, oy = calc_offset(roi_x, roi_y,
                              self.app.droplet_center_x,
                              self.app.droplet_center_y)
        self.app.offset_x = ox
        self.app.offset_y = oy
        self.app._refresh_params()

        # Draw FRAP marker
        if self._frap_marker:
            for item in self._frap_marker:
                self.canvas.delete(item)
        r_px = self.app.frap_radius.get() * self.ZOOM
        cx_c, cy_c = event.x, event.y
        items = [
            self.canvas.create_oval(cx_c - r_px, cy_c - r_px,
                                     cx_c + r_px, cy_c + r_px,
                                     outline="#4fc3f7", width=2),
            self.canvas.create_oval(cx_c - 3, cy_c - 3,
                                     cx_c + 3, cy_c + 3,
                                     fill="#4fc3f7", outline=""),
        ]
        self._frap_marker = items

        # 如果启用了背景，画一个示意性的「背景 FRAP 圆」叠加在当前 ROI 视图上
        # —— 由于背景 ROI 与主 ROI 同尺寸，相同的 (roi_x, roi_y) 在背景 ROI
        # 内对应同一个相对位置，因此可用同样的 canvas 坐标画绿色圆做提示。
        if self._bg_marker:
            for item in self._bg_marker:
                self.canvas.delete(item)
            self._bg_marker = None
        if self.app.bg_corner1_x is not None and self.app.bg_corner1_y is not None:
            self._bg_marker = [
                self.canvas.create_oval(cx_c - r_px, cy_c - r_px,
                                        cx_c + r_px, cy_c + r_px,
                                        outline="#2ecc71", width=2, dash=(3, 3)),
                self.canvas.create_text(cx_c, cy_c - r_px - 8,
                                        text="bg-FRAP (same ROI-coords)",
                                        fill="#2ecc71",
                                        font=("Courier New", 7)),
            ]

        self.status_var.set(f"FRAP 圆心（ROI）：({roi_x:.1f}, {roi_y:.1f})  "
                            f"偏移：({ox:.2f}, {oy:.2f})")
        self.app.log(f"  FRAP 圆心（ROI）：({roi_x:.2f}, {roi_y:.2f})  "
                     f"offset=({ox:.2f}, {oy:.2f})")

    def _reset(self):
        self.app.frap_x = self.app.frap_y = None
        self.app.offset_x = self.app.offset_y = None
        self.app._refresh_params()
        if self._frap_marker:
            for item in self._frap_marker:
                self.canvas.delete(item)
            self._frap_marker = None
        if self._bg_marker:
            for item in self._bg_marker:
                self.canvas.delete(item)
            self._bg_marker = None
        self.status_var.set("请单击选定 FRAP 圆心")
        self.app.log("  FRAP 参数已重置。")

    def _finish(self):
        if self.app.frap_x is None:
            messagebox.showwarning("未选定", "请先单击选定 FRAP 圆心。")
            return
        self.app.log("  偏移量计算完成。窗口已关闭。")
        self.parent_win.destroy()
        self.destroy()

# ════════════════════════════════════════════════════════════
#  以下类放在 FRAPPickerWindow 类之后
# ════════════════════════════════════════════════════════════

class AutoDetectWindow(tk.Toplevel):
    """
    自动检测窗口：在放大的 ROI 中两次点击框选 FRAP 大致范围，
    程序按「暗 + 圆形」特征自动检测圆心与半径，并写回主程序参数。
    与 FRAPPickerWindow 输出一致（frap_x/y、offset_x/y），可直接进入分析。
    若启用了背景 ROI，检测完成后会在画布叠加绿色示意圆。
    """
    ZOOM = 4

    def __init__(self, app: AutoFRAPApp, filepath: str, parent_win):
        super().__init__(app)
        self.app = app
        self.parent_win = parent_win
        self.title("自动检测 — 框选 FRAP 大致范围")
        self.configure(bg="#1a1a2e")
        self.resizable(False, False)

        pil = open_as_rgb(filepath)
        roi_pil = pil.crop((app.corner1_x, app.corner1_y,
                            app.corner2_x, app.corner2_y))
        self.roi_w, self.roi_h = roi_pil.size
        self.roi_zoom = roi_pil.resize(
            (self.roi_w * self.ZOOM, self.roi_h * self.ZOOM), Image.NEAREST)

        # 检测用灰度 ROI
        gray = load_image_as_gray(filepath)
        self.roi_gray = get_roi(gray, app.corner1_x, app.corner1_y,
                                app.corner2_x, app.corner2_y)

        # 质心（用于 offset，与手动流程一致）
        dcx, dcy = calc_centroid(self.roi_gray)
        self.app.droplet_center_x = dcx
        self.app.droplet_center_y = dcy
        self.app._refresh_params()
        self.app.log(f"  参考帧质心（ROI 坐标）：({dcx:.2f}, {dcy:.2f})")

        self._box_pts = []        # 最多两个 canvas 点
        self._box_id = None
        self._detect_items = []
        self._bg_marker = []
        self._build()

    # ── UI ──────────────────────────────────────────────────
    def _build(self):
        BG = "#1a1a2e"; CARD = "#16213e"; ACCENT = "#0f3460"
        TEXT = "#eaeaea"; MUTED = "#8892a4"

        tk.Label(self,
                 text=f"ROI 区域（{self.ZOOM}× 放大）— 左键依次点击框选 FRAP 大致范围",
                 font=("Courier New", 9), fg=MUTED, bg=BG).pack(pady=(8, 2))

        self.canvas = tk.Canvas(self,
                                width=self.roi_zoom.width,
                                height=self.roi_zoom.height,
                                bg=CARD, relief="flat", bd=0,
                                cursor="crosshair")
        self.canvas.pack(padx=10)
        self._tk_img = ImageTk.PhotoImage(self.roi_zoom)
        self._draw_base()
        self.canvas.bind("<Button-1>", self._on_click)

        self.status_var = tk.StringVar(value="点 #1：大致范围左上角")
        tk.Label(self, textvariable=self.status_var,
                 font=("Courier New", 9, "bold"), fg="#e94560", bg=BG).pack(pady=4)

        btn_row = tk.Frame(self, bg=BG)
        btn_row.pack(pady=(4, 10))
        tk.Button(btn_row, text="重新框选", font=("Courier New", 10, "bold"),
                  bg=ACCENT, fg=TEXT, relief="flat", padx=14, pady=6,
                  command=self._reset).pack(side="left", padx=8)
        tk.Button(btn_row, text="重新检测", font=("Courier New", 10, "bold"),
                  bg="#5c3a7a", fg="white", relief="flat", padx=14, pady=6,
                  command=self._run_detection).pack(side="left", padx=8)
        tk.Button(btn_row, text="Finish ✓", font=("Courier New", 10, "bold"),
                  bg="#e94560", fg="white", relief="flat", padx=14, pady=6,
                  command=self._finish).pack(side="left", padx=8)

    def _draw_base(self):
        """绘制底图 + 质心十字。"""
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=self._tk_img)
        cx = self.app.droplet_center_x * self.ZOOM
        cy = self.app.droplet_center_y * self.ZOOM
        self.canvas.create_line(cx - 8, cy, cx + 8, cy, fill="#f0e040", width=1)
        self.canvas.create_line(cx, cy - 8, cx, cy + 8, fill="#f0e040", width=1)
        self.canvas.create_text(cx + 10, cy - 10, text="centroid",
                                fill="#f0e040", font=("Courier New", 7))

    # ── 交互 ────────────────────────────────────────────────
    def _on_click(self, event):
        if len(self._box_pts) >= 2:
            return
        self._box_pts.append((event.x, event.y))
        self.canvas.create_oval(event.x - 3, event.y - 3,
                                event.x + 3, event.y + 3,
                                fill="#e94560", outline="white")
        if len(self._box_pts) == 1:
            self.status_var.set("点 #2：大致范围右下角")
        else:
            (x1c, y1c), (x2c, y2c) = self._box_pts
            if self._box_id:
                self.canvas.delete(self._box_id)
            self._box_id = self.canvas.create_rectangle(
                x1c, y1c, x2c, y2c, outline="#f0e040", dash=(4, 4), width=2)
            self._run_detection()   # 框选完毕自动检测

    def _run_detection(self):
        if len(self._box_pts) < 2:
            messagebox.showwarning("未框选", "请先框选 FRAP 大致范围（两次单击）。")
            return
        (x1c, y1c), (x2c, y2c) = self._box_pts
        bx1 = min(x1c, x2c) / self.ZOOM; by1 = min(y1c, y2c) / self.ZOOM
        bx2 = max(x1c, x2c) / self.ZOOM; by2 = max(y1c, y2c) / self.ZOOM
        try:
            res = detect_frap_circle(self.roi_gray, bx1, by1, bx2, by2)
        except Exception as e:
            messagebox.showerror("检测失败", str(e))
            self.app.log(f"  自动检测失败：{e}")
            return

        fx, fy, r = res["frap_x_roi"], res["frap_y_roi"], res["radius"]
        self.app.frap_x = fx
        self.app.frap_y = fy
        ox, oy = calc_offset(fx, fy,
                             self.app.droplet_center_x,
                             self.app.droplet_center_y)
        self.app.offset_x, self.app.offset_y = ox, oy
        self.app.frap_radius.set(round(r, 2))   # 半径写回主界面输入框
        self.app._refresh_params()

        self._draw_detection(fx, fy, r)
        self.status_var.set(
            f"检测：圆心ROI=({fx:.1f},{fy:.1f})  半径={r:.2f}  偏移=({ox:.2f},{oy:.2f})")
        self.app.log(
            f"  自动检测 → 圆心ROI=({fx:.2f},{fy:.2f})  半径={r:.2f}px  "
            f"面积={res['area']}px  阈值={res['threshold']:.1f}  "
            f"暗区数={res['n_components']}  offset=({ox:.2f},{oy:.2f})")

    def _draw_detection(self, fx, fy, r):
        for it in self._detect_items:
            self.canvas.delete(it)
        self._detect_items = []
        for it in self._bg_marker:
            self.canvas.delete(it)
        self._bg_marker = []

        cx, cy, rr = fx * self.ZOOM, fy * self.ZOOM, r * self.ZOOM
        self._detect_items.append(self.canvas.create_oval(
            cx - rr, cy - rr, cx + rr, cy + rr, outline="#4fc3f7", width=2))
        self._detect_items.append(self.canvas.create_oval(
            cx - 3, cy - 3, cx + 3, cy + 3, fill="#4fc3f7", outline=""))

        # 启用背景时，叠加一个绿色示意圆，提示背景 FRAP 圆将使用相同 ROI 坐标
        if (self.app.bg_corner1_x is not None
                and self.app.bg_corner1_y is not None):
            self._bg_marker = [
                self.canvas.create_oval(cx - rr, cy - rr, cx + rr, cy + rr,
                                        outline="#2ecc71", width=2, dash=(3, 3)),
                self.canvas.create_text(cx, cy - rr - 8,
                                        text="bg-FRAP (same ROI-coords)",
                                        fill="#2ecc71",
                                        font=("Courier New", 7)),
            ]

    def _reset(self):
        self._box_pts = []
        self._box_id = None
        self._detect_items = []
        self._bg_marker = []
        self._draw_base()
        self.app.frap_x = self.app.frap_y = None
        self.app.offset_x = self.app.offset_y = None
        self.app._refresh_params()
        self.status_var.set("点 #1：大致范围左上角")
        self.app.log("  自动检测：已重置框选。")

    def _finish(self):
        if self.app.frap_x is None:
            messagebox.showwarning("未检测", "请先框选并完成自动检测。")
            return
        self.app.log("  自动检测完成，参数已写入，窗口关闭。")
        self.parent_win.destroy()
        self.destroy()

# ─────────────────────────────────────────────
#  Window 3: Image Preview
# ─────────────────────────────────────────────

class PreviewWindow(tk.Toplevel):
    """
    窗口3：在整帧上叠加 ROI（黄色虚线）和 FRAP 圆（蓝色实线）。
    若启用了背景 ROI，同时叠加绿色背景 ROI 框 + 背景 FRAP 圆。
    ROI Focus / Whole Frame 切换。
    """
    _MAX = 700
    ZOOM = 4

    def __init__(self, app: AutoFRAPApp, filepath: str):
        super().__init__(app)
        self.app = app
        self.filepath = filepath
        self.title("图像预览")
        self.configure(bg="#1a1a2e")
        self.resizable(False, False)

        self.pil_full = open_as_rgb(filepath)
        self._in_roi_mode = False

        # Compute current frame's FRAP position for overlay
        gray = load_image_as_gray(filepath)
        res = calc_frap_intensity(gray,
                                  app.corner1_x, app.corner1_y,
                                  app.corner2_x, app.corner2_y,
                                  app.offset_x, app.offset_y,
                                  app.frap_radius.get())
        self._frap_global_x = res["frap_x_global"]
        self._frap_global_y = res["frap_y_global"]
        self._dcx_roi = res["droplet_center_x"]
        self._dcy_roi = res["droplet_center_y"]
        self._frap_x_roi = res["frap_x_roi"]
        self._frap_y_roi = res["frap_y_roi"]

        # 背景值（若启用）
        self._has_bg = (app.use_bg.get()
                        and app.bg_corner1_x is not None
                        and app.bg_corner1_y is not None)
        if self._has_bg:
            bg_res = calc_bg_intensity(gray,
                                       app.bg_corner1_x, app.bg_corner1_y,
                                       self._frap_x_roi, self._frap_y_roi,
                                       app.frap_radius.get())
            self._bg_frap_global_x = bg_res["bg_frap_x_global"]
            self._bg_frap_global_y = bg_res["bg_frap_y_global"]
            app.log(f"预览帧  I={res['FRAP_intensity']:.4f}  "
                    f"bg={bg_res['FRAP_bg_intensity']:.4f}  "
                    f"corrected={res['FRAP_intensity'] - bg_res['FRAP_bg_intensity']:.4f}")
        else:
            self._bg_frap_global_x = None
            self._bg_frap_global_y = None
            app.log(f"预览帧  intensity={res['FRAP_intensity']:.4f}  "
                    f"质心ROI=({self._dcx_roi:.2f},{self._dcy_roi:.2f})")

        self._build()
        self._show_full()

    def _annotated_full(self) -> Image.Image:
        img = self.pil_full.copy()
        draw = ImageDraw.Draw(img)
        cx1, cy1 = self.app.corner1_x, self.app.corner1_y
        cx2, cy2 = self.app.corner2_x, self.app.corner2_y
        r = self.app.frap_radius.get()

        # ROI: yellow dashed rectangle
        dash = 8
        for x in range(cx1, cx2, dash * 2):
            draw.line([(x, cy1), (min(x + dash, cx2), cy1)],
                      fill="#f0e040", width=2)
            draw.line([(x, cy2), (min(x + dash, cx2), cy2)],
                      fill="#f0e040", width=2)
        for y in range(cy1, cy2, dash * 2):
            draw.line([(cx1, y), (cx1, min(y + dash, cy2))],
                      fill="#f0e040", width=2)
            draw.line([(cx2, y), (cx2, min(y + dash, cy2))],
                      fill="#f0e040", width=2)

        # FRAP circle: blue solid
        fx, fy = self._frap_global_x, self._frap_global_y
        draw.ellipse([(fx - r, fy - r), (fx + r, fy + r)],
                     outline="#4fc3f7", width=2)

        # 背景 ROI + 背景 FRAP 圆
        if self._has_bg:
            bx1, by1 = self.app.bg_corner1_x, self.app.bg_corner1_y
            bx2, by2 = self.app.bg_corner2()
            for x in range(bx1, bx2, dash * 2):
                draw.line([(x, by1), (min(x + dash, bx2), by1)],
                          fill="#2ecc71", width=2)
                draw.line([(x, by2), (min(x + dash, bx2), by2)],
                          fill="#2ecc71", width=2)
            for y in range(by1, by2, dash * 2):
                draw.line([(bx1, y), (bx1, min(y + dash, by2))],
                          fill="#2ecc71", width=2)
                draw.line([(bx2, y), (bx2, min(y + dash, by2))],
                          fill="#2ecc71", width=2)
            bfx, bfy = self._bg_frap_global_x, self._bg_frap_global_y
            draw.ellipse([(bfx - r, bfy - r), (bfx + r, bfy + r)],
                         outline="#2ecc71", width=2)

        return img

    def _annotated_roi(self) -> Image.Image:
        cx1, cy1 = self.app.corner1_x, self.app.corner1_y
        cx2, cy2 = self.app.corner2_x, self.app.corner2_y
        roi_pil = self.pil_full.crop((cx1, cy1, cx2, cy2))
        rw, rh = roi_pil.size
        zoomed = roi_pil.resize((rw * self.ZOOM, rh * self.ZOOM), Image.LANCZOS)
        draw = ImageDraw.Draw(zoomed)

        # FRAP circle in ROI coords × ZOOM
        fx_r = (self._frap_global_x - cx1) * self.ZOOM
        fy_r = (self._frap_global_y - cy1) * self.ZOOM
        r = self.app.frap_radius.get() * self.ZOOM
        draw.ellipse([(fx_r - r, fy_r - r), (fx_r + r, fy_r + r)],
                     outline="#4fc3f7", width=2)

        # Centroid marker
        ccx = self._dcx_roi * self.ZOOM
        ccy = self._dcy_roi * self.ZOOM
        draw.ellipse([(ccx - 5, ccy - 5), (ccx + 5, ccy + 5)],
                     outline="#f0e040", width=2)

        # 背景 FRAP 圆（位置一致：在主 ROI 内的相对坐标即为其在背景 ROI 内的相对坐标）
        if self._has_bg:
            draw.ellipse([(fx_r - r, fy_r - r), (fx_r + r, fy_r + r)],
                         outline="#2ecc71", width=2)

        return zoomed

    def _resize_for_display(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        scale = min(self._MAX / w, self._MAX / h, 1.0)
        if scale < 1.0:
            img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        return img

    def _build(self):
        BG = "#1a1a2e"
        MUTED = "#8892a4"

        hint = "预览 — ROI（黄色虚线）· FRAP 圆（蓝色实线）"
        if self._has_bg:
            hint += " · 背景 ROI/圆（绿色）"
        tk.Label(self, text=hint,
                 font=("Courier New", 9), fg=MUTED, bg=BG).pack(pady=(8, 2))

        self.canvas = tk.Canvas(self, bg="#0d0d1a", relief="flat", bd=0)
        self.canvas.pack(padx=10)

        self.toggle_btn = tk.Button(self, text="ROI Focus",
                                    font=("Courier New", 10, "bold"),
                                    bg="#0f3460", fg="#eaeaea",
                                    activebackground="#1a4a80",
                                    relief="flat", padx=16, pady=7,
                                    command=self._toggle)
        self.toggle_btn.pack(pady=(6, 12))

    def _show_full(self):
        img = self._resize_for_display(self._annotated_full())
        self._tk_img = ImageTk.PhotoImage(img)
        self.canvas.config(width=img.width, height=img.height)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=self._tk_img)
        self._in_roi_mode = False
        self.toggle_btn.config(text="ROI Focus")

    def _show_roi(self):
        img = self._resize_for_display(self._annotated_roi())
        self._tk_img = ImageTk.PhotoImage(img)
        self.canvas.config(width=img.width, height=img.height)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=self._tk_img)
        self._in_roi_mode = True
        self.toggle_btn.config(text="Whole Frame")

    def _toggle(self):
        if self._in_roi_mode:
            self._show_full()
        else:
            self._show_roi()


# ─────────────────────────────────────────────
#  Window 4: Result Panel (figures)  —— 功能2
# ─────────────────────────────────────────────

class ResultPanelWindow(tk.Toplevel):
    """
    结果图表面板：读取 FRAP_results.csv，在一个窗口中绘制：
      a) 归一化 intensity (%) vs 帧编号
         （若 CSV 含 FRAP_intensity_corrected_norm，则同时绘制扣背景曲线）
      b) droplet 移动轨迹（center_x vs center_y，按帧着色，标出起止点）
      c) droplet_center_x vs 帧编号
      d) droplet_center_y vs 帧编号
    使用 matplotlib，嵌入 tkinter，带缩放/保存工具栏。
    """

    # 配色（与主程序暗色主题一致）
    _BG = "#16213e"
    _AX_BG = "#0d0d1a"
    _GRID = "#1f2b4a"
    _SPINE = "#0f3460"
    _TITLE = "#e94560"
    _LABEL = "#8892a4"

    def __init__(self, app: AutoFRAPApp, csv_path: str):
        super().__init__(app)
        self.app = app
        self.title("结果图表 — FRAP Results Panel")
        self.configure(bg="#1a1a2e")

        data = load_frap_results(csv_path)
        frames = data["frame_index"]
        norm = data["FRAP_intensity_norm"]
        corr_norm = data["FRAP_intensity_corrected_norm"]   # 可能为 None
        ratio_norm = data["FRAP_intensity_ratio_norm"]      # 可能为 None
        cx = data["droplet_center_x"]
        cy = data["droplet_center_y"]
        n = len(frames)
        extras = []
        if corr_norm is not None:
            extras.append("减法")
        if ratio_norm is not None:
            extras.append("除法")
        self.app.log(f"  读取 {n} 帧数据，开始绘图"
                     + (f"（含{'/'.join(extras)}校正曲线）。" if extras else "。"))

        fig = Figure(figsize=(11, 7.6), dpi=100, facecolor=self._BG)

        # ── a) 归一化 intensity vs frame ──────────────────────
        ax1 = fig.add_subplot(2, 2, 1)
        ax1.plot(frames, norm, "-o", color="#4fc3f7",
                 markersize=3, linewidth=1.3, label="raw")
        if corr_norm is not None:
            ax1.plot(frames, corr_norm, "-s", color="#2ecc71",
                     markersize=3, linewidth=1.3,
                     label="without bg (subtract)")
        if ratio_norm is not None:
            ax1.plot(frames, ratio_norm, "-^", color="#f39c12",
                     markersize=3, linewidth=1.3,
                     label="ratio (FRAP / bg)")
        ax1.axhline(100, color="#f0e040", linestyle="--", linewidth=0.9,
                    label="100% (baseline)")
        ax1.axhline(0, color="#e94560", linestyle="--", linewidth=0.9,
                    label="0% (bleach)")
        self._style_ax(ax1, "Normalized FRAP Intensity",
                       "Frame Index", "Intensity (%)")
        leg = ax1.legend(loc="best", fontsize=7, framealpha=0.25)
        for t in leg.get_texts():
            t.set_color("#eaeaea")

        # ── b) droplet 移动轨迹（x vs y，按帧着色）────────────
        ax2 = fig.add_subplot(2, 2, 2)
        ax2.plot(cx, cy, "-", color="#8892a4", linewidth=0.7, alpha=0.6)
        sc = ax2.scatter(cx, cy, c=frames, cmap="viridis", s=14, zorder=3)
        if n > 0:
            ax2.scatter([cx[0]], [cy[0]], color="#2ecc71", s=55,
                        marker="o", edgecolors="white", linewidths=0.6,
                        zorder=4, label="start")
            ax2.scatter([cx[-1]], [cy[-1]], color="#e94560", s=70,
                        marker="X", edgecolors="white", linewidths=0.6,
                        zorder=4, label="end")
        ax2.invert_yaxis()   # 图像坐标系 y 向下，轨迹方向与图像一致
        self._style_ax(ax2, "Droplet Movement (trajectory)",
                       "center_x (px)", "center_y (px)")
        leg2 = ax2.legend(loc="best", fontsize=7, framealpha=0.25)
        for t in leg2.get_texts():
            t.set_color("#eaeaea")
        cbar = fig.colorbar(sc, ax=ax2, fraction=0.046, pad=0.04)
        cbar.set_label("frame", color=self._LABEL, fontsize=7)
        cbar.ax.tick_params(colors=self._LABEL, labelsize=6)
        cbar.outline.set_edgecolor(self._SPINE)

        # ── c) center_x vs frame ──────────────────────────────
        ax3 = fig.add_subplot(2, 2, 3)
        ax3.plot(frames, cx, "-o", color="#2ecc71",
                 markersize=2.6, linewidth=1.0)
        self._style_ax(ax3, "Droplet Center X",
                       "Frame Index", "center_x (px)")

        # ── d) center_y vs frame ──────────────────────────────
        ax4 = fig.add_subplot(2, 2, 4)
        ax4.plot(frames, cy, "-o", color="#f39c12",
                 markersize=2.6, linewidth=1.0)
        self._style_ax(ax4, "Droplet Center Y",
                       "Frame Index", "center_y (px)")

        fig.tight_layout(pad=2.0)

        # ── 嵌入 tkinter ──────────────────────────────────────
        canvas = FigureCanvasTkAgg(fig, master=self)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True, padx=8, pady=(8, 2))

        toolbar_frame = tk.Frame(self, bg="#1a1a2e")
        toolbar_frame.pack(fill="x", padx=8, pady=(0, 8))
        toolbar = NavigationToolbar2Tk(canvas, toolbar_frame)
        toolbar.update()

    def _style_ax(self, ax, title, xlabel, ylabel):
        ax.set_facecolor(self._AX_BG)
        ax.set_title(title, color=self._TITLE, fontsize=10, pad=6)
        ax.set_xlabel(xlabel, color=self._LABEL, fontsize=8)
        ax.set_ylabel(ylabel, color=self._LABEL, fontsize=8)
        ax.tick_params(colors=self._LABEL, labelsize=7)
        for s in ax.spines.values():
            s.set_color(self._SPINE)
        ax.grid(True, color=self._GRID, linewidth=0.5)


# ─────────────────────────────────────────────
#  Entry Point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    app = AutoFRAPApp()
    app.mainloop()
