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


# ─────────────────────────────────────────────
#  Part 1: Core Functions
# ─────────────────────────────────────────────

def load_image_as_gray(filepath: str) -> np.ndarray:
    """格式转换：将 jpg/jpeg 图片转为灰度 numpy 数组。"""
    img = Image.open(filepath).convert("L")
    return np.array(img, dtype=np.float64)


def get_roi(gray: np.ndarray, cx1: int, cy1: int, cx2: int, cy2: int) -> np.ndarray:
    """ROI 裁切：返回 ROI 区域的灰度数组（行=y，列=x）。"""
    return gray[cy1:cy2, cx1:cx2]


def calc_centroid(roi: np.ndarray):
    """
    Droplet 质心计算（亮度加权重心）。
    返回 (droplet_center_x, droplet_center_y)，坐标参考系为 ROI 区域。
    """
    total = roi.sum()
    if total == 0:
        h, w = roi.shape
        return w / 2.0, h / 2.0
    ys, xs = np.mgrid[0:roi.shape[0], 0:roi.shape[1]]
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


def calc_frap_intensity(gray: np.ndarray,
                        cx1: int, cy1: int, cx2: int, cy2: int,
                        offset_x: float, offset_y: float,
                        frap_radius: float) -> dict:
    """
    荧光强度计算：
      1. 计算 ROI 内的质心
      2. 加上偏移量得到本帧的 FRAP 圆心（ROI 坐标系）
      3. 转为全图坐标后，在 frap_radius 范围内求均值
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

    # 在全图中计算圆内像素均值
    h, w = gray.shape
    ys, xs = np.mgrid[0:h, 0:w]
    mask = (xs - fx_global) ** 2 + (ys - fy_global) ** 2 <= frap_radius ** 2
    pixels = gray[mask]
    intensity = float(pixels.mean()) if pixels.size > 0 else 0.0

    return {
        "droplet_center_x": dcx,
        "droplet_center_y": dcy,
        "frap_x_roi": fx_roi,
        "frap_y_roi": fy_roi,
        "frap_x_global": fx_global,
        "frap_y_global": fy_global,
        "FRAP_intensity": intensity,
        "pixel_count": int(pixels.size),
    }


def annotate_frame(filepath: str,
                   cx1: int, cy1: int, cx2: int, cy2: int,
                   frap_global_x: float, frap_global_y: float,
                   frap_radius: float,
                   dcx_roi: float, dcy_roi: float) -> Image.Image:
    """
    在原始彩色图片上绘制标注并返回 PIL Image：
      · ROI         : 黄色虚线矩形
      · Droplet 质心: 黄色十字 + 小圆（ROI 坐标 → 全图坐标）
      · FRAP 圆     : 蓝色实线圆 + 圆心点
    """
    img = Image.open(filepath).convert("RGB")
    draw = ImageDraw.Draw(img)

    # ROI 黄色虚线矩形
    dash = 10
    for x in range(cx1, cx2, dash * 2):
        draw.line([(x, cy1), (min(x + dash, cx2), cy1)], fill="#f0e040", width=2)
        draw.line([(x, cy2), (min(x + dash, cx2), cy2)], fill="#f0e040", width=2)
    for y in range(cy1, cy2, dash * 2):
        draw.line([(cx1, y), (cx1, min(y + dash, cy2))], fill="#f0e040", width=2)
        draw.line([(cx2, y), (cx2, min(y + dash, cy2))], fill="#f0e040", width=2)

    # Droplet 质心：黄色十字 + 小圆（ROI → 全图坐标）
    gx = dcx_roi + cx1
    gy = dcy_roi + cy1
    cr, arm = 5, 10
    # draw.ellipse([(gx - cr, gy - cr), (gx + cr, gy + cr)],
    #              outline="#f0e040", width=1)
    draw.line([(gx - arm, gy), (gx + arm, gy)], fill="#f0e040", width=1)
    draw.line([(gx, gy - arm), (gx, gy + arm)], fill="#f0e040", width=1)

    # FRAP 圆：蓝色实线
    fx, fy, r = frap_global_x, frap_global_y, frap_radius
    draw.ellipse([(fx - r, fy - r), (fx + r, fy + r)],
                 outline="#4fc3f7", width=2)
    draw.ellipse([(fx - 2, fy - 2), (fx + 2, fy + 2)], fill="#4fc3f7")

    return img


def batch_analysis(folder: str,
                   cx1: int, cy1: int, cx2: int, cy2: int,
                   offset_x: float, offset_y: float,
                   frap_radius: float,
                   marked_dir: str = None,
                   progress_cb=None,
                   frame_cb=None) -> list:
    """
    批量数据分析：对文件夹内所有 jpg/jpeg 按文件名排序后逐帧计算。
    marked_dir 不为 None 时，将标注后的图片（ROI虚线框、质心、FRAP圆）保存到该目录。
    progress_cb(i, total, filename) 可用于更新进度。
    frame_cb(row) 每帧完成后立即回调。
    返回列表，每项为字典（含 filename + calc_frap_intensity 返回的字段）。
    """
    patterns = ["*.jpg", "*.jpeg", "*.JPG", "*.JPEG"]
    files = []
    for p in patterns:
        files.extend(glob.glob(os.path.join(folder, p)))
    files = sorted(set(files))

    if marked_dir:
        os.makedirs(marked_dir, exist_ok=True)

    results = []
    total = len(files)
    for i, fp in enumerate(files):
        gray = load_image_as_gray(fp)
        row = calc_frap_intensity(gray, cx1, cy1, cx2, cy2,
                                  offset_x, offset_y, frap_radius)
        row["filename"] = os.path.basename(fp)
        row["frame_index"] = i
        results.append(row)

        # 保存标注图片
        if marked_dir:
            marked_img = annotate_frame(
                fp, cx1, cy1, cx2, cy2,
                row["frap_x_global"], row["frap_y_global"],
                frap_radius,
                row["droplet_center_x"], row["droplet_center_y"],
            )
            base, _ = os.path.splitext(os.path.basename(fp))
            marked_img.save(os.path.join(marked_dir, f"{base}_marked.jpg"),
                            "JPEG", quality=95)

        if frame_cb:
            frame_cb(row)
        if progress_cb:
            progress_cb(i + 1, total, os.path.basename(fp))
    return results


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

        # ROI corners (full-image pixel coords)
        self.corner1_x = None
        self.corner1_y = None
        self.corner2_x = None
        self.corner2_y = None

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

        # FRAP radius sub-row
        rad_frame = tk.Frame(self, bg=CARD, padx=8, pady=4)
        rad_frame.pack(fill="x", padx=PAD, pady=(0, 3))
        tk.Label(rad_frame, text="FRAP RADIUS (px)", font=FONT_LABEL,
                 fg=MUTED, bg=CARD, width=18, anchor="w").pack(side="left")
        tk.Entry(rad_frame, textvariable=self.frap_radius,
                 font=FONT_LABEL, bg=ACCENT, fg=TEXT,
                 insertbackground=TEXT, relief="flat", width=8).pack(side="left", padx=6)

        # ── Row 2: Action buttons ────────────────────────────────
        btn_frame = tk.Frame(self, bg=BG, pady=6)
        btn_frame.pack(fill="x", padx=PAD)

        btn_style = dict(font=FONT_BTN, relief="flat", padx=18, pady=8,
                         cursor="hand2", activeforeground="white")
        tk.Button(btn_frame, text="⊕  偏移量计算",
                  bg=ACCENT, fg=TEXT, activebackground="#1a4a80",
                  command=self._run_offset_calc, **btn_style).pack(side="left", padx=4)
        tk.Button(btn_frame, text="◎  图像预览",
                  bg=ACCENT, fg=TEXT, activebackground="#1a4a80",
                  command=self._run_preview, **btn_style).pack(side="left", padx=4)
        tk.Button(btn_frame, text="▶  全数据分析",
                  bg=HIGHLIGHT, fg="white", activebackground="#c73652",
                  command=self._run_batch, **btn_style).pack(side="left", padx=4)

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
            ("offset_x", "offset_y", "", ""),
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
            filetypes=[("JPEG images", "*.jpg *.jpeg *.JPG *.JPEG")])
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
        self.log(f"偏移量计算 — 参考帧：{os.path.basename(fp)}")
        OffsetWindow(self, fp)

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

        radius = self.frap_radius.get()
        self.log(f"全数据分析开始 — 文件夹：{folder}，FRAP半径：{radius} px")

        # 用一个可变容器暂存每帧结果，供 progress_cb 读取
        _partial: list = []

        def _progress(i, t, fn):
            if _partial:
                last = _partial[-1]
                self.log(f"  [{i}/{t}] {fn}  intensity={last['FRAP_intensity']:.4f}")
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
            )
        except Exception as e:
            messagebox.showerror("分析错误", str(e))
            self.log(f"错误：{e}")
            return

        # Save CSV
        out_path = os.path.join(folder, "FRAP_results.csv")
        fields = ["frame_index", "filename",
                  "droplet_center_x", "droplet_center_y",
                  "frap_x_roi", "frap_y_roi",
                  "frap_x_global", "frap_y_global",
                  "FRAP_intensity", "pixel_count"]
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
            f"CSV 结果:\n{out_path}\n\n"
            f"标注图片:\n{marked_dir}/"
        )


# ─────────────────────────────────────────────
#  Window 1 & 2: Offset Calculation Workflow
# ─────────────────────────────────────────────

class OffsetWindow(tk.Toplevel):
    """
    窗口1：展示整帧图片，用户依次点击两个角确定 ROI。
    """
    _CANVAS_MAX = 700  # canvas 最大显示尺寸

    def __init__(self, app: AutoFRAPApp, filepath: str):
        super().__init__(app)
        self.app = app
        self.filepath = filepath
        self.title("Step 1 — 选择 ROI")
        self.configure(bg="#1a1a2e")
        self.resizable(False, False)

        # Load image
        self.pil_img = Image.open(filepath).convert("RGB")
        self.orig_w, self.orig_h = self.pil_img.size
        self.scale = min(self._CANVAS_MAX / self.orig_w,
                         self._CANVAS_MAX / self.orig_h, 1.0)
        dw = int(self.orig_w * self.scale)
        dh = int(self.orig_h * self.scale)
        self.display_img = self.pil_img.resize((dw, dh), Image.LANCZOS)

        self._click_pts = []  # up to 2 canvas coords
        self._rect_id = None

        self._build()

    def _build(self):
        BG = "#1a1a2e"
        CARD = "#16213e"
        ACCENT = "#0f3460"
        HIGHLIGHT = "#e94560"
        TEXT = "#eaeaea"
        MUTED = "#8892a4"

        tk.Label(self, text="左键单击依次选定 ROI 左上角、右下角",
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
        if len(self._click_pts) >= 2:
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
        else:
            self.app.corner2_x = ox
            self.app.corner2_y = oy
            self.status_var.set("ROI 已选定，单击 Continue 继续。")
            x1c, y1c = self._click_pts[0]
            x2c, y2c = self._click_pts[1]
            if self._rect_id:
                self.canvas.delete(self._rect_id)
            self._rect_id = self.canvas.create_rectangle(
                x1c, y1c, x2c, y2c,
                outline="#f0e040", dash=(4, 4), width=2)

        self.app._refresh_params()
        self.app.log(f"  ROI 角点 #{len(self._click_pts)} 选定：原图 ({ox}, {oy})")

    def _reset(self):
        self._click_pts.clear()
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=self._tk_img)
        self.app.corner1_x = self.app.corner1_y = None
        self.app.corner2_x = self.app.corner2_y = None
        self.app._refresh_params()
        self.status_var.set("点 #1：ROI 左上角")
        self.app.log("  ROI 坐标已重置。")

    def _continue(self):
        if None in (self.app.corner1_x, self.app.corner1_y,
                    self.app.corner2_x, self.app.corner2_y):
            messagebox.showwarning("未完成", "请先选定两个角点。")
            return

        # Normalise so corner1 is always top-left
        x1 = min(self.app.corner1_x, self.app.corner2_x)
        y1 = min(self.app.corner1_y, self.app.corner2_y)
        x2 = max(self.app.corner1_x, self.app.corner2_x)
        y2 = max(self.app.corner1_y, self.app.corner2_y)
        self.app.corner1_x, self.app.corner1_y = x1, y1
        self.app.corner2_x, self.app.corner2_y = x2, y2
        self.app._refresh_params()

        FRAPPickerWindow(self.app, self.filepath, self)


class FRAPPickerWindow(tk.Toplevel):
    """
    窗口2：展示 ROI 区域（4× 放大），用户点击选定 FRAP 圆心。
    坐标参考系为 ROI 内坐标。
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
        pil = Image.open(filepath).convert("RGB")
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
        self._build()

    def _build(self):
        BG = "#1a1a2e"
        CARD = "#16213e"
        ACCENT = "#0f3460"
        TEXT = "#eaeaea"
        MUTED = "#8892a4"

        tk.Label(self, text=f"ROI 区域（{self.ZOOM}× 放大）— 左键单击选定 FRAP 圆心",
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
        self.status_var.set("请单击选定 FRAP 圆心")
        self.app.log("  FRAP 参数已重置。")

    def _finish(self):
        if self.app.frap_x is None:
            messagebox.showwarning("未选定", "请先单击选定 FRAP 圆心。")
            return
        self.app.log("  偏移量计算完成。窗口已关闭。")
        self.parent_win.destroy()
        self.destroy()


# ─────────────────────────────────────────────
#  Window 3: Image Preview
# ─────────────────────────────────────────────

class PreviewWindow(tk.Toplevel):
    """
    窗口3：在整帧上叠加 ROI（黄色虚线）和 FRAP 圆（蓝色实线）。
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

        self.pil_full = Image.open(filepath).convert("RGB")
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

        tk.Label(self, text="预览 — ROI（黄色虚线）· FRAP 圆（蓝色实线）",
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
#  Entry Point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    app = AutoFRAPApp()
    app.mainloop()