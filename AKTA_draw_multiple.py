import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from charset_normalizer import detect

# ────────────────────────────────────────────────
#  Global settings
# ────────────────────────────────────────────────
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.unicode_minus'] = False
plt.rc('xtick', labelsize=36)
plt.rc('ytick', labelsize=36)



LINE_THICKNESS = 4
LABEL_FONT_SIZE = 36
FRACTION_FONT_SIZE = 12
LEGEND_FONT_SIZE = 36

COLOR_CYCLE = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
]


# ────────────────────────────────────────────────
#  Helper functions
# ────────────────────────────────────────────────

def detect_encoding(file_path):
    with open(file_path, 'rb') as f:
        result = detect(f.read())
        encoding = result['encoding']
        confidence = result['confidence']
        print(f"Detected encoding: {encoding} (Confidence: {confidence:.2%})")
        return encoding


def convert_to_utf8(input_file, output_file):
    try:
        encoding = detect_encoding(input_file) or 'utf-8'
        with open(input_file, 'r', encoding=encoding) as f:
            content = f.read()
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"File converted to UTF-8: {output_file}")
    except UnicodeDecodeError:
        print(f"Error: Cannot decode file with {encoding}")
    except Exception as e:
        print(f"Conversion error: {str(e)}")


def curve_find(x, df):
    where = (df == x)
    if where.sum().sum() == 1:
        return tuple(where.stack().idxmax())
    else:
        raise Exception(f"{where.sum().sum()} curves found for '{x}'!")


def round_up_to_second_sig_digit(x):
    if x == 0:
        return 0
    magnitude = math.floor(math.log10(abs(x)))
    scaled = x / (10 ** magnitude)
    rounded_scaled = math.ceil(scaled * 10) / 10
    return rounded_scaled * (10 ** magnitude)


def in_range(x, boundaries):
    lower, upper = boundaries
    return lower <= x <= upper


# ────────────────────────────────────────────────
#  Data reading
# ────────────────────────────────────────────────

def read_unicorn_curves(
    filepath,
    baseline_start_ml=2.0,
    baseline_end_ml=3.0,
    do_baseline_correction=True
):
    print(f"\nReading: {os.path.basename(filepath)}")
    if do_baseline_correction:
        print(f"  Baseline correction: average between {baseline_start_ml}–{baseline_end_ml} mL")
    else:
        print("  Baseline correction DISABLED")

    # ── Read header and data ─────────────────────────────────────
    curve_names = pd.read_csv(filepath, sep='\t', nrows=1)
    data = pd.read_csv(filepath, sep='\t', skiprows=2)

    curve_names.columns = range(len(curve_names.columns))
    data.columns = range(len(data.columns))

    # ── Locate columns ───────────────────────────────────────────
    try:
        UV_col = curve_find('UV 1_280', curve_names)
    except:
        UV_col = curve_find('UV', curve_names)

    try:
        UV_260_col = curve_find('UV 2_260', curve_names)
    except:
        UV_260_col = None

    Cond_col     = curve_find('Cond', curve_names)
    Fraction_col = curve_find('Fraction', curve_names)

    print(f"  UV       → {UV_col}")
    print(f"  Cond     → {Cond_col}")
    print(f"  Fraction → {Fraction_col}\n")

    # ── Fraction marker processing ───────────────────────────────
    frac_col_idx = Fraction_col[1] + 1
    fracx_1 = data[frac_col_idx].dropna()
    fracx_1 = fracx_1[fracx_1 != "Waste"].reset_index(drop=True)
    fracx_1 = fracx_1.replace("Frac", 2)           # handle manual Frac notation
    frac_numbers = fracx_1.astype(int)

    # ── Extract UV and Cond data ─────────────────────────────────
    uv_key = 'UV' if 'UV' in UV_col else 'UV 1_280'

    uv_ml   = data[UV_col[1]].astype(float)
    uv_mau  = data[UV_col[1] + 1].astype(float)
    cond_ml = data[Cond_col[1]].astype(float)
    cond_ms = data[Cond_col[1] + 1].astype(float)

    baseline_value = 0.0

    # ── Optional baseline correction ─────────────────────────────
    if do_baseline_correction:
        mask = (uv_ml >= baseline_start_ml) & (uv_ml <= baseline_end_ml)

        if mask.sum() >= 5:
            baseline_value = np.nanmean(uv_mau[mask])
            print(f"  Per-curve baseline: {baseline_value:8.2f} mAU  ({mask.sum()} points)")
        else:
            baseline_value = np.nanmean(uv_mau.iloc[:20])
            print(f"  Warning: few points in window → used first 20 pts → {baseline_value:.2f} mAU")

        if np.isnan(baseline_value):
            baseline_value = 0.0
            print("  Warning: baseline NaN → set to 0")

    uv_mau_corrected = uv_mau - baseline_value

    # ── Build result dictionary ──────────────────────────────────
    result = {
        'UV': {
            'mL': uv_ml,
            'mAU': uv_mau_corrected,
            'baseline_used': baseline_value          # for global alignment later
        },
        'Cond': {
            'mL': cond_ml,
            'mS/cm': cond_ms
        },
        'Fraction': {
            'mL': data[Fraction_col[1]].astype(float),
            'Fraction': frac_numbers                 # ← now correctly defined
        }
    }

    if UV_260_col is not None:
        result['UV260'] = {
            'mL': data[UV_260_col[1]].astype(float),
            'mAU': data[UV_260_col[1] + 1].astype(float)
        }

    return result

# ────────────────────────────────────────────────
#  Multi-curve plotting
# ────────────────────────────────────────────────

def draw_multiple_UV_Cond_Fracx(
    curves_list,
    labels=None,
    title="Multiple AKTA runs",
    elution=None,
    UV=None,
    Cond=None,
    show_cond=True,
    save=False,
    save_path="multi_comparison.png",
    dpi=300,
    fraction_colors=None
):
    """
    Plot multiple AKTA UV curves (with baseline alignment) + optional Cond curves.
    Always keeps full figure outline (all spines visible).
    """
    if not curves_list:
        raise ValueError("curves_list cannot be empty")

    n = len(curves_list)
    if labels is None:
        labels = [f"Run {i+1}" for i in range(n)]
    if len(labels) != n:
        raise ValueError("labels length does not match curves_list")

    if fraction_colors is None:
        fraction_colors = COLOR_CYCLE[:n]

    # ── Create figure and axes ───────────────────────────────────
    fig, ax1 = plt.subplots(figsize=(16, 9))

    # Create twin axis only if Cond is requested
    ax2 = ax1.twinx() if show_cond else None

    # ── Always show full outline (box) on ax1 ────────────────────
    for spine in ax1.spines.values():
        spine.set_visible(True)

    # If Cond is shown, also show full outline on ax2
    if show_cond and ax2 is not None:
        for spine in ax2.spines.values():
            spine.set_visible(True)

    # ── Auto x-range ─────────────────────────────────────────────
    if elution is None:
        all_ml = []
        for c in curves_list:
            k = next((k for k in ['UV', 'UV 1_280', 'UV280'] if k in c), None)
            if k:
                all_ml.append(c[k]['mL'])
        if all_ml:
            elution = (min(m.min() for m in all_ml) - 5,
                       max(m.max() for m in all_ml) + 5)
        else:
            elution = (-20, 400)

    ax1.set_xlim(elution)

    # ── Collect baselines & compute global shift ─────────────────
    all_baselines = []
    for curves in curves_list:
        uv_key = next((k for k in ['UV', 'UV 1_280', 'UV280'] if k in curves), None)
        if uv_key and 'baseline_used' in curves[uv_key]:
            all_baselines.append(curves[uv_key]['baseline_used'])

    global_shift = -min(all_baselines) if all_baselines else 0
    if all_baselines:
        print(f"Global shift applied: {global_shift:.2f} mAU (lowest baseline → 0)")

    # ── Prepare y-limits ─────────────────────────────────────────
    all_shifted_uv = []
    all_cond_max = []

    for curves in curves_list:
        uv_key = next((k for k in ['UV', 'UV 1_280', 'UV280'] if k in curves), None)
        if uv_key:
            all_shifted_uv.append(curves[uv_key]['mAU'] + global_shift)
        if show_cond and 'Cond' in curves:
            all_cond_max.append(curves['Cond']['mS/cm'].max())

    # UV limit
    if UV is None and all_shifted_uv:
        flat = np.concatenate(all_shifted_uv)
        ymin, ymax = np.nanmin(flat), np.nanmax(flat)
        margin = (ymax - ymin) * 0.12
        UV = (ymin - margin, ymax + margin * 1.5)
    elif UV is None:
        UV = (-10, 120)

    ax1.set_ylim(UV)

    # Cond limit
    if show_cond:
        if Cond is None:
            cond_max = max(all_cond_max) * 1.12 if all_cond_max else 100
            Cond = (0, cond_max)
        ax2.set_ylim(Cond)

    # ── Ticks ────────────────────────────────────────────────────
    ax1.xaxis.set_minor_locator(ticker.AutoMinorLocator(4))
    ax1.yaxis.set_major_locator(ticker.MultipleLocator(round_up_to_second_sig_digit(UV[1]/5)))
    # ax1.yaxis.set_minor_locator(ticker.AutoMinorLocator(4))

    if show_cond:
        ax2.yaxis.set_major_locator(ticker.MultipleLocator(round_up_to_second_sig_digit(Cond[1]/5)))
        ax2.yaxis.set_minor_locator(ticker.AutoMinorLocator(4))

    plt.tick_params(axis='both', which='major', width=2, length=8)
    plt.tick_params(axis='both', which='minor', width=1, length=4)

    # ── Remove the largest tick on left y-axis ───────────────────
    yticks = ax1.get_yticks()
    if len(yticks) >= 2:
        ax1.set_yticks(yticks[1:-2])           # drop the last (largest) one

    # Optional: same for right axis
    if show_cond and ax2 is not None:
        yticks_right = ax2.get_yticks()
        if len(yticks_right) >= 2:
            ax2.set_yticks(yticks_right[1:-2])

    # ── Labels ───────────────────────────────────────────────────
    ax1.set_title(title, fontsize=LABEL_FONT_SIZE + 4, pad=20)
    ax1.set_xlabel('Elution / mL', fontsize=LABEL_FONT_SIZE, fontweight='bold')

    ax1.text(0.06,   0.90, 'mAU',   transform=ax1.transAxes, fontsize=LABEL_FONT_SIZE, ha='center', va='bottom')
    if show_cond:
        ax2.text(1.0, 1.06, 'mS/cm', transform=ax2.transAxes, fontsize=LABEL_FONT_SIZE, ha='center', va='bottom')

    # ── Plot curves ──────────────────────────────────────────────
    handles = []

    for i, curves in enumerate(curves_list):
        color = COLOR_CYCLE[i % len(COLOR_CYCLE)]

        uv_key = next((k for k in ['UV', 'UV 1_280', 'UV280'] if k in curves), None)
        if not uv_key:
            continue

        uv_ml  = curves[uv_key]['mL']
        uv_mau = curves[uv_key]['mAU'] + global_shift

        h, = ax1.plot(uv_ml, uv_mau, color=color, lw=LINE_THICKNESS, label=labels[i])
        handles.append(h)

        # Cond curve
        if show_cond and 'Cond' in curves and ax2 is not None:
            ax2.plot(
                curves['Cond']['mL'],
                curves['Cond']['mS/cm'],
                color=color,
                lw=LINE_THICKNESS * 0.75,
                alpha=0.85,
                linestyle='--'
            )

        # Fraction markers: turned off manually
        if 'Fraction' in curves:
            frac_ml  = curves['Fraction']['mL']
            frac_num = curves['Fraction']['Fraction']
            fcolor   = fraction_colors

            for vol, fnum in zip(frac_ml, frac_num):
                if fnum < 21:
                    print("low")
                elif fnum > 38:
                    print("high")
                elif fnum == 38:
                    print(fnum)
                    ax1.axvline(vol, ymin=0, ymax=0.06, color=fcolor, lw=1.4, alpha=0.7, zorder=5)

                elif in_range(vol, elution):
                    ax1.axvline(vol, ymin=0, ymax=0.06, color=fcolor, lw=1.4, alpha=0.7, zorder=5)
                    ax1.text(
                        vol+0.2, 0.015, str(int(round(fnum))),
                        transform=ax1.get_xaxis_transform(),
                        fontsize=FRACTION_FONT_SIZE,
                        color=fcolor,
                        ha='center', va='bottom',
                        bbox=dict(facecolor='white', alpha=0.75, edgecolor='none', pad=1.0)
                    )
                else:
                    pass

    # ── Legend ───────────────────────────────────────────────────
    ax1.legend(handles=handles, loc='upper right', fontsize=LEGEND_FONT_SIZE, frameon=False)


    plt.tight_layout()

    if save:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"Figure saved: {os.path.abspath(save_path)}")
        plt.close(fig)
    else:
        plt.show()
        plt.close(fig)


# ────────────────────────────────────────────────
#  Main execution
# ────────────────────────────────────────────────

if __name__ == "__main__":
    # Note: Cond & Fracx is manually turned off
    files = [
        '20231228_CS-AC+R2-SEC_Superdex200increase_20mMH-K-7.7_200mMKCl_R2ctrl.csv',
        '20231228_CS-AC+R2-SEC_Superdex200increase_20mMH-K-7.7_200mMKCl_CSctrl.csv',
        '20231228_CS-AC+R2-SEC_Superdex200increase_20mMH-K-7.7_200mMKCl_CS+R2.csv'
    ]
    elution = (-20, 370)    # Default for SEC (-20, 370)
    elution = (0.1,25)  # Manually set elution volume
    UV=(-5, 75)
    Cond = (1, 50)
    save_figure = True
    save_name = "20231228_CSR2_SEC_withFracx.svg"



    curves_list = []
    labels = ["Rvb1/2", "Cbf5-His/Shq1", "Rvb1/2 + Cbf5-His/Shq1"]

    for filename in files:
        input_path = os.path.join('AKTA', filename)
        output_path = os.path.join('AKTA', 'utf-8_format', filename)

        if not os.path.exists(input_path):
            print(f"File not found: {input_path}")
            continue

        convert_to_utf8(input_path, output_path)
        curves = read_unicorn_curves(output_path, baseline_start_ml=1.8, baseline_end_ml=3.2, do_baseline_correction=True)
        curves_list.append(curves)

    if not curves_list:
        print("No data loaded.")
    else:
        draw_multiple_UV_Cond_Fracx(
            curves_list=curves_list,
            labels=labels,
            title="",
            elution=elution,
            UV=UV,
            Cond=Cond,
            save=save_figure,
            save_path=save_name,
            show_cond = False,
            fraction_colors='#000000'
        )