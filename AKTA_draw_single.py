# ============================================================
# AKTA_draw_v4.1
# Refactored from v3.2 + batch-by-date feature
#
# Changes vs v3.2:
#   - Fixed bug: draw_UV_Cond_Fracx() no longer reads global `filename`
#   - Removed dead code: round_up_to_first_sig_digit(), get_csv()
#   - All bare `except:` replaced with typed exceptions
#   - convert_to_utf8() now raises on failure instead of silently returning
#   - Output directory is auto-created if missing
#   - uv_260_curve scoping issue resolved via dict.get()
#   - in_range() simplified to a one-liner
#   - rcParams consolidated into configure_plot_style()
#   - Main block refactored into run() for testability
#
# v4.1 additions:
#   - parse_date()         : normalise user date input (YYYYMMDD / YYYY-MM-DD / YYYY/MM/DD)
#   - find_files_by_date() : scan a directory for CSV/ASC files matching a date prefix
#   - run_batch_by_date()  : encode → read → plot every matched file; per-file error
#                            isolation + summary report at the end
# ============================================================

import re
import math
import logging
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from dataclasses import dataclass, field
from pathlib import Path
from charset_normalizer import detect

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
log = logging.getLogger(__name__)


# ── Plot style ────────────────────────────────────────────────────────────────

LINE_THICKNESS  = 2
LABEL_FONT_SIZE = 32
TICK_FONT_SIZE  = 24

def configure_plot_style() -> None:
    """Apply global matplotlib style settings."""
    plt.rcParams.update({
        'font.family':         'Arial',
        'axes.unicode_minus':  False,
        'xtick.labelsize':     TICK_FONT_SIZE,
        'ytick.labelsize':     TICK_FONT_SIZE,
    })


# ── Encoding helpers ──────────────────────────────────────────────────────────

def detect_encoding(file_path: str | Path) -> str:
    """Return the detected encoding of a file (falls back to utf-8)."""
    with open(file_path, 'rb') as fh:
        result = detect(fh.read())
    encoding   = result['encoding'] or 'utf-8'
    confidence = result['confidence'] or 0.0
    log.info("Detected encoding: %s (confidence %.0f%%)", encoding, confidence * 100)
    return encoding


def convert_to_utf8(input_file: str | Path, output_file: str | Path) -> Path:
    """
    Re-encode *input_file* as UTF-8 and write to *output_file*.

    The output directory is created automatically if it does not exist.
    Raises RuntimeError on failure so callers are never silently broken.
    """
    input_file  = Path(input_file)
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    encoding = detect_encoding(input_file)
    try:
        content = input_file.read_text(encoding=encoding)
    except UnicodeDecodeError as exc:
        raise RuntimeError(
            f"Could not decode '{input_file}' with encoding '{encoding}'. "
            "Try specifying the encoding manually."
        ) from exc

    output_file.write_text(content, encoding='utf-8')
    log.info("Saved UTF-8 file → %s", output_file)
    return output_file


# ── Data helpers ──────────────────────────────────────────────────────────────

def _curve_find(name: str, header_row: pd.DataFrame) -> tuple[int, int]:
    """
    Return the (row, col) index of *name* inside *header_row*.
    Raises ValueError when the name is absent or duplicated.
    """
    mask = (header_row == name)
    count = mask.sum().sum()
    if count == 1:
        return tuple(mask.stack().idxmax())
    raise ValueError(f"Expected exactly 1 match for '{name}', found {count}.")


def _tick_step(axis_max: float) -> float:
    """
    Compute a readable major-tick interval ≈ axis_max / 5,
    rounded up to two significant digits.
    """
    x = axis_max / 5
    if x == 0:
        return 1
    magnitude    = math.floor(math.log10(abs(x)))
    scaled       = x / 10 ** magnitude
    rounded      = math.ceil(scaled * 10) / 10
    return math.copysign(rounded * 10 ** magnitude, x)


def in_range(value: float, bounds: tuple[float, float]) -> bool:
    return bounds[0] <= value <= bounds[1]


# ── Core I/O ──────────────────────────────────────────────────────────────────

def read_unicorn_curves(unicorn_data: str | Path) -> dict:
    """
    Parse a UNICORN-exported CSV/ASC file.

    Returns a dict with keys 'UV', 'Cond', 'Fraction', and optionally
    'UV 2_260', each containing a sub-dict of labelled pd.Series.
    """
    log.info("Reading UNICORN file: %s", unicorn_data)

    # Row 0 → channel names; rows 2+ → numeric data
    header = pd.read_csv(unicorn_data, sep='\t', nrows=1)
    data   = pd.read_csv(unicorn_data, sep='\t', skiprows=2)
    # Integer column indices are easier to offset
    header.columns = range(len(header.columns))
    data.columns   = range(len(data.columns))

    # ── Locate columns ───────────────────────────────────────────────────────
    # UV 280 nm  (UNICORN ≥ 7 uses "UV 1_280", older uses "UV")
    for uv_name in ('UV 1_280', 'UV'):
        try:
            uv_col = _curve_find(uv_name, header)
            break
        except ValueError:
            pass
    else:
        raise ValueError("No UV column found in file.")

    # UV 260 nm (optional)
    try:
        uv260_col: tuple | None = _curve_find('UV 2_260', header)
    except ValueError:
        uv260_col = None

    cond_col = _curve_find('Cond',     header)
    frac_col = _curve_find('Fraction', header)

    log.info("UV→%s  Cond→%s  Fraction→%s  UV260→%s",
             uv_col, cond_col, frac_col, uv260_col)

    # ── Fraction numbers ─────────────────────────────────────────────────────
    frac_labels = (
        data[frac_col[1] + 1]
        .dropna()
        .pipe(lambda s: s[s != 'Waste'])          # drop waste marker
        .replace('Frac', 2)                        # manual-fraction edge case
        .reset_index(drop=True)
        .astype(int)
    )

    # ── Assemble output ──────────────────────────────────────────────────────
    def col_float(c: int) -> pd.Series:
        return data[c].astype(float)

    result = {
        'UV':       {'mL': col_float(uv_col[1]),     'mAU':   col_float(uv_col[1] + 1)},
        'Cond':     {'mL': col_float(cond_col[1]),   'mS/cm': col_float(cond_col[1] + 1)},
        'Fraction': {'mL': col_float(frac_col[1]),   'Fraction': frac_labels},
    }
    if uv260_col is not None:
        result['UV 2_260'] = {
            'mL':  col_float(uv260_col[1]),
            'mAU': col_float(uv260_col[1] + 1),
        }
    return result


# ── Plotting ──────────────────────────────────────────────────────────────────

def draw_UV_Cond_Fracx(
    curves: dict,
    title:   str             = 'default',
    UV:      tuple[float, float] = (-20, 3000),
    Cond:    tuple[float, float] = (0,   100),
    elution: tuple[float, float] = (-20, 370),
    save:    bool            = False,
) -> None:
    """
    Plot UV absorbance, conductivity, and fraction markers.

    Parameters
    ----------
    curves  : dict returned by read_unicorn_curves()
    title   : figure title (also used as the output filename stem)
    UV      : (min, max) for the left y-axis  [mAU]
    Cond    : (min, max) for the right y-axis [mS/cm]
    elution : (min, max) x-axis window         [mL]
    save    : write PNG to disk instead of calling plt.show()
    """
    uv_curve       = curves['UV']
    cond_curve     = curves['Cond']
    fraction_curve = curves['Fraction']
    uv260_curve    = curves.get('UV 2_260')   # None when absent – no scoping issue

    # ── Canvas ───────────────────────────────────────────────────────────────
    fig, ax1 = plt.subplots(figsize=(16, 8))
    ax2      = ax1.twinx()

    ax1.set_xlim(elution)
    ax1.set_ylim(UV)
    ax2.set_ylim(Cond)
    ax1.set_ylabel('')
    ax2.set_ylabel('')

    for ax in (ax1, ax2):
        ax.spines['top'].set_visible(False)

    # ── Tick spacing ─────────────────────────────────────────────────────────
    ax1.yaxis.set_major_locator(ticker.MultipleLocator(_tick_step(UV[1])))
    ax1.yaxis.set_minor_locator(ticker.AutoMinorLocator(4))
    ax2.yaxis.set_major_locator(ticker.MultipleLocator(_tick_step(Cond[1])))
    ax2.yaxis.set_minor_locator(ticker.AutoMinorLocator(4))

    # ── Labels ───────────────────────────────────────────────────────────────
    ax1.set_title(title, y=1.02, fontsize=LABEL_FONT_SIZE - 12)
    ax1.set_xlabel('')

    ax1.text(0.5, -0.12, 'Elution / mL',
             transform=ax1.transAxes, fontsize=LABEL_FONT_SIZE,
             ha='center', va='top')
    ax1.text(0, 1.04, 'mAU',
             transform=ax1.transAxes, fontsize=LABEL_FONT_SIZE,
             ha='center', va='bottom')
    ax2.text(1, 1.04, 'mS/cm',
             transform=ax2.transAxes, fontsize=LABEL_FONT_SIZE,
             ha='center', va='bottom')

    # ── Curves ───────────────────────────────────────────────────────────────
    ax1.plot(uv_curve['mL'],   uv_curve['mAU'],   label='UV 280',
             color='black', linewidth=LINE_THICKNESS)
    ax2.plot(cond_curve['mL'], cond_curve['mS/cm'], label='Cond',
             color='brown',  linewidth=LINE_THICKNESS * 0.8)

    if uv260_curve is not None:
        ax1.plot(uv260_curve['mL'], uv260_curve['mAU'], label='UV 260',
                 color='gray', linewidth=LINE_THICKNESS)

    # ── Fraction markers ─────────────────────────────────────────────────────
    xform = ax1.get_xaxis_transform()
    for frac_num, frac_ml in zip(fraction_curve['Fraction'], fraction_curve['mL']):
        if not in_range(frac_ml, elution):
            continue
        ax1.axvline(x=frac_ml, ymin=0, ymax=0.05,
                    color='orange', linewidth=LINE_THICKNESS,
                    linestyle='-', alpha=0.6)
        ax1.text(frac_ml, 0.01, str(frac_num),
                 transform=xform, fontsize=8,
                 color='orange', ha='left', va='bottom')

    # ── Legend ───────────────────────────────────────────────────────────────
    ax1.add_artist(ax1.legend(loc='upper left',  fontsize=16))
    ax2.add_artist(ax2.legend(loc='upper right', fontsize=16))

    plt.tight_layout()

    if save:
        out_path = f'{title}.png'
        plt.savefig(out_path, dpi=300)
        log.info("Figure saved → %s", out_path)
    else:
        plt.show()

    plt.close(fig)


# ── Entry point (single file) ─────────────────────────────────────────────────

def run(
    filename:    str,
    input_dir:   str                 = '0_AKTA',
    output_dir:  str                 = '0_AKTA/utf-8_format',
    elution:     tuple[float, float] = (-20, 370),
    UV:          tuple[float, float] = (-50, 300),
    Cond:        tuple[float, float] = (0,   50),
    save_figure: bool                = True,
) -> None:
    """End-to-end pipeline for a single file: encode → read → plot."""
    configure_plot_style()

    input_file  = Path(input_dir)  / filename
    output_file = Path(output_dir) / filename

    convert_to_utf8(input_file, output_file)
    curves = read_unicorn_curves(output_file)

    title = filename.split()[0]   # part before first space
    draw_UV_Cond_Fracx(curves, title=title, UV=UV, Cond=Cond,
                       elution=elution, save=save_figure)


# ── Batch-by-date ─────────────────────────────────────────────────────────────

# Expected filename convention: YYYYMMDD_<anything>.csv  (or .asc)
_DATE_IN_FILENAME = re.compile(r'^\d{8}')


def parse_date(date_input: str) -> str:
    """
    Normalise a user-supplied date string to compact YYYYMMDD format.

    Accepted formats
    ----------------
    ``20260520``, ``2026-05-20``, ``2026/05/20``

    Returns
    -------
    str
        Eight-digit date string, e.g. ``'20260520'``.

    Raises
    ------
    ValueError
        If the input cannot be parsed or represents an impossible date.
    """
    # Strip separators, then validate
    compact = re.sub(r'[-/]', '', date_input.strip())
    if not re.fullmatch(r'\d{8}', compact):
        raise ValueError(
            f"Unrecognised date format: '{date_input}'. "
            "Use YYYYMMDD, YYYY-MM-DD, or YYYY/MM/DD."
        )
    # Basic calendar sanity check via pandas (raises on bad month/day)
    try:
        pd.Timestamp(year=int(compact[:4]),
                     month=int(compact[4:6]),
                     day=int(compact[6:]))
    except ValueError as exc:
        raise ValueError(f"Invalid calendar date '{date_input}': {exc}") from exc

    return compact


def find_files_by_date(
    date:      str,
    directory: str | Path = '0_AKTA',
) -> list[Path]:
    """
    Return all CSV/ASC files in *directory* whose name starts with *date*
    (an 8-digit YYYYMMDD string produced by :func:`parse_date`).

    The search is case-insensitive and limited to the top level of the
    directory (subdirectories are not traversed).
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"Input directory not found: '{directory}'")

    matched = sorted(
        p for p in directory.iterdir()
        if p.suffix.lower() in {'.csv', '.asc'}
        and p.name.startswith(date)
        and _DATE_IN_FILENAME.match(p.name)
    )
    log.info("Found %d file(s) for date %s in '%s'.", len(matched), date, directory)
    return matched


@dataclass
class BatchResult:
    """Summary returned by :func:`run_batch_by_date`."""
    date:      str
    succeeded: list[str] = field(default_factory=list)
    failed:    list[tuple[str, str]] = field(default_factory=list)   # (name, reason)
    skipped:   list[str] = field(default_factory=list)

    # ── convenience ──────────────────────────────────────────────────────────
    @property
    def total(self) -> int:
        return len(self.succeeded) + len(self.failed) + len(self.skipped)

    def print_summary(self) -> None:
        sep = '─' * 60
        print(f"\n{sep}")
        print(f"  Batch summary  │  date: {self.date}  │  total: {self.total}")
        print(sep)
        print(f"  ✓ succeeded : {len(self.succeeded)}")
        print(f"  ✗ failed    : {len(self.failed)}")
        print(f"  ⊘ skipped   : {len(self.skipped)}")
        if self.failed:
            print(f"\n  Failed files:")
            for name, reason in self.failed:
                print(f"    • {name}")
                print(f"      {reason}")
        print(sep + '\n')


def run_batch_by_date(
    date_input:  str,
    input_dir:   str                 = '0_AKTA',
    output_dir:  str                 = '0_AKTA/utf-8_format',
    elution:     tuple[float, float] = (-20, 370),
    UV:          tuple[float, float] = (-50, 300),
    Cond:        tuple[float, float] = (0,   50),
    save_figure: bool                = True,
) -> BatchResult:
    """
    Find every CSV/ASC file in *input_dir* that matches *date_input* and
    process each one through the full encode → read → plot pipeline.

    Each file is handled independently: an error in one file is logged and
    recorded in the returned :class:`BatchResult`, but processing continues
    for the remaining files.

    Parameters
    ----------
    date_input  : date string in any supported format (YYYYMMDD / YYYY-MM-DD / YYYY/MM/DD)
    input_dir   : directory that contains the raw UNICORN exports
    output_dir  : directory for UTF-8 re-encoded copies (created if absent)
    elution     : (min, max) x-axis window [mL]
    UV          : (min, max) left y-axis   [mAU]
    Cond        : (min, max) right y-axis  [mS/cm]
    save_figure : write PNG files to disk when True, display interactively when False

    Returns
    -------
    BatchResult
        Structured summary of which files succeeded, failed, or were skipped.
    """
    configure_plot_style()

    date   = parse_date(date_input)
    result = BatchResult(date=date)

    files = find_files_by_date(date, input_dir)

    if not files:
        log.warning("No files found for date '%s' in '%s'. Nothing to do.", date, input_dir)
        return result

    log.info("Starting batch: %d file(s) to process.", len(files))

    for i, input_path in enumerate(files, start=1):
        filename = input_path.name
        log.info("[%d/%d] Processing: %s", i, len(files), filename)

        try:
            output_path = Path(output_dir) / filename
            convert_to_utf8(input_path, output_path)
            curves = read_unicorn_curves(output_path)
            title  = filename.split()[0]
            draw_UV_Cond_Fracx(
                curves, title=title,
                UV=UV, Cond=Cond, elution=elution,
                save=save_figure,
            )
            result.succeeded.append(filename)
            log.info("[%d/%d] ✓ Done: %s", i, len(files), filename)

        except Exception as exc:        # isolate per-file failures
            reason = f"{type(exc).__name__}: {exc}"
            result.failed.append((filename, reason))
            log.error("[%d/%d] ✗ Failed: %s\n  %s", i, len(files), filename, reason)

    result.print_summary()
    return result


# ── __main__ ─────────────────────────────────────────────────────────────────

if __name__ == '__main__':

    # ── Option A: process a single known file ─────────────────────────────────
    # run(
    #     filename    = '20260520_SUMO-mRE-IDR[S15D][S37D][T39D]-linkerB_GBwith5mMAMP_S200F 001.csv',
    #     elution     = (-20, 370),
    #     UV          = (-50, 300),
    #     Cond        = (0,   50),
    #     save_figure = True,
    # )

    # ── Option B: process ALL files from a given date ─────────────────────────
    run_batch_by_date(
        date_input  = '20260520',        # also accepts '2026-05-20' or '2026/05/20'
        input_dir   = '0_AKTA',
        output_dir  = '0_AKTA/utf-8_format',
        elution     = (-20, 370),
        UV          = (-50, 300),
        Cond        = (0,   50),
        save_figure = True,
    )