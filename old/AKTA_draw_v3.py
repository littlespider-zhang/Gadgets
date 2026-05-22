# V3.2 update
# optimized UV curve recognition


import os
import math
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

plt.rcParams['font.family'] = 'Arial'  # 使用 SimHei 字体（支持中文）
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示为方块的问题
plt.rc('xtick', labelsize=24)  # 全局 x 轴刻度大小
plt.rc('ytick', labelsize=24) # 全局 y 轴刻度大小
line_thickness = 2
label_font_size = 32

from charset_normalizer import detect

# Input formatting： converted files are stored in utf-8_format
def detect_encoding(file_path):
    """Detect the encoding of a file."""
    with open(file_path, 'rb') as file:
        result = detect(file.read())
        encoding = result['encoding']
        confidence = result['confidence']
        print(f"Detected encoding: {encoding} (Confidence: {confidence:.2%})")
        return encoding


def convert_to_utf8(input_file, output_file):
    """Detect and Convert a file to UTF-8 encoding."""
    try:
        # Detect the encoding of the input file
        encoding = detect_encoding(input_file)

        if encoding is None:
            print("Could not detect encoding. Trying to read as UTF-8 by default.")
            encoding = 'utf-8'

        # Read the file with the detected encoding
        with open(input_file, 'r', encoding=encoding) as file:
            content = file.read()

        # Write the content to a new file in UTF-8
        with open(output_file, 'w', encoding='utf-8') as file:
            file.write(content)

        print(f"File successfully converted to UTF-8 and saved as {output_file}")

    except UnicodeDecodeError:
        print(f"Error: Could not decode the file with {encoding}. Try specifying the correct encoding manually.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

# Gadgets
def curve_find(x, df):
    """From chatGPT"""
    where = (df == x)
    if where.sum().sum() == 1:
        return tuple(where.stack().idxmax())
    else:
        raise Exception(f"{where.sum().sum()} curves found for {x}!")


def get_csv():
    current_directory = os.getcwd()

    # 获取所有 .csv 文件的路径
    csv_files = [os.path.join(current_directory, f) for f in os.listdir(current_directory) if f.endswith('.csv') or f.endswith('.asc')]

    # 打印文件路径
    return csv_files


def round_up_to_second_sig_digit(x):
    if x == 0:
        return 0

    # 获取数量级（10的指数）
    magnitude = math.floor(math.log10(abs(x)))

    # 将数缩放到1~10之间
    scaled = x / (10 ** magnitude)

    # 保留两位有效数字并向上取整第二位
    # 例如 15.6 -> 第二位有效数字是 5
    # 我们取 ceil(scaled * 10) / 10 再乘回去
    rounded_scaled = math.ceil(scaled * 10) / 10

    # 乘回原来的数量级
    result = rounded_scaled * (10 ** magnitude)

    # 保持符号
    return math.copysign(result, x)


def round_up_to_first_sig_digit(x):
    # not used # same as second
    if x == 0:
        return 0
    print(x)
    # 获取数量级（10的指数）
    magnitude = math.floor(math.log10(abs(x)))
    print(magnitude)

    # 将数缩放到1~10之间
    scaled = x / (10 ** magnitude)
    print(scaled)

    # 保留两位有效数字并向上取整第二位
    # 例如 15.6 -> 第二位有效数字是 5
    # 我们取 ceil(scaled * 10) / 10 再乘回去
    rounded_scaled = math.ceil(scaled * 10) / 10
    print(rounded_scaled)

    # 乘回原来的数量级
    result = rounded_scaled * (10 ** magnitude)
    print(result)
    print('\n')

    # 保持符号
    return math.copysign(result, x)


# Main functions
def read_unicorn_curves(unicron_data):
    print("\nReading unicorn files...")

    # Read curve name and data seperately
    curve_names = pd.read_csv(unicron_data, sep='\t', nrows=1)
    data = pd.read_csv(unicron_data, sep='\t', skiprows=2) # do not use .astype(float) here. you will encounter 'Method Settings'
    # Change columns to numbers for easier positioning
    curve_names.columns = range(len(curve_names.columns))
    data.columns = range(len(data.columns))
    # print(curve_names)
    # print(data)

    # Find which col contains corresponding data
    try:
        UV_col = curve_find('UV 1_280', curve_names)  # UV_col -> mL, UV_col -> mAU;
    except:
        UV_col = curve_find('UV', curve_names)  # UV_col -> mL, UV_col -> mAU;
    try:
        UV_260_col = curve_find('UV 2_260', curve_names)  # UV_col -> mL, UV_col -> mAU;
    except:
        UV_260_col = None

    Cond_col = curve_find('Cond', curve_names)
    Fraction_col = curve_find('Fraction', curve_names)
    print(f"UV -> {UV_col}\nCond -> {Cond_col}\nFraction -> {Fraction_col}\n")
    # print(data[UV_col[1]+1])

    # Processing Fraction data
    fracx_1 = data[Fraction_col[1] + 1].dropna()
    fracx_1 = fracx_1[fracx_1 != "Waste"].reset_index(drop=True)    # remove Waste flag
    fracx_1 = fracx_1.replace("Frac", 2)    # If manually set Fracx, /Frac/ occurs in Fraction data; Seen in HiTrap Q HP manual fraction case
    fracx_2 = fracx_1.astype(int)
    # try:
    #     fracx_2 = fracx_1[0:-1][:].astype(int) # If not, you will encounter 'Waste', which slows down drawing (drawing number is faster than drawing text) see line 117
    # except:
    #     fracx_2 = fracx_1[0:-2][:].astype(int)
    # print(fracx_2)

    # Extract data
    try:
        return {'UV':{'mL':data[UV_col[1]].astype(float), 'mAU':data[UV_col[1] + 1].astype(float)},
                'Cond':{'mL':data[Cond_col[1]].astype(float), 'mS/cm':data[Cond_col[1]+1].astype(float)},
                'Fraction':{'mL':data[Fraction_col[1]].astype(float), 'Fraction':fracx_2},
                'UV 2_260':{'mL':data[UV_260_col[1]].astype(float), 'mAU':data[UV_260_col[1] + 1].astype(float)}}
    except:
        return {'UV': {'mL': data[UV_col[1]].astype(float), 'mAU': data[UV_col[1] + 1].astype(float)},
                'Cond': {'mL': data[Cond_col[1]].astype(float), 'mS/cm': data[Cond_col[1] + 1].astype(float)},
                'Fraction': {'mL': data[Fraction_col[1]].astype(float), 'Fraction': fracx_2}}




def draw_UV_Cond_Fracx(curves_from_read_unicorn_curves, title='default', UV=(-20,3000), Cond=(0,100), elution=(-20, 370),save=False):
    # Get data from each curves
    try:
        uv_curve = curves_from_read_unicorn_curves['UV']
    except:
        uv_curve = curves_from_read_unicorn_curves['UV 1_280']

    try:
        uv_260_curve = curves_from_read_unicorn_curves['UV 2_260']
    except:
        pass

    cond_curve = curves_from_read_unicorn_curves['Cond']
    fraction_curve = curves_from_read_unicorn_curves['Fraction']

    # Draw
    # Create a figure and axis
    fig, ax1 = plt.subplots(figsize=(16, 8))

    # Figure settings
    ## x axis
    ax2 = ax1.twinx()
    ax1.set_xlim(elution)


    ## left-axis
    ax1.set_ylabel('')
    ax1.set_ylim(UV)
    ax1.yaxis.set_major_locator(ticker.MultipleLocator(round_up_to_second_sig_digit(UV[1]/5)))  # Major ticks every 0.5 units
    ax1.yaxis.set_minor_locator(ticker.AutoMinorLocator(4))

    ## right-axis
    ax2.set_ylabel('')
    ax2.set_ylim(Cond)
    ax2.yaxis.set_major_locator(ticker.MultipleLocator(round_up_to_second_sig_digit(Cond[1]/5)))  # Major ticks every 0.5 units
    ax2.yaxis.set_minor_locator(ticker.AutoMinorLocator(4))

    # Label settings
    ax1.set_title(filename.split()[0], y=1.02) # name
    ## x label
    ax1.set_xlabel('')
    ax1.text(
        0.5, -0.12, 'Elution / mL',
        transform=ax1.transAxes,
        fontsize=label_font_size,
        ha='center',
        va='top'
    )
    ## left y label
    ax1.text(
        0, 1.04, 'mAU',
        transform=ax1.transAxes,
        fontsize=label_font_size,
        ha='center',
        va='bottom'
    )
    ## right y label
    ax2.text(
        1, 1.04, 'mS/cm',
        transform=ax2.transAxes,
        fontsize=label_font_size,
        ha='center',
        va='bottom'
    )
    ax1.spines['top'].set_visible(False)
    ax2.spines['top'].set_visible(False)

    # Plot UV/Cond on the left/right y-axis;
    ax1.plot(uv_curve['mL'], uv_curve['mAU'], label='UV280', color='black', linewidth=line_thickness)
    ax2.plot(cond_curve['mL'], cond_curve['mS/cm'], label='Cond', color='brown', linewidth=line_thickness*0.8)

    try:
        a = uv_260_curve
        ax1.plot(uv_curve['mL'], a['mAU'], label='UV260', color='gray', linewidth=line_thickness)
    except:
        pass

    # Add fraction numbers
    for f, b in zip(fraction_curve['Fraction'], fraction_curve['mL']):
        ax1.axvline(x=b, ymin=0, ymax=0.05, color='orange', linewidth=line_thickness, linestyle='-', alpha=0.6)
        # ax1.text(b, UV[0]+0.02, round(f), fontsize=8, color='orange')
        ax1.text(
            b, 0.01, str(round(f)),
            transform=ax1.get_xaxis_transform(),  # x 用数据坐标，y 用 axes 坐标
            fontsize=8,
            color='orange',
            ha='left',
            va='bottom'
        )

    # legend
    legend1 = ax1.legend(loc='upper left', fontsize=16)  # 左轴图例
    legend2 = ax2.legend(loc='upper right', fontsize=16)  # 右轴图例
    ax1.add_artist(legend1)  # 保证左轴图例不会被覆盖
    ax2.add_artist(legend2)  # 保证左轴图例不会被覆盖

    # Adjust layout
    plt.tight_layout()

    # Save figure
    if save:
        print("File saved!")
        plt.savefig(f'{title}.png', dpi=300)
        plt.close(fig)
    else:
        plt.show()
        plt.close(fig)

    return True



if __name__ == '__main__':
# Batch drawing

    filename = '20260227_32m3C-mRE-IDR[del81-92]_WT_T39A_S77A_GB_ResQ.csv'
    elution = (-20, 370)    # Default for SEC (-20, 370)
    elution = (250,350)  # Manually set elution volume
    UV=(-20, 500)
    Cond = (10, 200)
    save_figure = True



    # Convert file into utf-8 encoding
    input_file = 'AKTA/' + filename
    output_file = 'AKTA/utf-8_format/' + filename
    convert_to_utf8(input_file, output_file)

    # Read utf-8 encoding files
    curves = read_unicorn_curves(output_file)

    # Main function
    draw_UV_Cond_Fracx(curves, title=filename.split()[0], UV=UV, Cond=Cond, elution=elution, save=save_figure)
