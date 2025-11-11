import os
import math
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

plt.rcParams['font.family'] = 'Arial'  # 使用 SimHei 字体（支持中文）
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示为方块的问题
plt.rc('xtick', labelsize=20)  # 全局 x 轴刻度大小
plt.rc('ytick', labelsize=20) # 全局 y 轴刻度大小
line_thickness = 5
label_font_size = 20
color_set = ['green','red','orange']

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


def read_miniDAWN_curves(miniDAWN_data):

    data = pd.read_csv(miniDAWN_data)
    curve_names = data.columns
    samples = []
    sample_index = int(len(data.columns)/4)*4 # Each sample has four curves
    for i in range(0,sample_index,4):
        new_sample = dict()
        print(f"Sample {int(i/4)} -> {curve_names[i+1]}")
        new_sample['Name'] = curve_names[i+1][:-6]
        new_sample['dRI'] = {'mL': data[curve_names[i]],    # see original data sequence
                             'dRI':data[curve_names[i+1]]}
        new_sample['Molar mass'] = {'mL': data[curve_names[i+2]],  # see original data sequence
                                    'mm': data[curve_names[i+3]]}
        samples.append(new_sample)

    print(f"{len(samples)} samples found !")
    return samples

def draw_dRI_Molar_mass(curves_from_read_miniDAWN_curves, x_range=(12,17), save=False, output=''):



    # Draw
    # Create a figure and axis
    fig, ax1 = plt.subplots(figsize=(8, 6)) # Molar mass
    ax2 = ax1.twinx() # dRI

    sample_index = len(curves_from_read_miniDAWN_curves)
    max_molar_mass = 0 # for ylim
    max_dRI = 0 # for ylim
    for i in range(sample_index):
        current_sample = curves_from_read_miniDAWN_curves[i]
        sample_name = current_sample['Name'].split('_')[1]
        color = color_set[i]
        ax1.plot(current_sample['Molar mass']['mL'],current_sample['Molar mass']['mm']/1000,label=sample_name,linewidth=line_thickness,color=color) # Change Da into kDa
        ax2.plot(current_sample['dRI']['mL'], current_sample['dRI']['dRI'],linewidth=line_thickness,color=color)

        molar_mass = max(current_sample['Molar mass']['mm']/1000)
        dRI = max(current_sample['dRI']['dRI'])
        if molar_mass > max_molar_mass:
            max_molar_mass = molar_mass
        if dRI > max_dRI:
            max_dRI = dRI

    # x-axis
    ax1.set_xlabel('Elution volume (mL)', fontsize=label_font_size)
    ax1.set_xlim(x_range[0] - 0.2, x_range[1] + 0.2)
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))

    # Molar mass y-axis
    ax1.set_ylabel('Molar Mass (kDa)',fontsize=label_font_size, labelpad=12)
    ax1.set_ylim(0,max_molar_mass*2)
    ax1.set_yticks([0,100, 200, 300])
    ax1.set_yticklabels([0,100, 200, 300])
    print(max_molar_mass)

    # dRI y-axis
    ax2.set_ylabel('Differential refractive index',fontsize=label_font_size,rotation=270, labelpad=24)
    ax2.set_ylim(0,max_dRI*1.2)
    ax2.set_yticklabels([])
    ax2.tick_params(axis='y', length=0)

    # figure settings
    for axis in ['top', 'bottom', 'left', 'right']:
        ax1.spines[axis].set_linewidth(3)  # 3个点的线宽
    plt.tight_layout()
    ax1.legend()

    if save:
        print("File saved!")
        plt.savefig(f'{output[:-4]}.png', dpi=300)
    else:
        plt.show()


if __name__ == '__main__':
    filename = '251110_b-30aa.csv'
    save_figure = False

    # Convert file into utf-8 encoding
    input_file = 'miniDAWN/' + filename
    output_file = 'miniDAWN/utf-8_format/' + filename
    convert_to_utf8(input_file, output_file)

    curves = read_miniDAWN_curves(output_file)
    draw_dRI_Molar_mass(curves, x_range=(13,16),save=save_figure, output=filename)