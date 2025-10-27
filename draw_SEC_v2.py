import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

plt.rcParams['font.family'] = 'Arial'  # 使用 SimHei 字体（支持中文）
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示为方块的问题

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
def zyy_find(x, df):
    position = [(idx,col)
                for idx in df.index
                for col in df.columns
                if df.loc[idx, col] == x]
    if len(position) == 1:
        return position[0]
    else:
        print("Warning from zyy_find!!!")
        return None


# Section One
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
    UV_col = zyy_find('UV', curve_names) # UV_col -> mL, UV_col -> mAU;
    Cond_col = zyy_find('Cond', curve_names)
    Fraction_col = zyy_find('Fraction', curve_names)
    print(f"UV -> {UV_col}\nCond -> {Cond_col}\nFraction -> {Fraction_col}\n")
    # print(data[UV_col[1]+1])

    # Processing Fraction data
    fracx_1 = data[Fraction_col[1] + 1].dropna()
    fracx_2 = fracx_1[:-1][:].astype(float) # If not, you will encounter 'Waste', which slows down drawing (drawing number is faster than drawing text) see line 117
    # print(fracx_2)

    # Extract data
    return {'UV':{'mL':data[UV_col[1]].astype(float), 'mAU':data[UV_col[1] + 1].astype(float)},
            'Cond':{'mL':data[Cond_col[1]].astype(float), 'mS/cm':data[Cond_col[1]+1].astype(float)},
            'Fraction':{'mL':data[Fraction_col[1]].astype(float), 'Fraction':fracx_2}}



def draw_UV_Cond_Fracx(curves_from_read_unicorn_curves, UV=(-20,3000), Cond=(0,100), save=False, output=''):
    # Get data from each curves
    uv_curve = curves_from_read_unicorn_curves['UV']
    cond_curve = curves_from_read_unicorn_curves['Cond']
    fraction_curve = curves_from_read_unicorn_curves['Fraction']

    # check UV upper limit
    max_UV = max(uv_curve['mAU'])
    if max_UV > UV[1]:
        UV = UV[0],max_UV

    # Draw
    # Create a figure and axis
    fig, ax1 = plt.subplots(figsize=(16, 8))

    # Plot UV  on the left y-axis
    ax1.plot(uv_curve['mL'], uv_curve['mAU'], label='', color='black', linewidth=0.6)

    ax1.set_xlabel('mL')
    ax1.set_ylabel('mAU', rotation=1)
    ax1.set_ylim(UV)
    ax1.yaxis.set_major_locator(ticker.MultipleLocator(round(UV[1]/20)))  # Major ticks every 0.5 units
    ax1.yaxis.set_minor_locator(ticker.AutoMinorLocator(4))
    ax1.yaxis.set_label_coords(0, 1.02)
    ax1.spines['top'].set_visible(False)
    ax1.grid(False)
    # ax1.legend(loc='upper left')

    # Add fraction numbers
    for f, b in zip(fraction_curve['Fraction'], fraction_curve['mL']):
        ax1.axvline(x=b, ymin=0, ymax=0.05, color='orange', linewidth=0.5, linestyle='-')
        ax1.text(b, UV[0] + 5, round(f), fontsize=8, color='orange')

    # Create a second y-axis for Conc B
    ax2 = ax1.twinx()
    ax2.plot(cond_curve['mL'], cond_curve['mS/cm'], label='', color='brown', linewidth=0.6)

    ax2.set_ylabel('mS/cm', rotation=1, loc='top')
    ax2.set_ylim(Cond)
    ax2.yaxis.set_major_locator(ticker.MultipleLocator(round(Cond[1]/10)))  # Major ticks every 0.5 units
    ax2.yaxis.set_minor_locator(ticker.AutoMinorLocator(4))
    ax2.yaxis.set_label_coords(1.02, 1.05)
    ax2.spines['top'].set_visible(False)
    # ax2.legend(loc='upper right')

    # Set the title
    ax1.set_title(filename.split()[0], y=1.02)

    # Adjust layout
    plt.tight_layout()

    if save:
        print("File saved!")
        plt.savefig(f'{output.split()[0]}.png', dpi=300)
    else:
        plt.show()

    return True



if __name__ == '__main__':
    filename = '20251024_MBP3C-W-30aa[WT]+H3C_S75A.csv'
    UV=(-20, 1000) # if surpass this figure, programme will automatically chose the upper bound
    Cond = (0, 70)
    save_figure = True

    # Convert file into utf-8 encoding
    input_file = 'SEC/' + filename
    output_file = 'SEC/utf-8_format/' + filename
    convert_to_utf8(input_file, output_file)

    # Read utf-8 encoding files
    curves = read_unicorn_curves(output_file)
    # print(curves['UV']['mL'])
    draw_UV_Cond_Fracx(curves, UV=UV, Cond=Cond, save=save_figure, output=filename)