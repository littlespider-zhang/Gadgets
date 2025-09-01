import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import multiprocessing

from scipy.optimize import curve_fit
from tqdm import tqdm


def gaussian(x, amp, mean, sigma):
    return amp * np.exp(-((x - mean)/sigma) ** 2)
    
def double_gaussian(x, a1, m1, s1, a2, m2, s2):
    return gaussian(x, a1, m1, s1) + gaussian(x, a2, m2, s2)

def calculate_rmsd(P, Q):
    diff = P - Q
    rmsd = np.sqrt(np.mean(np.sum(diff ** 2, axis=0)))
    return rmsd

def generate_coordinates(x_range=(0,1),y_range=(0,1),z_range=(0,1), points=50):
    """
    给定范围和点密度，生成所有点的坐标
    """
    # Generation guess: generation
    x_series = np.linspace(x_range[0], x_range[1], points)
    y_series = np.linspace(y_range[0], y_range[1], points)
    z_series = np.linspace(z_range[0], z_range[1], points)
    

    
    xx, yy, zz = np.meshgrid(x_series, y_series, z_series, indexing='ij')
    
    coordinates = np.stack((xx, yy, zz), axis=-1)
    
    coordinates = coordinates.reshape(-1, 3)
    
    return coordinates



def auto_guess_gaussian(x, y, iteration=50, rmsd_limit=400, ouput_dir='E:/two-Gaussian/'):
    
    """
    给定数组，猜测可能的高斯函数
    """
    param_space = generate_coordinates(x_range=(1000,7000),
                                       y_range=(-0.02,0.01),
                                       z_range=(0,1),
                                       points=iteration)
    p = len(param_space)
    pp = p*p*p*p
    current = 0
    for i in  tqdm(range(p)):
        for j in range(p):
            # 计算高斯分布
            a1, m1, s1 = param_space[i]
            a2, m2, s2 = param_space[j]
            
            y1 = gaussian(x, a1, m1, s1)
            y2 = gaussian(x, a2, m2, s2)

            y_fit = y1 + y2
            rmsd = calculate_rmsd(y,y_fit)
            # print(f"{current}/{pp}")
            if rmsd < rmsd_limit:
                print(f"RMSD：{round(rmsd,2)}\n")
                # 绘制图形
                plt.figure(figsize=(8, 6))
                plt.scatter(x, y, label='Data', color='green')
                plt.plot(x, y_fit, label='Fit', color='yellow')
                plt.plot(x, y1, label=f'Gaussian 1\na1:{a1}\nm1:{m1}\ns1:{s1}', color='blue')
                plt.plot(x, y2, label=f'Gaussian 2\na2:{a2}\nm2:{m2}\ns1:{s2}', color='red')
                plt.title(f'Data vs Fit: RMSD {round(rmsd,2)}')
                plt.xlabel('scores')
                plt.ylabel('Probability Density')
                plt.legend()
                plt.grid(True)
                plt.savefig(ouput_dir + f'{i}_{j}.jpg')
                #plt.show()
            current += 1
    return None

if __name__ == "__main__":
    data = pd.read_csv('sorted_scores_count.txt', sep='\t')
    x = data['Score']
    y = data['Count']
    
    auto_guess_gaussian(x, y, iteration=16, rmsd_limit=10000, ouput_dir='E:/two-Gaussian/') 
    

    



