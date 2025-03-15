import numpy as np
import torch

def gaussian_mask(img_size, sigma):
    # 生成坐标范围 [-size//2, size//2)
    l, r = -((img_size) // 2), img_size // 2
    ax = np.linspace(l, r, img_size)
    x, y = np.meshgrid(ax, ax)

    # 计算高斯权重
    mask = np.exp(-((x * 3 / r) ** 2 + (y * 3 / r) ** 2) / (2 * sigma ** 2))

    mask = torch.from_numpy(mask).float()

    return mask