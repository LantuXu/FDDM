import torch
from torch.utils.data import Dataset
import os
from PIL import Image

class ImagePathDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.image_name = [f for f in os.listdir(root_dir)]
        self.transform = transform

    def __len__(self):
        """返回数据集的大小。"""
        return len(self.image_name)

    def __getitem__(self, idx):
        """根据索引获取单个样本。"""
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # 获取图像文件名和标签
        image_name = self.image_name[idx]
        img_path = os.path.join(self.root_dir, image_name)
        base_name = image_name.split('.')[0]
        # 加载图像
        image = Image.open(img_path).convert('RGB')
        # 应用数据变换
        if self.transform:
            image = self.transform(image)
        # 返回图像和标签
        return image