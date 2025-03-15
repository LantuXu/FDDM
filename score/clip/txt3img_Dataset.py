import torch
import os
from PIL import Image
from torch.utils.data import Dataset

class txt3img_Dataset(Dataset):
    def __init__(self, root_dir, dict_text, processor, transform=None):
        """
        初始化数据集。

        参数:
            image_dir (str): 图像文件夹的路径。
            label_file (str): 包含图像文件名和标签的CSV文件路径。
            transform (callable, optional): 可选的数据变换函数。
        """
        self.root_dir = root_dir
        self.image_name = [f for f in os.listdir(root_dir)]
        self.dict_text = dict_text
        self.transform = transform
        self.processor = processor

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
        base_name = image_name.split('.')[0].lstrip('0')
        text = self.dict_text[base_name]

        # 加载图像
        image = Image.open(img_path).convert('RGB')

        # 应用数据变换
        if self.transform:
            image = self.transform(image)

        # 使用CLIP官方预处理
        processed = self.processor(
            text=[text],
            images=image,
            return_tensors="pt",
            padding="max_length",
            max_length=77,
            do_rescale=False
        )
        # print((processed["input_ids"].shape))
        # print(processed["attention_mask"])
        return {
            "pixel_values": processed["pixel_values"].squeeze(),
            "input_ids": processed["input_ids"].squeeze(),
            "attention_mask": processed["attention_mask"].squeeze(),
            "image_name": image_name
        }
