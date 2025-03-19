from torch.utils.data import Dataset
import os.path
from PIL import Image
import torch

class txt2img_Dataset(Dataset):
    def __init__(self, root_dir, dict_text, transform=None):
        self.root_dir = root_dir
        self.image_name = [f for f in os.listdir(root_dir)]
        self.dict_text = dict_text
        self.transform = transform

    def __len__(self):
        return len(self.image_name)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Get the image filename and label
        image_name = self.image_name[idx]
        img_path = os.path.join(self.root_dir, image_name)
        base_name = image_name.split('.')[0].lstrip('0')
        text = self.dict_text[base_name]

        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, text