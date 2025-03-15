import torch
from torchvision import transforms
from ImagePathDataset import *
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

def calculate_activation_statistics(files, model, batch_size=50, dims=2048,
                                    device='cpu', num_workers=1):
    act = get_activations(files, model, batch_size, dims, device, num_workers)
    act = torch.from_numpy(act).to(device)
    mu = torch.mean(act, dim=0).detach().cpu().numpy()
    sigma = torch.cov(act.T, correction=0).detach().cpu().numpy()
    return mu, sigma

def get_activations(files, model, batch_size=50, dims=2048, device='cpu',
                    num_workers=1):
    model.eval()

    if batch_size > len(files):
        print(('Warning: batch size is bigger than the data size. '
               'Setting batch size to data size'))
        batch_size = len(files)
    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor()
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet 数据集的均值和标准差
    ])
    dataset = ImagePathDataset(files, transforms=transform)
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            drop_last=False,
                            num_workers=num_workers)

    pred_arr = np.empty((len(files), dims))

    start_idx = 0

    for batch in tqdm(dataloader):
        batch = batch.to(device)

        with torch.no_grad():
            pred = model(batch)[0]
        pred = torch.flatten(pred, start_dim=1)
        pred = pred.detach().cpu().numpy()

        pred_arr[start_idx:start_idx + pred.shape[0]] = pred
        start_idx = start_idx + pred.shape[0]

    return pred_arr