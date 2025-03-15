import torchvision.models as models
from torchvision import transforms
from ImagePathDataset import *
from torch.utils.data import DataLoader

def get_model(model_path, device):
    model = models.inception_v3(weights=None, aux_logits=True, init_weights=False).to(device)
    checkpoint = torch.load(model_path, weights_only=True)
    model.load_state_dict(checkpoint)
    model.eval()
    # block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
    # model = InceptionV3([block_idx], localmodel_path=model_path).to(device)
    # # model = torch.nn.Sequential(*list(model.children())[:-1])  # 只取特征部分
    # model_list = list(model.children())
    # for l in model_list:
    #     print(l)
    return model

def get_dataloader(path, batch_size, num_workers=1):
    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet 数据集的均值和标准差
    ])
    dataset = ImagePathDataset(path, transform=transform)
    dataloader = DataLoader(dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             drop_last=False,
                                             num_workers=num_workers)
    return dataloader