import torch
from tqdm import tqdm
from torch.nn.functional import softmax

@torch.no_grad()
def get_probabilities(model, dataloader, device):
    probs = []
    for step, batch_img in tqdm(enumerate(dataloader), desc='batch', unit='it',
                                             total=len(dataloader)):
        # print(batch_img.shape)
        batch_img = batch_img.to(device)
        outputs = model(batch_img)
        probs.append(softmax(outputs, dim=1))  # 计算softmax概率

    return torch.cat(probs, 0)
def inception_score(probs, num_splits=10):
    scores = []
    num_samples = probs.shape[0]
    for i in range(num_splits):
        subset = probs[i * (num_samples // num_splits): (i+1) * (num_samples // num_splits)]
        p_yx = subset
        p_y = p_yx.mean(0, keepdim=True)  # 计算边缘分布p(y)
        kl = p_yx * (torch.log(p_yx + 1e-16) - torch.log(p_y + 1e-16))  # 计算KL散度
        kl = kl.sum(1)  # 对类别维度求和
        kl_mean = kl.mean()  # 对样本求平均
        scores.append(kl_mean.exp().item())  # 取指数得到IS
    return sum(scores) / len(scores)  # 返回平均IS