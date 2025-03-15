import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

class ToVectorEmbedding(nn.Module):
    def __init__(self, in_dim=3, out_dim=64, layernum=1):
        super().__init__()
        self.norm = nn.GroupNorm(1, in_dim)  # 组归一化
        self.convs = nn.ModuleList([])  # layernum层卷积
        dim2 = in_dim
        for i in range(layernum):
            dim1 = dim2
            dim2 = dim1 * 2
            self.convs.append(
                nn.Conv2d(dim1, dim2, kernel_size=3, stride=2, padding=1)
            )
        self.flatten = nn.Flatten()
        self.out_dim = out_dim
        self.weight = None  # 动态创建的权重
        self.bias = None  # 动态创建的偏置

    def forward(self, x):
        x = self.norm(x)  # 组归一化
        for conv in self.convs:
            x = conv(x)  # 卷积操作
            x = F.relu(x)  # ReLU激活函数
        x = self.flatten(x)  # 展平操作

        input_dim = x.size(1)

        # 只在第一次前向传播时创建 weight 和 bias
        if self.weight is None:
            # 使用 Kaiming 初始化权重
            self.weight = nn.Parameter(torch.empty(self.out_dim, input_dim).to(x.device))
            init.kaiming_normal_(self.weight, mode='fan_in', nonlinearity='relu')

            # 使用均匀分布初始化偏置
            self.bias = nn.Parameter(torch.empty(self.out_dim).to(x.device))
            init.uniform_(self.bias, -1 / input_dim ** 0.5, 1 / input_dim ** 0.5)

        x = F.linear(x, self.weight, self.bias)

        x = torch.mean(x, dim=0)  # 对 batch 维度求均值
        return x