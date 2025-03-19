import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

class ToVectorEmbedding(nn.Module):
    def __init__(self, in_dim=3, out_dim=64, layernum=1):
        super().__init__()
        self.norm = nn.GroupNorm(1, in_dim)
        self.convs = nn.ModuleList([])
        dim2 = in_dim
        for i in range(layernum):
            dim1 = dim2
            dim2 = dim1 * 2
            self.convs.append(
                nn.Conv2d(dim1, dim2, kernel_size=3, stride=2, padding=1)
            )
        self.flatten = nn.Flatten()
        self.out_dim = out_dim
        self.weight = None
        self.bias = None

    def forward(self, x):
        x = self.norm(x)
        for conv in self.convs:
            x = conv(x)
            x = F.relu(x)
        x = self.flatten(x)

        input_dim = x.size(1)

        if self.weight is None:
            self.weight = nn.Parameter(torch.empty(self.out_dim, input_dim).to(x.device))
            init.kaiming_normal_(self.weight, mode='fan_in', nonlinearity='relu')

            self.bias = nn.Parameter(torch.empty(self.out_dim).to(x.device))
            init.uniform_(self.bias, -1 / input_dim ** 0.5, 1 / input_dim ** 0.5)

        x = F.linear(x, self.weight, self.bias)

        x = torch.mean(x, dim=0)
        return x