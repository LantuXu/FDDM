import torch
import torch.nn as nn
from inspect import isfunction

class GroupNorm32(nn.GroupNorm):
    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)

def normalization(channels):

    return GroupNorm32(32, channels)

def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)

def default(val, d):
    if val is not None:
        return val
    return d() if isfunction(d) else d

def zero_module(module):

    for p in module.parameters():
        p.detach().zero_()
    return module