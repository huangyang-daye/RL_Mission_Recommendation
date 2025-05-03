# -*- coding: utf-8 -*-

# @Author : 黄杨
# @Time   : 2025/5/4 1:15
# @description : 一个基本的MLP结构，按照具体要求继承调用
from ast import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

class MLP:
    def __init__(self, input_channels: int, outputs: List[Tuple[int, nn.Module]],
                 activation: Callable[[Tensor], Tensor] = F.relu,
                 hidden_dims: List[int] = None, device: torch.device = "cuda"):
        super().__init__()
        self.input_channels = input_channels
        self.device = device
        self.activation = activation
        self.outputs = []
        if hidden_dims is None:
            hidden_dims = [256,256]
        hiddens = []
        last_dim = input_channels
        for d in hidden_dims:
            hiddens.append(nn.Linear(last_dim, d, device=device))
            last_dim = d
        self.hiddens = nn.Sequential(*hiddens)
        for out_dim, m in outputs:
            if m is None:
                self.outputs.append(nn.Linear(last_dim, out_dim, device=device))
            else:
                self.outputs.append(nn.Sequential(
                    nn.Linear(last_dim, out_dim, device=device), m))
    
    def forward(self, x: Tensor) -> List[Tensor]:
        '''
        前向传播
        '''
        x = x.to(self.device)
        for hidden_dim in self.hiddens:
            x = self.activation(hidden_dim(x))
        return [out[x] for out in self.outputs]

