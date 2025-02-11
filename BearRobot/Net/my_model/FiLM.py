import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from BearRobot.Net.basic_net.mlp import MLP
from BearRobot.utils.net.initialization import init_weight


class FiLM_layer(nn.Module):
    def __init__(
        self,
        condition_dim: int,
        dim: int,
    ):
        super().__init__()
        self.net = MLP(condition_dim, [dim * 4], dim * 2)
        self.apply(init_weight)
        nn.init.zeros_(self.net.layers[-1].weight)
        nn.init.zeros_(self.net.layers[-1].bias)

    def forward(self, conditions, hiddens):
        # conditions shape: B, C'
        # hidden shape: B, C, H, W or B, N, C
        scale, shift = self.net(conditions).chunk(2, dim = -1)
        assert hiddens.shape[0] == scale.shape[0]
        if len(hiddens.shape) == 4:
            assert scale.shape[-1] == hiddens.shape[1]
            scale = scale.unsqueeze(-1).unsqueeze(-1) # shape -> B, C, 1, 1
            shift = shift.unsqueeze(-1).unsqueeze(-1) # shape -> B, C, 1, 1
        elif len(hiddens.shape) == 3:
            assert scale.shape[-1] == hiddens.shape[-1]
            scale = scale.unsqueeze(1) # shape -> B, 1, C
            shift = shift.unsqueeze(1)
        return hiddens * (scale + 1) + shift