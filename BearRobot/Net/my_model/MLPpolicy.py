import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from BearRobot.Net.basic_net.mlp import MLP


class MLPPi(nn.Module):
       """
       Basic MLP policy implementation pi(s)->a
       """
       def __init__(
              self, 
              input_dim: int, 
              output_dim: int,
              hidden_size: list=[256, 256],
              ac_fn: str='relu',
              layer_norm: bool=False,
              dropout_rate: float=0.,
              device='cpu',
       ):
              super().__init__()
              
              self.model = MLP(input_dim, hidden_size, output_dim, ac_fn, layer_norm, dropout_rate)
              self.device = device
              
       def forward(self, s):
              a = self.model(s)
              a = torch.tanh(a)
              return a
              