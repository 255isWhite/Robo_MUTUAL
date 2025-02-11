import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from BearRobot.Net.basic_net.mlp import MLP


class MLPV(nn.Module):
       """
       Basic MLP state value function implementation V(s)
       """
       def __init__(
              self, 
              input_dim: int, 
              output_dim: int,
              hidden_size: list=[256, 256],
              ac_fn: str='relu',
              layer_norm: bool=False,
              dropout_rate: float=0.,
       ):
              super().__init__()
              
              self.model = MLP(input_dim, hidden_size, output_dim, ac_fn, layer_norm, dropout_rate)
              
       def forward(self, s):
              v = self.model(s)
              return v
              

class MLPQ(nn.Module):
       """
       Basic MLP action value function implementation Q(s, a)
       """
       def __init__(
              self, 
              input_dim: int, 
              output_dim: int,
              hidden_size: list=[256, 256],
              ac_fn: str='relu',
              layer_norm: bool=False,
              dropout_rate: float=0.,
       ):
              super().__init__()
              
              self.model = MLP(input_dim, hidden_size, output_dim, ac_fn, layer_norm, dropout_rate)
       
       def forward(self, s, a):
              sa = torch.cat([s, a], dim=-1)
              q = self.model(sa)
              return q
       
       
class MLPQs(nn.Module):
       """
       Ensemble MLP Q networks [Q1(s,a), Q2(s,a), ..., Qn(s,a)]
       """
       def __init__(
              self, 
              input_dim: int, 
              output_dim: int,
              hidden_size: list=[256, 256],
              ensemble_num: int=2,
              ac_fn: str='relu',
              layer_norm: bool=False,
              dropout_rate: float=0.,
       ):
              super().__init__()
              
              self.models = nn.ModuleList()
              for _ in range(ensemble_num):
                     model = MLPQ(input_dim, output_dim, hidden_size, ac_fn, layer_norm, dropout_rate)
                     self.models.append(model)
                     
       def forward(self, s, a):
              qs = [model(s, a) for model in self.models]
              return qs
              
              
              
              