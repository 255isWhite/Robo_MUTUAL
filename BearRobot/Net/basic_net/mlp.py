import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Callable, Optional, Union, Tuple, List


AC_FN ={'relu': F.relu,
        'mish': F.mish,
        'gelu': F.gelu}


class MLP(nn.Module):
       def __init__(
              self, 
              input_size:int, 
              hidden_sizes:list, 
              output_size:int, 
              ac_fn: str='relu', 
              use_layernorm: bool=False, 
              dropout_rate: float=0.
       ):
              super().__init__()             
              self.use_layernorm = use_layernorm
              self.dropout_rate = dropout_rate
              
              # initialize layers
              self.layers = nn.ModuleList()
              self.layernorms = nn.ModuleList() if use_layernorm else None
              self.ac_fn = AC_FN[ac_fn]
              if self.dropout_rate > 0:
                     self.dropout = nn.Dropout(self.dropout_rate)
              
              self.layers.append(nn.Linear(input_size, hidden_sizes[0]))
              for i in range(1, len(hidden_sizes)):
                     self.layers.append(nn.Linear(hidden_sizes[i-1], hidden_sizes[i]))
                     
              if self.use_layernorm:
                     self.layernorms.append(nn.LayerNorm(input_size))
                     
              self.layers.append(nn.Linear(hidden_sizes[-1], output_size))

              
       def forward(self, x):
              if self.use_layernorm:
                     x = self.layernorms[-1](x)
              
              for layer in self.layers[:-1]:
                     x = layer(x)
                     if self.dropout_rate > 0:
                            x = self.dropout(x)
                     x = self.ac_fn(x)

              x = self.layers[-1](x)
              return x
       

class MLPResNetBlock(nn.Module):
       """
       the MLPResnet Blocks used in IDQL: arXiv:2304.10573, Appendix G
       """
       def __init__(self, hidden_dim:int, ac_fn='relu', use_layernorm=False, dropout_rate=0.1):
              super(MLPResNetBlock, self).__init__()
              self.hidden_dim = hidden_dim
              self.use_layernorm = use_layernorm
              self.dropout = nn.Dropout(dropout_rate)
              self.norm1 = nn.LayerNorm(hidden_dim)
              self.dense1 = nn.Linear(hidden_dim, hidden_dim * 4)
              self.ac_fn = AC_FN[ac_fn]
              self.dense2 = nn.Linear(hidden_dim * 4, hidden_dim)
              
       def forward(self, x):
              identity = x
              
              out = self.dropout(x)
              out = self.norm1(out)
              out = self.dense1(out)
              out = self.ac_fn(out)
              out = self.dense2(out)
              out = identity + out
              
              return out


class MLPResNet(nn.Module):
       """
       the LN_Resnet used in IDQL: arXiv:2304.10573
       """
       def __init__(self, num_blocks:int, input_dim:int, hidden_dim:int, output_size:int, ac_fn='relu', use_layernorm=True, dropout_rate=0.1):
              super(MLPResNet, self).__init__()
              
              self.dense1 = nn.Linear(input_dim, hidden_dim)
              self.ac_fn = AC_FN[ac_fn]
              self.dense2 = nn.Linear(hidden_dim, output_size)
              self.mlp_res_blocks = nn.ModuleList()
              for _ in range(num_blocks):
                     self.mlp_res_blocks.append(MLPResNetBlock(hidden_dim, ac_fn, use_layernorm, dropout_rate))
              
       def forward(self, x):
              out = self.dense1(x)
              for mlp_res_block in self.mlp_res_blocks:
                     out = mlp_res_block(out)
              out = self.ac_fn(out)
              out = self.dense2(out)
              return out
              

if __name__=='__main__':
       input_size = 100
       hidden_size = [256, 256, 256]
       output_size = 2
       ac_fn = 'relu'
       
       # initialize the MLP
       mlp = MLP(input_size, hidden_size, output_size, ac_fn, use_layernorm=True).cuda()
        
       input_test = torch.randn(10, input_size).cuda()
       
       output_test = mlp(input_test)
       
       print("Output shape:", output_test.shape)
       print("Output:", output_test)
       
       
       # initialize the MLPResNet
       mlp_res = MLPResNet(3, 10, 256, 10, 'relu').cuda()
       input_test = torch.randn(10, 10).cuda()
       output_test = mlp_res(input_test)
       
       print("Output shape:", output_test.shape)
       print("Output:", output_test)
       