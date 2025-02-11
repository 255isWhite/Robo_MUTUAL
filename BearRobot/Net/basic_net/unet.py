import torch
import torch.nn as nn
import torch.nn.functional as F

AC_FN ={'relu': F.relu}

class MLP(nn.Module):
       def __init__(self, input_size, hidden_sizes, output_size, ac_fn='relu', use_layernorm=False):
              super(MLP, self).__init__()
              self.use_layernorm = use_layernorm
              
              # initialize layers
              self.layers = nn.ModuleList()
              self.layernorms = nn.ModuleList() if use_layernorm else None
              self.ac_fn = AC_FN[ac_fn]
              
              self.layers.append(nn.Linear(input_size, hidden_sizes[0]))
              for i in range(1, len(hidden_sizes)):
                     self.layers.append(nn.Linear(hidden_sizes[i-1], hidden_sizes[i]))
                     
              if self.use_layernorm:
                     self.layernorms.append(nn.LayerNorm(hidden_sizes[i]))
                     
              self.layers.append(nn.Linear(hidden_sizes[-1], output_size))

              
       def forward(self, x):
              for layer in self.layers[:-1]:
                     x = self.ac_fn(layer(x))
              if self.use_layernorm:
                     x = self.layernorms[-1](x)
              x = self.layers[-1](x)
              return x
       

if __name__=='__main__':
       input_size = 100
       hidden_size = [256, 256]
       output_size = 2
       ac_fn = 'relu'
       
       # initialize the MLP
       mlp = MLP(input_size, hidden_size, output_size, ac_fn, use_layernorm=True).cuda()
        
       input_test = torch.randn(10, input_size).cuda()
       
       output_test = mlp(input_test)
       
       print("Output shape:", output_test.shape)
       print("Output:", output_test)