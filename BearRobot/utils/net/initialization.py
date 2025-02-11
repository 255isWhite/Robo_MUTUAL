import torch
import torch.nn as nn
import torch.nn.functional as F

import argparse

def boolean(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def init_weight(m):
       if isinstance(m, nn.Linear):
              nn.init.kaiming_normal_(m.weight)
              if m.bias is not None:
                     nn.init.constant_(m.bias, 0)
       elif isinstance(m, nn.LayerNorm):
              nn.init.constant_(m.bias, 0)
              nn.init.constant_(m.weight, 1.0)