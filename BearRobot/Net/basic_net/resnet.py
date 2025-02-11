import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import timm

norm_layer = {"bn": nn.BatchNorm2d, 
              "gn": nn.GroupNorm}

pooling_layer = {"avg": nn.AdaptiveAvgPool2d, 
                 "max": nn.AdaptiveMaxPool2d}

class ResNet(nn.Module):
       def __init__(self, 
              model_name: str = "resnet18",
              pretrained: bool = False,
              norm_type: str = "bn",
              pooling_type: str = "avg",
              add_spatial_coordinates: bool = False,
              use_alpha_channel: bool = False,
              return_interm_layers: bool = False,
              *args, **kwargs
       ):
              super().__init__(*args, **kwargs)
              pretraine_path = f"{os.path.expanduser('~')}/.cache/huggingface/hub/models--timm--{model_name}.a1_in1k/pytorch_model.bin"
              self.model = timm.create_model(model_name, 
                                             pretrained=pretrained,
                                             pretrained_cfg_overlay=dict(file=pretraine_path))

              self.norm_type = norm_type
              assert norm_type in norm_layer
              self.norm_layer = norm_layer[norm_type]
              
              assert pooling_type in pooling_layer
              self.pooling_layer = pooling_layer[pooling_type]
              
              if norm_type != "bn":
                     self._replace_bn()

              del self.model.fc
              self.model.fc = nn.Identity()
              
              # add spatial information to the image
              self.add_spatial_coordinates = add_spatial_coordinates
              self.use_alpha_channel = use_alpha_channel
              self.c_num = 3
              if self.add_spatial_coordinates:
                     self.spatial_coordinates = AddSpatialCoordinates(dtype=self.model.conv1.weight.dtype)
                     self.c_num += 2
              
              if self.use_alpha_channel:
                     self.c_num += 1
              
              if self.c_num > 3:
                     self.model.conv1 = nn.Conv2d(self.c_num, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
                     
              if return_interm_layers:
                     from torchvision.models._utils import IntermediateLayerGetter
                     return_layers = {'layer4': "0"}
                     self.model = IntermediateLayerGetter(self.model, return_layers)
              
              self.num_channels = 512 if model_name in ('resnet18', 'resnet34') else 2048
              
       def forward(self, img):
              if self.add_spatial_coordinates:
                     img = self.spatial_coordinates(img)
              output = self.model(img)
              return output
       
       def stem(self, img):
              if self.add_spatial_coordinates:
                     img = self.spatial_coordinates(img)
              output = self.model.conv1(img)
              output = self.model.bn1(output)
              output = self.model.act1(output)
              output = self.model.maxpool(output)
              return output
       
       def _replace_bn(self):
              root_module = self.model
              bn_list = [k.split('.') for k, m in self.model.named_modules() if isinstance(m, nn.BatchNorm2d)]
              
              for *parent, k in bn_list:
                     parent_module = root_module
                     if len(parent) > 0:
                            parent_module = root_module.get_submodule('.'.join(parent))
                     if isinstance(parent_module, nn.Sequential):
                            src_module = parent_module[int(k)]
                     else:
                            src_module = getattr(parent_module, k)
                     
                     tgt_module = self.norm_layer(src_module.num_features//16, src_module.num_features)
                     
                     if isinstance(parent_module, nn.Sequential):
                            parent_module[int(k)] = tgt_module
                     else:
                            setattr(parent_module, k, tgt_module)
                     
              # verify that all BN are replaced
              bn_list = [k.split('.') for k, m in self.model.named_modules() if isinstance(m, nn.BatchNorm2d)]
              assert len(bn_list) == 0
              

class AddSpatialCoordinates(nn.Module):
    def __init__(self, dtype=torch.float32):
        super(AddSpatialCoordinates, self).__init__()
        self.dtype = dtype

    def forward(self, x):
        grids = [
            torch.linspace(-1, 1, steps=s, device=x.device, dtype=self.dtype) 
            for s in x.shape[-2:]  # add spatial coordinates with HxW shape
        ]

        grid = torch.meshgrid(grids, indexing='ij')
        grid = torch.stack(grid, dim=0)
        
        # reshape to B*F*V, 2, H, W
        BFV, *_ = x.shape
        grid = grid.expand(BFV, *grid.shape)

        # cat on the channels dimension
        return torch.cat([x, grid], dim=-3)

   
              
if __name__ == '__main__':
       timmresnet = ResNet(model_name="resnet50", norm_type="gn")
       x = torch.randn(1, 3, 224, 224)
       print(timmresnet(x).shape)
      