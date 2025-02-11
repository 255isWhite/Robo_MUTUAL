import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

import timm.models
from timm.models import create_model
from timm.models.layers import DropPath
from einops import repeat, rearrange

from typing import Callable, Optional, Union, Tuple, List, Any

from BearRobot.Net.basic_net.mlp import MLP
from BearRobot.Net.my_model.t5 import T5Encoder
from BearRobot.Net.basic_net.transformer import Transformer, TransformerDecoder, TransformerDecoderLayer, TransformerEncoder, TransformerEncoderLayer
from BearRobot.utils.net.initialization import init_weight


class SinusoidalPosEmb(nn.Module):
       def __init__(self, input_size, output_size):
              super().__init__()
              self.output_size = output_size
              
       def forward(self, x):
              device = x.device
              half_dim = self.output_size // 2
              f = math.log(10000) / (half_dim - 1)
              f = torch.exp(torch.arange(half_dim, device=device) * -f)
              f = x * f[None, :]
              f = torch.cat([f.cos(), f.sin()], axis=-1)
              return f

# learned positional embeds
class LearnedPosEmb(nn.Module):
       def __init__(self, input_size, output_size):
              super().__init__()
              self.output_size = output_size
              self.kernel = nn.Parameter(torch.randn(output_size // 2, input_size) * 0.2)
              
       def forward(self, x):
              f = 2 * torch.pi * x @ self.kernel.T
              f = torch.cat([f.cos(), f.sin()], axis=-1)
              return f       

TIMEEMBED = {"fixed": SinusoidalPosEmb,
             "learned": LearnedPosEmb}


def get_sinusoid_encoding_table(n_position, d_hid):
    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    return torch.FloatTensor(sinusoid_table).unsqueeze(0)


def reparametrize(mu, logvar):
    std = logvar.div(2).exp()
    eps = Variable(std.data.new(std.size()).normal_())
    return mu + std * eps


class ACTModel(nn.Module):
       def __init__(
              self, 
              output_dim: int=7,  # a dim
              ac_num: int=4,
              s_dim: int=0,  # qpos_dim, use qpos when > 0 
              hidden_dim: int=256,
              dim_feedforward: int=2048,
              nheads: int=8,
              cond_dim: int=768,  # s dim, if condition on s
              dropout: float=0.1,
              num_encoder_layers: int=4,
              num_decoder_layers: int=6,
              visual_encoder: str='resnet18',
              visual_pretrain: bool=False,
              ft_vision: bool=True,
              use_alpha_channel: bool=False,
              return_interm_layers: bool=True,
              device: str='cpu',
              *args,
              **kwargs
       ):
              """A modified version from ACT implementation

              Args:
                  view_num (int, optional): _description_. Defaults to 2.
                  output_dim (int, optional): _description_. Defaults to 7.
                  s_dim (int, optional): qpos_dim, use qpos when > 0.
                  hidden_dim (int, optional): _description_. Defaults to 256.
                  dim_feedforward (int, optional): _description_. Defaults to 2048.
                  nheads (int, optional): _description_. Defaults to 8.
                  cond_dim (int, optional): _description_. Defaults to 768.
                  ifconditiononsdropout (float, optional): _description_. Defaults to 0.1.
                  num_encoder_layers (int, optional): _description_. Defaults to 4.
                  num_decoder_layers (int, optional): _description_. Defaults to 6.
                  visual_encoder (str, optional): _description_. Defaults to 'resnet18'.
                  visual_pretrain (bool, optional): _description_. Defaults to False.
                  ft_vision (bool, optional): _description_. Defaults to True.
                  add_spatial_coordinates (bool, optional): _description_. Defaults to False.
                  use_alpha_channel (bool, optional): _description_. Defaults to False.
                  return_interm_layers (bool, optional): _description_. Defaults to True.
                  device (str, optional): _description_. Defaults to 'cpu'.
              """
              super().__init__()
              
              # transformer encoder
              self.ac_num = ac_num
              self.encoder = build_encoder(hidden_dim=hidden_dim,
                                           dropout=dropout,
                                           nhead=nheads,
                                           dim_feedforward=dim_feedforward,
                                           num_encoder_layers=num_encoder_layers,
                                           pre_norm=False)

              # cnn backbone
              self.ft_vision = ft_vision
              self.backbone = build_backbone(hidden_dim=hidden_dim,
                                             backbone=visual_encoder,
                                             pretrained=visual_pretrain,
                                             use_alpha_channel=use_alpha_channel,
                                             return_interm_layers=return_interm_layers)

              self.input_proj = nn.Conv2d(self.backbone.num_channels, hidden_dim, kernel_size=1)
              self.input_proj_robot_state = nn.Linear(output_dim, hidden_dim)
              if not ft_vision:
                     for param in self.backbone.parameters():
                            param.requires_grad =False
              
              # transformer decorder (DETR architecture)
              self.transformer = build_transformer(hidden_dim=hidden_dim,
                                                   dim_feedforward=dim_feedforward,
                                                   dropout=dropout,
                                                   nheads=nheads,
                                                   enc_layers=num_encoder_layers,
                                                   dec_layers=num_decoder_layers,
                                                   pre_norm=False,
                                                   cond_dim=cond_dim)

              hidden_dim = self.transformer.d_model
              self.action_head = nn.Linear(hidden_dim, output_dim)
              self.is_pad_head = nn.Linear(hidden_dim, 1)
              self.query_embed = nn.Embedding(ac_num, hidden_dim)
              
              # encoder extra parameters
              self.s_dim = s_dim
              encode_token_num = 1 + ac_num # [CLS], a_seq
              self.latent_dim = 32 # final size of latent z # TODO tune
              self.cls_embed = nn.Embedding(1, hidden_dim)  # extra cls token embedding
              self.encoder_action_proj = nn.Linear(output_dim, hidden_dim)  # project action to embedding
              self.encoder_joint_proj = nn.Linear(output_dim, hidden_dim)  # project qpos to embedding
              if s_dim > 0:
                     encode_token_num += 1  # [CLS], qpos, a_seq
              self.latent_proj = nn.Linear(hidden_dim, self.latent_dim*2) # project hidden state to latent std, var
              self.register_buffer('pos_table', get_sinusoid_encoding_table(encode_token_num, hidden_dim))
              
              # decoder extra parameters
              self.latent_out_proj = nn.Linear(self.latent_dim, hidden_dim) # project latent sample to embedding
              additional_pos_embed_num = 2 if s_dim > 0 else 1
              self.additional_pos_embed = nn.Embedding(additional_pos_embed_num, hidden_dim) # learned position embedding for proprio and latent         
                   
              self.device = device
              

       def forward(self, qpos: torch.Tensor, image: torch.Tensor, actions=None, cond: torch.Tensor=None, is_pad=None):
              """
              qpos: batch, qpos_dim
              image: batch, num_cam, channel, height, width
              actions: batch, seq, action_dim,
              current is_pad is not used. All data is not padded
              """
              
              is_training = actions is not None # train or val
              bs, _ = qpos.shape
              if is_training:
                     # project action sequence to embedding dim, and concat with a CLS token
                     action_embed = self.encoder_action_proj(actions) # (bs, seq, hidden_dim)
                     cls_embed = self.cls_embed.weight # (1, hidden_dim)
                     cls_embed = torch.unsqueeze(cls_embed, axis=0).repeat(bs, 1, 1) # (bs, 1, hidden_dim)
                     if self.s_dim > 0:
                            # use qpos information
                            qpos_embed = self.encoder_joint_proj(qpos)  # (bs, hidden_dim)
                            qpos_embed = torch.unsqueeze(qpos_embed, axis=1)  # (bs, 1, hidden_dim)
                            encoder_input = torch.cat([cls_embed, qpos_embed, action_embed], axis=1) # (bs, seq+1+1, hidden_dim)
                     else:
                            # do not use qpos information
                            encoder_input = torch.cat([cls_embed, action_embed], axis=1) # (bs, seq+1, hidden_dim)
                     encoder_input = encoder_input.permute(1, 0, 2) # (seq+1+1, bs, hidden_dim)
                     
                     # do not mask cls token
                     cls_joint_is_pad = torch.full((bs, 2), False).to(qpos.device) if self.s_dim > 0 else torch.full((bs, 1), False).to(qpos.device)  # False: not a padding
                     is_pad = torch.full((bs, self.ac_num), False).to(qpos.device) # False: not a padding
                     is_pad = torch.cat([cls_joint_is_pad, is_pad], axis=1)  # (bs, seq+1+1)
                     
                     # obtain position embedding
                     pos_embed = self.pos_table.clone().detach()
                     pos_embed = pos_embed.permute(1, 0, 2)  # (seq+1+1, 1, hidden_dim)
                     
                     # query model
                     encoder_output = self.encoder(encoder_input, pos=pos_embed, src_key_padding_mask=is_pad)
                     encoder_output = encoder_output[0] # take cls output only
                     latent_info = self.latent_proj(encoder_output)
                     mu = latent_info[:, :self.latent_dim]
                     logvar = latent_info[:, self.latent_dim:]
                     latent_sample = reparametrize(mu, logvar)
                     latent_input = self.latent_out_proj(latent_sample)
              else:
                     mu = logvar = None
                     latent_sample = torch.zeros([bs, self.latent_dim], dtype=torch.float32).to(qpos.device)
                     latent_input = self.latent_out_proj(latent_sample)
                     
              # Image observation features and position embeddings
              B, F, V, C, H, W = image.shape
              image = image.view(B*F*V, C, H, W)
              image_features, pos = self.backbone(image)
              
              image_features = self.input_proj(image_features[0]) # take the last layer feature
              pos = pos[0].repeat(F*V, 1, 1, 1)

              # proprioception features
              proprio_input = self.input_proj_robot_state(qpos) if self.s_dim > 0 else None
              # fold camera dimension into width dimension
              src = rearrange(image_features, '(b f v) c h w -> (b f) c h (w v)', v=V, f=F)
              pos = rearrange(pos, '(f v) c h w -> f c h (w v)', v=V, f=F)
              hs = self.transformer(src, None, self.query_embed.weight, pos, latent_input, proprio_input, self.additional_pos_embed.weight, cond)[0][-1]
              
              a_hat = self.action_head(hs)
              is_pad_hat = self.is_pad_head(hs)
              return a_hat, is_pad_hat, [mu, logvar]


def build_encoder(hidden_dim: int=256, 
                  dropout: float=0.1,
                  nhead: int=8,
                  dim_feedforward: int=2048,
                  num_encoder_layers: int=4,
                  pre_norm: bool=False,
       ):
       d_model = hidden_dim # 256    
       dropout = dropout # 0.1
       nhead = nhead # 8
       dim_feedforward = dim_feedforward # 2048
       num_encoder_layers = num_encoder_layers # 4 # TODO shared with VAE decoder
       normalize_before = pre_norm # False
       activation = "relu"

       encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation, normalize_before)
       encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
       encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)
       return encoder
       

def build_backbone(hidden_dim: int=256,
                   backbone: str='resnet18',
                   pretrained: bool=False,
                   use_alpha_channel: bool=False,
                   return_interm_layers: bool=False,
                   ):
       from BearRobot.Net.basic_net.resnet import ResNet
       position_embedding = build_position_encoding(hidden_dim=hidden_dim)
       return_interm_layers = return_interm_layers
       backbone = ResNet(backbone, 
                         pretrained,
                         use_alpha_channel=use_alpha_channel,
                         return_interm_layers=return_interm_layers)
       
       model = Joiner(backbone, position_embedding)
       model.num_channels = backbone.num_channels
       return model
       
       
def build_transformer(hidden_dim: int=256,
                      cond_dim: int=768, # default t5 encoder
                      dropout: float=0.1,
                      nheads: int=8,
                      dim_feedforward: int=2048,
                      enc_layers: int=4,
                      dec_layers: int=4,
                      pre_norm: bool=False,
                      ):
       return Transformer(
              d_model=hidden_dim,
              cond_dim=cond_dim,
              dropout=dropout,
              nhead=nheads,
              dim_feedforward=dim_feedforward,
              num_encoder_layers=enc_layers,
              num_decoder_layers=dec_layers,
              normalize_before=pre_norm,
              return_intermediate_dec=True,
       )


def build_position_encoding(hidden_dim: int=256):
       N_steps = hidden_dim // 2
       # TODO find a better way of exposing other arguments
       position_embedding = PositionEmbeddingSine(N_steps, normalize=True)
       # position_embedding = PositionEmbeddingLearned(N_steps)

       return position_embedding


class NestedTensor(object):
       def __init__(self, tensors, mask: Optional[torch.Tensor]):
              self.tensors = tensors
              self.mask = mask

       def to(self, device):
              # type: (Device) -> NestedTensor # noqa
              cast_tensor = self.tensors.to(device)
              mask = self.mask
              if mask is not None:
                     assert mask is not None
                     cast_mask = mask.to(device)
              else:
                     cast_mask = None
              return NestedTensor(cast_tensor, cast_mask)

       def decompose(self):
              return self.tensors, self.mask

       def __repr__(self):
              return str(self.tensors)


class Joiner(nn.Sequential):
       def __init__(self, backbone, position_embedding):
              super().__init__(backbone, position_embedding)

       def forward(self, tensor_list: NestedTensor):
              xs = self[0](tensor_list)
              out: List[NestedTensor] = []
              pos = []
              for name, x in xs.items():
                     out.append(x)
                     # position encoding
                     pos.append(self[1](x).to(x.dtype))

              return out, pos


class PositionEmbeddingSine(nn.Module):
       """
       This is a more standard version of the position embedding, very similar to the one
       used by the Attention is all you need paper, generalized to work on images.
       """
       def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
              super().__init__()
              self.num_pos_feats = num_pos_feats
              self.temperature = temperature
              self.normalize = normalize
              if scale is not None and normalize is False:
                     raise ValueError("normalize should be True if scale is passed")
              if scale is None:
                     scale = 2 * math.pi
                     self.scale = scale


       def forward(self, tensor):
              x = tensor   # B, C, H, W
              # mask = tensor_list.mask
              # assert mask is not None
              # not_mask = ~mask

              not_mask = torch.ones_like(x[0, [0]])  # 1, C, H, W
              y_embed = not_mask.cumsum(1, dtype=torch.float32) # 1, C, H, W
              x_embed = not_mask.cumsum(2, dtype=torch.float32) # 1, C, H, W 
              if self.normalize:
                     eps = 1e-6
                     y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
                     x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

              dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)  # 1, C, H, W
              dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

              pos_x = x_embed[:, :, :, None] / dim_t
              pos_y = y_embed[:, :, :, None] / dim_t
              pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
              pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
              pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)  # B, C, H, W
              return pos


class PositionEmbeddingLearned(nn.Module):
       """
       Absolute pos embedding, learned.
       """
       def __init__(self, num_pos_feats=256):
              super().__init__()
              self.row_embed = nn.Embedding(50, num_pos_feats)
              self.col_embed = nn.Embedding(50, num_pos_feats)
              self.reset_parameters()

       def reset_parameters(self):
              nn.init.uniform_(self.row_embed.weight)
              nn.init.uniform_(self.col_embed.weight)

       def forward(self, tensor_list: NestedTensor):
              x = tensor_list.tensors
              h, w = x.shape[-2:]
              i = torch.arange(w, device=x.device)
              j = torch.arange(h, device=x.device)
              x_emb = self.col_embed(i)
              y_emb = self.row_embed(j)
              pos = torch.cat([
              x_emb.unsqueeze(0).repeat(h, 1, 1),
              y_emb.unsqueeze(1).repeat(1, w, 1),
              ], dim=-1).permute(2, 0, 1).unsqueeze(0).repeat(x.shape[0], 1, 1, 1)
              return pos

       
if __name__ == '__main__':
       pass