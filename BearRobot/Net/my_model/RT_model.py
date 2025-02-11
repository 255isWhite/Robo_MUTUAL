import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import timm.models
from timm.models import create_model
from timm.models.layers import DropPath
from einops import repeat

from typing import Callable, Optional, Union, Tuple, List, Any

from BearRobot.Net.basic_net.mlp import MLP
from BearRobot.Net.my_model.t5 import T5Encoder
from BearRobot.utils.net.initialization import init_weight


def posemb_sincos_1d(seq, dim, temperature = 10000, device = None, dtype = torch.float32):
       n = torch.arange(seq, device = device)
       omega = torch.arange(dim // 2, device = device) / (dim // 2 - 1)
       omega = 1. / (temperature ** omega)

       n = n[:, None] * omega[None, :]
       pos_emb = torch.cat((n.sin(), n.cos()), dim = 1)
       return pos_emb.type(dtype)


class DecoderBlock(nn.Module):
       def __init__(
              self, 
              d_model: int=768,
              nhead: int=12,
              mlp_ratio: float=4.,
              attn_drop: float=0.,
              proj_drop: float=0.2,
              path_drop: float=0.2
       ):
              super().__init__()
              self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=attn_drop, batch_first=True)
              dim_feedforward = int(mlp_ratio * d_model)
              self.mlp = MLP(d_model, [dim_feedforward], d_model, 'gelu', proj_drop)
              self.norm1 = nn.LayerNorm(d_model)
              self.norm2 = nn.LayerNorm(d_model)
              self.drop_path = DropPath(path_drop) if path_drop > 0. else nn.Identity()
              self.apply(init_weight)

       def forward(self, x, attn_mask):
              # out -> B, 
              out = self.norm1(x)
              out = self.self_attn(out, out, out, attn_mask = attn_mask)[0]
              out = out + self.drop_path(out)
              out = self.norm2(out)
              out = self.mlp(out)
              out = out + self.drop_path(out)
              return out


class FiLM(nn.Module):
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


class TokenLearner(nn.Module):
       def __init__(
              self, 
              input_dim: int=768, 
              num_output_token: int=8
       ):
              super().__init__()
              self.layer_norm = nn.LayerNorm(input_dim)
              self.mlp_generator = MLP(input_dim, [input_dim], output_size=num_output_token, ac_fn='gelu')
              self.apply(init_weight)

        
       def forward(self, x: torch.tensor):
              B, N, C = x.shape
              weight = self.mlp_generator(self.layer_norm(x)).transpose(-1, -2)
              weight = weight.softmax(-1) # weight shape [B, W ,N] w->num_output_token
              x = torch.einsum("bnc,bwn->bwc", x, weight)
              return x # B, W, C



class RT1Model(nn.Module):
       def __init__(
              self, 
              num_actions_bins: int=256,
              num_actions: int=7,
              decoder_depth: int=6,
              img_size: int=224,
              num_output_token: int=8,
              condition_dim_mult: int=2,
              vision_encoder: str='maxvit_rmlp_base_rw_224',
              vision_pretrain: bool=True,
              state_dim=None,
              device: str='cpu',
              **kwargs
       ):
              super().__init__()
              self.device = device
              self.img_size = img_size
              self.num_action_bins = num_actions_bins
              self.num_actions = num_actions
              self.num_output_token = num_output_token
              
              print("-"*88)
              condition_dim = 768
              self.condition_dim = condition_dim * condition_dim_mult
              
              assert vision_encoder == 'maxvit_rmlp_base_rw_224'
              self.visual_encoder: timm.models.MaxxVit = create_model(
                     vision_encoder, 
                     pretrained = vision_pretrain,
                     num_classes = 0,
                     img_size = img_size,
                     drop_path_rate = 0.1
              )
              del self.visual_encoder.head
              print("vision encoder load success")
              
              ####### module need to be init #########
              self.visual_condition_projector = nn.Sequential(
                     nn.Linear(condition_dim, condition_dim * condition_dim_mult),
                     nn.SiLU()
              )
              
              self.head = nn.Sequential(
                     nn.LayerNorm(self.visual_encoder.feature_info[-1]['num_chs']),
                     nn.Linear(self.visual_encoder.feature_info[-1]['num_chs'], num_actions * num_actions_bins)
              )
              self.visual_condition_projector.apply(init_weight)
              self.head.apply(init_weight)
              ###### include other modules #######
              
              
              self.visual_conditioner = nn.ModuleList()
              for idx, stage in enumerate(self.visual_encoder.stages):
                     stage_depth = len(stage.blocks)
                     self.visual_conditioner.append(
                            FiLM(condition_dim=self.condition_dim, 
                            dim=self.visual_encoder.feature_info[idx]['num_chs']))
                     self.visual_conditioner.extend([
                            FiLM(condition_dim=self.condition_dim, 
                            dim=self.visual_encoder.feature_info[idx+1]['num_chs']) for _ in range(stage_depth-1)
                     ])                     

              self.token_learner = TokenLearner(
                     input_dim = self.visual_encoder.feature_info[-1]['num_chs'],
                     num_output_token = num_output_token
              )
              
              self.decoder = nn.ModuleList()
              for _ in range(decoder_depth):
                     decoder_block = DecoderBlock(self.visual_encoder.feature_info[-1]['num_chs'])
                     self.decoder.append(decoder_block)
                     
              self.state_encoder = nn.Linear(state_dim, self.visual_encoder.feature_info[-1]['num_chs']) if state_dim is not None else None
              print("model init success!")
              print("-"*88)
              
       
       def forward_image_feature(self, images: torch.Tensor, condition: torch.Tensor):
              # images shape [B, Frame, View, C, H, W]
              # condition shape [B, D]              
              assert images.shape[0] == condition.shape[0]
              B, F, V, C, H, W = images.shape
              B, D = condition.shape
              
              # text condition projection
              condition = self.visual_condition_projector(condition)
              condition = condition.unsqueeze(1).repeat(1, F*V, 1).view(B*F*V, self.condition_dim)
              
              # image projection
              image_feature = images.view(B * F * V, C, H, W)
              image_feature = self.visual_encoder.stem(image_feature)
              
              # Film
              visual_blocks = [block for stage in self.visual_encoder.stages for block in stage.blocks]
              for visual_block, condition_block in zip(visual_blocks, self.visual_conditioner):
                     image_feature = condition_block(condition, image_feature)
                     image_feature = visual_block(image_feature)
                     
              image_feature = self.visual_encoder.norm(image_feature)
              #image feature shape [B * F * V, C, H \ 32, W \ 32]
              _, C, H, W = image_feature.shape
              
              image_tokens = image_feature.view(B * F, V, C, H * W).transpose(-1, -2).reshape(B * F, V * H * W, C)
              image_tokens = self.token_learner(image_tokens)  # B * F, 8, C
              
              return image_tokens.view(B, F, self.num_output_token, C)
       
       def forward_decoder(self, image_tokens: torch.Tensor):
              # image_tokens shape [B, Frame, N, C]              
              B, F, N, C = image_tokens.shape
              
              attn_mask = torch.ones((F, F), dtype = torch.bool, device = image_tokens.device).triu(1)
              attn_mask = repeat(attn_mask, 'i j -> (i r1) (j r2)', r1=N, r2=N)
              
              # sinusoidal positional embedding
              pos_emb = posemb_sincos_1d(F, C, dtype = image_tokens.dtype, device = image_tokens.device)
              pos_emb = pos_emb.view(1, F, 1, C).repeat(1, 1, N, 1)
              
              image_tokens += pos_emb
              image_tokens = image_tokens.reshape(B, F * N, C)
              
              pre_feature = image_tokens
              for b in self.decoder:
                     pre_feature = b(pre_feature, attn_mask)  
                     
              pre_feature = pre_feature.view(B, F, N, C).mean(dim = 2)           
              return pre_feature
       
       def forward_head(self, pre_feature):
              # pre feature shape [B, F, C]
              B, F, C = pre_feature.shape
              logits = self.head(pre_feature) # -> B, F, Bins * Action
              logits = logits.view(B, F, self.num_actions, self.num_action_bins)
              
              return logits  # B, F, Action, Bins

       def forward(self, images: torch.Tensor, condition: torch.Tensor, state: torch.Tensor = None):
              # images shape [B, Frame, V, C, H, W]
              # state shape [B, Frame, D_s]
              # condition, text emb
              # images_view_2 [B, Frame, C, H, W]
              image_tokens = self.forward_image_feature(images, condition)
              B, F, N, D = image_tokens.shape

              if state is not None and self.state_encoder is not None:
                     state_token = self.state_encoder(state).view(B, F, 1, D)
                     image_tokens = torch.cat((image_tokens, state_token), dim = 2)

              pre_features = self.forward_decoder(image_tokens)
              logits = self.forward_head(pre_features)
              return logits[:, -1].contiguous() # B, Action, Bins
       
       
       
if __name__ == '__main__':
       rt1model = RT1Model()
       print(1)
              