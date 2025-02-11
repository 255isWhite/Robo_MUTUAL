import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import copy

from BearRobot.Agent.base_agent import BaseAgent
from BearRobot.Net.my_model.ACT_model import ACTModel
from BearRobot.Net.my_model.t5 import T5Encoder

def kl_divergence(mu, logvar):
    batch_size = mu.size(0)
    assert batch_size != 0
    if mu.data.ndimension() == 4:
        mu = mu.view(mu.size(0), mu.size(1))
    if logvar.data.ndimension() == 4:
        logvar = logvar.view(logvar.size(0), logvar.size(1))

    klds = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    total_kld = klds.sum(1).mean(0, True)
    dimension_wise_kld = klds.mean(0)
    mean_kld = klds.mean(1).mean(0, True)

    return total_kld, dimension_wise_kld, mean_kld


class ACTAgent(BaseAgent):
       def __init__(
              self, 
              policy: ACTModel, 
              text_encoder = 't5',
              kl_weight: float=10,
              loss_type: str="l1",
              device = 'cuda'
       ):
              super().__init__(
                     policy,
                     None,
                     None
              )
              
              assert text_encoder == 't5'
              self.lang_encoder = T5Encoder(device=device)
              print("lang encoder load success")
              
              self.kl_weight = kl_weight
              if loss_type == "l1":
                     self.loss_fn = nn.L1Loss()
              elif loss_type == 'l2':
                     self.loss_fn = nn.MSELoss()
              else:
                     raise ValueError(f"{loss_type} is not supported, only support l1 and l2 loss")
              self.device = device

       def forward(self, images: torch.Tensor, texts: list, action_gt: torch.Tensor, state=None):
              '''
              calculate ACT loss (L1 or L2) and CVAE loss
              # images: batch of frames of different views, [B, Frame, View, C, H, W]
              # texts: list of instructions
              # state shape [B, D_s], batch of robot arm x,y,z, gripper state, et al
              # action_gt shape [B, D_a], batch of robot control value, e.g., delta_x, delta_y, delta_z,..., et al.
              '''
              text_emb = self.lang_encoder.embed_text(texts).to(images.device).detach()
              B, _ = action_gt.shape
              action_gt = action_gt.reshape(B, -1, 7)
              loss, recons_loss, kl_loss = self.policy_loss(images, text_emb, action_gt, state)
              loss_dict = {"policy_loss": loss,
                           "recons_loss": recons_loss,
                           "kl_loss": kl_loss}
              return loss_dict
       
       def policy_loss(self, images: torch.Tensor, text_emb: torch.Tensor, action_gt: torch.Tensor, state=None):
              '''
              calculate ACT loss (L1 or L2) and CVAE loss
              # images: batch of frames of different views, [B, Frame, View, C, H, W]
              # text_emb: text embedding, [B, N]
              # state shape [B, D_s], batch of robot arm x,y,z, gripper state, et al
              # action_gt shape [B, D_a], batch of robot control value, e.g., delta_x, delta_y, delta_z,..., et al.
              '''
              a_hat, is_pad_hat, (mu, logvar) = self.policy(state, images, action_gt, text_emb)
              
              total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)
              recons_loss = self.loss_fn(action_gt, a_hat)
              kl_loss = self.kl_weight * total_kld[0]
              loss = recons_loss + kl_loss
              return loss, recons_loss, kl_loss
       
       def q_loss(self):
              raise ValueError("ACT agent is a BC agent, has no q value")
       
       def v_loss(self):
              raise ValueError("ACT agent is a BC agent, has no v value")
              
       @torch.no_grad()
       def get_action(self, imgs, text: list, state=None):
              """
              get one action
              # imgs: one frames of different views, [1, Frame, View, C, H, W]
              # texts: list of instruction
              # state shape [1, D] robot arm x,y,z, gripper state, et al
              """
              if not isinstance(imgs, torch.Tensor):
                     # transform lists to torch.Tensor
                     imgs = torch.stack([self.transform(Image.fromarray(frame).convert("RGB")) for frame in imgs]).unsqueeze(0).unsqueeze(0).to('cuda')

              state = torch.from_numpy(state.astype(np.float32)).view(1, -1) if state is not None else None
              state = (state - self.s_mean) / self.s_std if state is not None else None
              
              text_emb = self.lang_encoder.embed_text([text]).to(imgs.device).detach()
              action, _, (_, _) = self.policy(state, imgs, cond=text_emb)

              action = action.squeeze(0).detach().cpu()
              N, _ = action.shape
              a_max = self.a_max.repeat(N, 1)
              a_min = self.a_min.repeat(N, 1)
              a_mean = self.a_mean.repeat(N, 1)
              a_std = self.a_std.repeat(N, 1)
              action = (action + 1) * (a_max - a_min) / 2 + a_min
              return action.numpy()