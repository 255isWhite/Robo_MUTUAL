import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import copy

from BearRobot.Agent.base_agent import BaseAgent
from BearRobot.Net.my_model.RT_model import RT1Model
from BearRobot.Net.my_model.t5 import T5Encoder

class RT1Agent(BaseAgent):
       def __init__(
              self, 
              policy: RT1Model, 
              text_encoder = 't5',
              device = 'cuda'
       ):
              super().__init__(
                     policy,
                     None,
                     None
              )
              self.img_size = self.policy.img_size
              
              transform = [
                     transforms.Resize(256, interpolation=Image.BICUBIC),
                     transforms.CenterCrop(self.img_size),
                     transforms.ToTensor()
              ]
              self.transform = transforms.Compose(transform)
              assert text_encoder == 't5'
              self.lang_encoder = T5Encoder(device=device)
              print("lang encoder load success")
              self.device = device

       def forward(self, images: torch.Tensor, texts: list, action_gt: torch.Tensor, state=None):
              '''
              calculate RT1 loss / cross entropy
              # images: batch of frames of different views, [B, Frame, View, C, H, W]
              # texts: list of instructions
              # state shape [B, D_s], batch of robot arm x,y,z, gripper state, et al
              # action_gt shape [B, D_a], batch of robot control value, e.g., delta_x, delta_y, delta_z,..., et al.
              '''
              text_emb = self.lang_encoder.embed_text(texts).to(images.device).detach()
              loss = self.policy_loss(images, text_emb, action_gt, state)
              loss_dict = {"policy_loss": loss}
              return loss_dict

       def logits(self, images: torch.Tensor, texts: list, state=None):
              """
              predict the loogits
              # images: batch of frames of different views, [B, Frame, View, C, H, W]
              # texts: list of instructions
              # state shape [B, D] robot arm x,y,z, gripper state, et al
              """                    
              logits = self.policy(images, texts, state)
              return logits
       
       def policy_loss(self, images: torch.Tensor, texts: list, action_gt: torch.Tensor, state=None):
              '''
              calculate RT1 loss / cross entropy
              # images: batch of frames of different views, [B, Frame, View, C, H, W]
              # texts: list of instructions
              # state shape [B, D_s], batch of robot arm x,y,z, gripper state, et al
              # action_gt shape [B, D_a], batch of robot control value, e.g., delta_x, delta_y, delta_z,..., et al.
              '''
              logits = self.logits(images, texts, state)
              _, _, N = logits.shape  # [bs, a_dim, bins]
              logits = logits.view(-1, N)
              label = action_gt.view(-1).to(torch.int64)
              loss = F.cross_entropy(logits, label)
              return loss              
       
       def q_loss(self):
              raise ValueError("RT1 agent is a BC agent, has no q value")
       
       def v_loss(self):
              raise ValueError("RT1 agent is a BC agent, has no v value")
              
       @torch.no_grad()
       def get_action(self, images, text: list, state=None):
              """
              get one action
              # images: one frames of different views, [1, Frame, View, C, H, W]
              # texts: list of instruction
              # state shape [1, D] robot arm x,y,z, gripper state, et al
              """
              # TODO here has a bug
              if not isinstance(images, torch.Tensor):
                     # transform lists to torch.Tensor
                     images = [torch.stack([torch.stack([self.transform(view).reshape(3, self.img_size, self.img_size) for view in frame_list]) for frame_list in one_images]).unsqueeze(0) for one_images in images]
                     images = torch.cat(images, dim=0)  # tensor [1, Frame, View, C, H, W]
              
              logits = self.logits(images, text, state).view(-1, 256)  # [7, 256]
              joint = logits[:6].softmax(-1).max(-1).indices  # [6]
              joint = -1 + joint / 128  # decode to -1~1
              
              gripper = logits[-1].softmax(-1).max(-1).indices.unsqueeze(0) / 256 # [1]
              
              action = torch.cat([joint, gripper]).cpu().numpy()
              return action