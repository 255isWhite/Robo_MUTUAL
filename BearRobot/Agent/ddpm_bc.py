import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
import copy

from PIL import Image

from BearRobot.Agent.base_agent import BaseAgent
from BearRobot.Net.my_model.diffusion_model import VisualDiffusion
from BearRobot.Net.my_model.t5 import T5Encoder
from BearRobot.Net.my_model.clip_encoder import ClipEncoder
from BearRobot.Net.encoder.DecisionNCE import DecisionNCE_encoder, DecisionNCE_lang, DecisionNCE_visual_diff
from BearRobot.Net.my_model.Align_net import AlignNet

def extract(a, x_shape):
       '''
       align the dimention of alphas_cumprod_t to x_shape
       
       a: alphas_cumprod_t, B
       x_shape: B x F x F x F
       output: alphas_cumprod_t B x 1 x 1 x 1]
       '''
       b, *_ = a.shape
       return a.reshape(b, *((1,) * (len(x_shape) - 1)))


def linear_beta_schedule(timesteps):
       """
       linear schedule, proposed in original ddpm paper
       """
       scale = 1000 / timesteps
       beta_start = scale * 0.0001
       beta_end = scale * 0.02
       return torch.linspace(beta_start, beta_end, timesteps)


def cosine_beta_schedule(timesteps, s=0.008):
       """
       cosine schedule
       as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
       """
       steps = timesteps + 1
       t = torch.linspace(0, timesteps, steps) / timesteps
       alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
       alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
       betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
       return torch.clip(betas, 0, 0.999)


def sigmoid_beta_schedule(timesteps, start = -3, end = 3, tau = 1, clamp_min = 1e-5):
       """
       sigmoid schedule
       proposed in https://arxiv.org/abs/2212.11972 - Figure 8
       better for images > 64x64, when used during training
       """
       steps = timesteps + 1
       t = torch.linspace(0, timesteps, steps) / timesteps
       v_start = torch.tensor(start / tau).sigmoid()
       v_end = torch.tensor(end / tau).sigmoid()
       alphas_cumprod = (-((t * (end - start) + start) / tau).sigmoid() + v_end) / (v_end - v_start)
       alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
       betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
       return torch.clip(betas, 0, 0.999)


def vp_beta_schedule(timesteps):
       """Discret VP noise schedule
       """
       t = torch.arange(1, timesteps + 1)
       T = timesteps
       b_max = 10.
       b_min = 0.1
       alpha = torch.exp(-b_min / T - 0.5 * (b_max - b_min) * (2 * t - 1) / T ** 2)
       betas = 1 - alpha
       return betas       

def generate_vector_with_given_cosine_similarity(vector, target_similarity):
       """
       生成一个与给定向量有特定余弦相似度的向量。
       
       :param vector: 原始向量，大小为 [10, 1024]
       :param target_similarity: 目标余弦相似度，大小为 [10]
       :return: 与原始向量有指定余弦相似度的向量
       """
       # print("vector size:", vector.size())
       # print("target_similarity size:", target_similarity.size())

       # 归一化原始向量
       unit_vector = vector / torch.norm(vector, dim=1, keepdim=True)
       
       # 生成一个与原始向量正交的随机向量
       random_vector = torch.randn_like(vector)
       random_vector -= torch.sum(random_vector * unit_vector, dim=1, keepdim=True) * unit_vector
       random_vector /= torch.norm(random_vector, dim=1, keepdim=True)
       
       # 线性组合得到目标向量
       target_vector = (target_similarity.view(-1, 1) * unit_vector + 
                            torch.sqrt(1 - target_similarity.view(-1, 1)**2) * random_vector)
       
       return target_vector
       

SCHEDULE = {'linear': linear_beta_schedule,
            'cosine': cosine_beta_schedule,
            'sigmoid': sigmoid_beta_schedule,
            'vp': vp_beta_schedule}
                     
class DDPM_BC(BaseAgent):
       def __init__(
              self, 
              policy: torch.nn.Module, 
              beta: str='cosine',
              T: int=5,
              ac_num: int=1,
       ):
              super().__init__(
                     policy, None, None, ac_num=ac_num
              )
              
              self.device = policy.device
              if beta not in SCHEDULE.keys():
                     raise ValueError(f"Invalid schedule '{beta}'. Expected one of: {list(SCHEDULE.keys())}")
              self.schedule = SCHEDULE[beta]
              
              self.num_timesteps = T
              self.betas = self.schedule(self.num_timesteps).to(self.device)
              self.alphas = (1 - self.betas).to(self.device)
              self.alphas_cumprod = torch.cumprod(self.alphas, dim=0).to(self.device)

       def forward(self, xt: torch.Tensor, t: torch.Tensor, cond: torch.Tensor=None):
              """
              predict the noise
              Input:
              xt: noisy samples, here xt is typically the ground truth action
              t: timestep
              cond: condition information, here cond is typically the state rather than other condition like language or goal image
              
              Return: predicted noise
              """
              noise_pred = self.policy(xt, t, cond)
              return noise_pred
       
       def predict_noise(self, xt: torch.Tensor, t: torch.Tensor, cond: torch.Tensor=None):
              """
              predict the noise
              
              Input:
              xt: noisy samples, here xt is typically the noisy ground truth action
              t: timestep
              cond: condition information, here cond is typically the state rather than other condition like language or goal image
              
              Return: predicted noise
              """
              noise_pred = self.policy(xt, t, cond)
              return noise_pred
       
       def policy_loss(self, x0: torch.Tensor, cond: torch.Tensor=None):
              '''
              calculate ddpm loss
              
              Input:
              x0: ground truth value, here x0 is typically the ground truth action
              cond: condition information, here cond is typically the state rather than other condition like language or goal image
              
              Return: ddpm loss
              '''
              batch_size = x0.shape[0]
              
              noise = torch.randn_like(x0, device=self.device)
              t = torch.randint(0, self.num_timesteps, (batch_size, ), device=self.device)
              
              xt = self.q_sample(x0, t, noise)
              
              noise_pred = self.predict_noise(xt, t, cond)
              loss = (((noise_pred - noise) ** 2).sum(axis = -1)).mean()
              
              return loss
              
       def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor):
              """
              sample noisy xt from x0, q(xt|x0), forward process
              
              Input:
              x0: ground truth value, here x0 is typically the ground truth action
              t: timestep
              noise: noise
              
              Return: noisy samples
              """
              alphas_cumprod_t = self.alphas_cumprod[t]
              xt = x0 * extract(torch.sqrt(alphas_cumprod_t), x0.shape) + noise * extract(torch.sqrt(1 - alphas_cumprod_t), x0.shape)
              return xt
       
       @torch.no_grad()
       def p_sample(self, xt: torch.Tensor, t: torch.Tensor, noise_pred: torch.Tensor, guidance_strength=0, clip_sample=False, ddpm_temperature=1.):
              """
              sample xt-1 from xt, p(xt-1|xt)
              
              Input:
              xt: noisy samples, here xt is typically the noisy ground truth action
              t: timestep
              noise_pred: predicted noise
              guidance_strength: the strength of the guidance
              clip_sample: whether to clip the sample to [-1, 1]
              ddpm_temperature: the temperature of the noise
              
              Return: sample xt-1
              """
              alpha1 = 1 / torch.sqrt(self.alphas[t])
              alpha2 = (1 - self.alphas[t]) / (torch.sqrt(1 - self.alphas_cumprod[t]))
              
              xtm1 = alpha1 * (xt - alpha2 * noise_pred)
              
              noise = torch.randn_like(xtm1, device=self.device) * ddpm_temperature
              xtm1 = xtm1 + (t > 0) * (torch.sqrt(self.betas[t]) * noise)
              
              if clip_sample:
                     xtm1 = torch.clip(xtm1, -1., 1.)
              return xtm1
       
       @torch.no_grad()
       def ddpm_sampler(self, shape, cond=None, guidance_strength=0, clip_sample=False):
              """
              sample x0 from xT, reverse process
              
              Input:
              shape: the desired shape of the sample
              cond: condition information, here cond is typically the state rather than other condition like language or goal image
              guidance_strength: the strength of the guidance
              clip_sample: whether to clip the sample to [-1, 1]
              
              Return: sampled x0
              """
              x = torch.randn(shape, device=self.device)
              cond = cond.repeat(x.shape[0], 1)
              
              for t in reversed(range(self.num_timesteps)):
                     noise_pred = self.predict_noise(x, t, cond)
                     x = self.p_sample(x, torch.full((shape[0], 1), t, device=self.device, clip_sample=clip_sample), noise_pred)
              return x

       @torch.no_grad()
       def get_action(self, state, num=1, clip_sample=False):
              return self.ddpm_sampler((num, self.policy.output_dim), cond=state, clip_sample=clip_sample)
       

class DDPM_BC_latent(DDPM_BC):
       """
       this is a video prediction diffusion model in the latent space, based on DDPM
       """
       def __init__(
              self, 
              policy: torch.nn.Module, 
              schedule: str='cosine',  
              num_timesteps: int=5,
              mmencoder: torch.nn.Module=None,            
              *args, **kwargs) -> None:
              super().__init__(policy, schedule, num_timesteps)
              
              self.mmencoder = mmencoder
       

       def forward(self, cond_img: torch.Tensor=None, cond_lang: list=None, img: torch.Tensor=None):
              """
              img: torch.tensor, s_t+1, the predicted img
              cond_img: torch.tensor, s_t, the current observed img
              cond_lang: list, the language instruction
              """
              img, cond = self.encode(img, cond_img, cond_lang)
              loss = self.policy_loss(img, cond)
              return loss


       def encode(self, img: torch.Tensor, cond_img: torch.Tensor=None, cond_lang: list=None):
              """
              encode the img and language instruction into latent space
              img: [B, C, H, W] torch.tensor, s_t+1, the predicted img
              cond_img: [B, F, C, H, W] torch.tensor, s_t, s_{t-1}, s_{t-2}, ..., s_{t-F}, the current observed history imgs
              cond_lang: list, the language instruction
              """
              with torch.no_grad():
                     try:
                            img = self.mmencoder.encode_image(img)
                     except:
                            img = None
                     
                     B, F, C, H, W = cond_img.shape
                     cond_img = cond_img.view(B * F, C, H, W)
                     cond_img = self.mmencoder.encode_image(cond_img) if cond_img is not None else None
                     cond_img = cond_img.view(B, -1)
                     cond_lang = self.mmencoder.encode_lang(cond_lang) if cond_lang is not None else None
                     
                     try:
                            cond = torch.cat([cond_img, cond_lang], dim=-1)
                     except:
                            try:
                                   cond = cond_img
                            except:
                                   cond = None
              return img, cond 
       
       @torch.no_grad()
       def get_action(self, cond_img: torch.Tensor, cond_lang: list, num=1, clip_sample=False):
              _, cond = self.encode(None, cond_img, cond_lang)
              return self.ddpm_sampler((num, self.policy.output_dim), cond=cond, clip_sample=clip_sample)
              # TODO implement DDIM sampler to accelerate the sampling
              
              

class VLDDPM_BC(DDPM_BC):
       """
       this is a vision-language diffusion agent, based on DDPM
       """
       def __init__(
              self, 
              policy: VisualDiffusion, 
              beta: str='cosine',  
              T: int=5,
              ac_num: int=1,
              text_encoder: str="T5",
              device = 'cuda',
              *args, **kwargs
       ):
              super().__init__(policy, beta, T, ac_num)
              
              self.text_encoder = text_encoder
              # self.img_size = self.policy.img_size
              assert text_encoder in ['T5', 'DecisionNCE-T', 'DecisionNCE-P', "DecisionNCE-V", "CLIP", "CUT", "EPIC"]
              if text_encoder == 'T5':
                     self.lang_encoder = T5Encoder(device=device)
              elif text_encoder in ['DecisionNCE-T', 'DecisionNCE-P']:
                     mm_encoder = DecisionNCE_encoder(text_encoder, device=device)
                     self.lang_encoder = DecisionNCE_lang(mm_encoder)
              elif text_encoder == "DecisionNCE-V":
                     text_encoder = 'r50_frozen_7800ep'
                     mm_encoder = DecisionNCE_encoder(text_encoder, device=device)
                     self.frame_diff_encoder = DecisionNCE_visual_diff(mm_encoder)
                     self.lang_encoder = DecisionNCE_visual_diff(mm_encoder)
              elif text_encoder == "CUT":
                     text_encoder = 'r50_frozen_7800ep'
                     mm_encoder = DecisionNCE_encoder(text_encoder, device=device)
                     self.frame_diff_encoder = DecisionNCE_visual_diff(mm_encoder)
                     self.lang_encoder = DecisionNCE_visual_diff(mm_encoder)
              elif text_encoder == "EPIC":
                     text_encoder = 'DecisionNCE-T_epick'
                     mm_encoder = DecisionNCE_encoder(text_encoder, device=device)
                     self.frame_diff_encoder = DecisionNCE_visual_diff(mm_encoder)
                     self.lang_encoder = DecisionNCE_visual_diff(mm_encoder)
              elif text_encoder == "CLIP":
                     self.lang_encoder = ClipEncoder(device=device)
              else:
                     raise ValueError(f"Invalid text_encoder '{text_encoder}'. Expected one of: ['t5', 'DecisionNCE-T', 'DecisionNCE-P']")
              print("lang encoder load success")
              print(f" text encoder is {text_encoder}")
              self.device = device
              
              try:
                     self.add_noise = kwargs['add_noise']
                     self.cos_noise = kwargs['cos_noise']
                     self.cos_noise_decay = kwargs['cos_noise_decay']
              except:
                     self.add_noise = False
                     
              try:
                     self.noise_data_path = kwargs['noise_data_path']
                     self.noise_data = np.load(self.noise_data_path)
                     self.noise_std = self.noise_data['std_noise']
              except:
                     self.noise_data_path = 'WRONG LOGIC PATH'
              
              try:
                     self.minus_mean = kwargs['minus_mean']
                     self.mean_data_path = kwargs['mean_data_path']
                     self.avg_data = np.load(self.mean_data_path)
                     self.mean_traj_emb = self.avg_data['mean_traj_emb']
                     self.mean_lang_emb = self.avg_data['mean_lang_emb']
              except:
                     self.minus_mean = False
                     self.mean_data_path = 'WRONG LOGIC PATH'
      
              try:
                     self.lang_fit_img = kwargs['lang_fit_img']
              except:
                     self.lang_fit_img = False
              
              try:
                     self.cut_path = kwargs['cut_path']
                     self.topk_index = np.load(self.cut_path)['topk_index']
              except:
                     self.topk_index = None
                     self.cut_path = "NO CUT"
                     print(f"topk_index is None, while text_encoder is {text_encoder}")
                     

              print("_____________________________")
              print("self.text_encoder:", self.text_encoder)
              print("slef.add_noise:", self.add_noise)
              print("slef.minus_mean:", self.minus_mean)
              print("slef.cos_noise:", self.cos_noise)
              print("1-self.cos_noise:", self.cos_noise_decay)
              print("mean file path:", self.mean_data_path)
              print("cut path:", self.cut_path)
              print("noise_data_path:",self.noise_data_path)
              if self.cos_noise < 9.0:
                     print("Cosine noise is injected!!!")
              else:
                     print("STD noise is injected!!!")
              print("_____________________________")
                     
              
       def forward(self, images: torch.Tensor, cond: dict, action_gt: torch.Tensor, state=None, img_goal=False):
              '''
              calculate ddpm loss
              # images: batch of frames of different views, [B, Frame, View, C, H, W]
              # texts: list of instructions
              # state shape [B, D_s], batch of robot arm x,y,z, gripper state, et al
              # action_gt shape [B, D_a], batch of robot control value, e.g., delta_x, delta_y, delta_z,..., et al.
              # img_begin: one begin frame of a video
              # img_end: one end frame of a video
              # img_goal: bool, whether to use img_begin and img_end to calculate visual difference as goal
              '''
              img_begin = cond['img_begin']
              img_end = cond['img_end']
              texts = cond['lang']
                     
              if img_goal:
                     if img_begin != None and img_end != None:      
                            visual_diff_emb = self.frame_diff_encoder.embed_frame(img_begin, img_end).to(images.device).detach()
                            loss = self.policy_loss(action_gt, images, visual_diff_emb, state, img_goal)
                            loss_dict = {"policy_loss": loss}
                            return loss_dict
                     else:
                            raise ValueError("img_begin or img_end is None")
              else:
                     text_emb = self.lang_encoder.embed_text(texts).to(images.device).detach()
                     loss = self.policy_loss(action_gt, images, text_emb, state, img_goal)
                     loss_dict = {"policy_loss": loss}
                     return loss_dict
       
       
       def policy_loss(self, x0: torch.Tensor, imgs: torch.Tensor, condition: torch.Tensor, state: torch.Tensor=None, img_goal: bool=False):
              '''
              calculate ddpm loss
              Input:
              x0: [B, D_a] ground truth action
              imgs: [B, F, V, C, H, W] batch of frames of different views
              condition: [B, D] batch of instruction embbeding
              state: [B, D_s] batch of robot arm x,y,z, gripper state
              
              return: loss
              '''
              batch_size = x0.shape[0]
              
              noise = torch.randn_like(x0, device=self.device)
              t = torch.randint(0, self.num_timesteps, (batch_size, ), device=self.device)
              xt = self.q_sample(x0, t, noise)
              
              noise_pred = self.predict_noise(xt, t, imgs, condition, state, img_goal)
              loss = (((noise_pred - noise) ** 2).sum(axis = -1)).mean()
              
              return loss
       
       
       def predict_noise(self, xt: torch.Tensor, t: torch.Tensor, imgs: torch.Tensor, condition: torch.Tensor, state: torch.Tensor=None, img_goal: bool=False):
              '''
              calculate ddpm loss
              
              Input:
              xt: [B, D_a] noise action, generated by ground truth action
              t: [B, 1] time step
              imgs: [B, F, V, C, H, W] batch of frames of different views
              condition: [B, D] batch of instruction embbeding
              state: [B, D_s] batch of robot arm x,y,z, gripper state
              
              Return: [B, D_a] predicted noise
              '''
              mod_cond = condition.clone()
              # print("mod_cond origin type:", mod_cond.dtype)
              
              # collapse
              if self.minus_mean:
                     
                     if img_goal:
                            mean_cond_emb = torch.from_numpy(self.mean_traj_emb).to(self.device)
                     else:
                            mean_cond_emb = torch.from_numpy(self.mean_lang_emb).to(self.device)
                     mod_cond -= mean_cond_emb
                     
              # cut       
              if self.text_encoder == "CUT":
                     # 创建布尔掩码
                     mask = torch.ones(mod_cond.shape[1], dtype=torch.bool).to(self.device)
                     mask[self.topk_index] = False
                     # 使用掩码来选择元素
                     mod_cond = mod_cond[:, mask]
                            

              # noise add method
              if self.add_noise and self.training:
                     
                     if self.cos_noise < 9.0:
                            # cos noise
                            rand_cos = self.cos_noise + self.cos_noise_decay*torch.rand(int(mod_cond.shape[0])).to(self.device)
                            mod_cond = generate_vector_with_given_cosine_similarity(mod_cond, rand_cos)
                     else:
                            # std noise
                            dtype = mod_cond.dtype
                            gaussian_noise = np.random.normal(loc=0.0, scale=self.noise_std*self.cos_noise_decay, size=self.noise_std.shape)
                            mod_cond = mod_cond + torch.from_numpy(gaussian_noise).to(self.device).to(dtype)
                            mod_cond = mod_cond / torch.norm(mod_cond, dim=-1, keepdim=True)   
              
              else:

                     mod_cond = mod_cond / torch.norm(mod_cond, dim=-1, keepdim=True)
                     mod_cond = mod_cond.to(torch.float)
                     
              noise_pred = self.policy(xt, t, imgs, mod_cond, state)
              return noise_pred


       def rank_transform_by_abs(self,tensor):
              b, n = tensor.shape
              ranks = torch.empty_like(tensor, dtype=torch.long).to(self.device)
              
              for i in range(b):
                     sorted_indices = torch.argsort(torch.abs(tensor[i]))
                     ranks[i, sorted_indices] = torch.arange(n, dtype=torch.long, device=self.device)
              
              ranks = (ranks - 512.0) / 512.0
              return ranks
       
       @torch.no_grad()
       def ddpm_sampler(self, shape, imgs: torch.Tensor, lang: list, state: torch.Tensor=None, guidance_strength=0, clip_sample=False, img_begin=None, img_end=None, img_goal=False):
              """
              sample x0 from xT, reverse process. Note this ddpm_sampler is different from the original ddpm_sampler as the condition type including both img & lang
              
              Input:
              shape: [num_samples, D_a], desired shapes of samples
              imgs: [B, F, V, C, H, W] batch of frames of different views
              lang: list of instructions
              state: [B, D_s] batch of robot arm x,y,z, gripper state. Do not use this when is None
              guidance_strength: strength of guidance
              clip_sample: whether to clip samples to [-1, 1]
              
              Return: [num_samples, D_a] samples
              """
              x = torch.randn(shape, device=self.device)
              if img_goal:
                     if img_begin != None and img_end != None:
                            cond = self.frame_diff_encoder.embed_frame(img_begin, img_end)
                     else:
                            raise ValueError("img_begin or img_end is None")
              else:
                     cond = self.lang_encoder.embed_text(lang) if self.lang_encoder is not None else lang        
              # cond = cond.repeat(x.shape[0], 1)
              # imgs = imgs.repeat(x.shape[0], *((1,) * (len(imgs.shape) - 1)))
              
              for t in reversed(range(self.num_timesteps)):
                     t_tensor = torch.tensor([t]).unsqueeze(0).repeat(x.shape[0], 1).to(self.device)
                     noise_pred = self.predict_noise(x, t_tensor, imgs, cond, state, img_goal)
                     x = self.p_sample(x, torch.full((shape[0], 1), t, device=self.device), noise_pred, clip_sample=clip_sample)
              return x
       

       @torch.no_grad()
       def get_action(self, imgs, lang, state=None, num=1, t=1, k=0.25, clip_sample=True, img_begin=None, img_end=None, img_goal=False):
              if not isinstance(imgs, torch.Tensor):
                # transform lists to torch.Tensor
                imgs = torch.stack([self.transform(Image.fromarray(frame).convert("RGB")) for frame in imgs]).unsqueeze(0).unsqueeze(0).to('cuda')
              else:
                imgs = imgs.to('cuda')

              B, F, V, C, H, W = imgs.shape
              try:
                   s_dim = self.policy.s_dim
              except:
                   s_dim = self.policy.module.s_dim
              state = torch.from_numpy(state.astype(np.float32)).view(-1, s_dim) if state is not None else None
              state = ((state - self.s_mean) / self.s_std).to('cuda') if state is not None else None
            
              try:
                   output_dim = self.policy.output_dim
              except:
                   output_dim = self.policy.module.output_dim
              
              # use img goal or language goal
              if img_goal:
                     if img_begin != None and img_end != None:
                            if len(img_begin.shape) == 3:
                                   img_begin_pools = img_begin.unsqueeze(0).repeat(B, 1, 1, 1)
                            elif len(img_begin.shape) == 4 and img_begin.shape[0] == B:
                                   img_begin_pools = img_begin
                            else:
                                   raise ValueError(f"Please check the shape of img_begin: {img_begin.shape}")
                            img_end_pools = img_end.unsqueeze(0).repeat(B, 1, 1, 1) 
                            action = self.ddpm_sampler((B, output_dim), imgs,lang,state, clip_sample=clip_sample,img_begin=img_begin_pools,img_end=img_end_pools, img_goal=True).detach().cpu()
                            action = action.view(B, -1, 7)
                     else:
                            raise ValueError("img_begin or img_end is None")
              else:
                     action = self.ddpm_sampler((B, output_dim), imgs, [lang] * B, state, clip_sample=clip_sample).detach().cpu()
                     action = action.view(B, -1, 7)

              
              B, N, D_a = action.shape
              a_max = self.a_max.repeat(B, N, 1)
              a_min = self.a_min.repeat(B, N, 1)
              a_mean = self.a_mean.repeat(B, N, 1)
              a_std = self.a_std.repeat(B, N, 1)
              
              action = (action + 1) * (a_max - a_min) / 2 + a_min
              action = self.get_ac_action(action.numpy(), t, k)
              # action = action * a_std + a_mean
              return action
       

class IDQL_Agent(BaseAgent):
       def __init__(
              self, 
              policy_model: torch.nn.Module,
              v_model: torch.nn.Module, 
              q_models: torch.nn.Module,
              schedule: str='vp', 
              num_timesteps: int=5,
              num_sample: int=64,
              gamma: float=0.99,
              expectile: float=0.7,
       ):       
              super().__init__(
                     policy_model,
                     v_model,
                     q_models,
                     gamma=gamma
              )
              ddpm_policy = DDPM_BC(policy_model, schedule=schedule, num_timesteps=num_timesteps)
              self.policy = ddpm_policy
              self.policy_target = copy.deepcopy(ddpm_policy)
              
              self.expectile = expectile
              self.num_sample = num_sample
              self.device = self.policy.device
              

       def q_loss(self, s, a, r, next_s, d):
              with torch.no_grad():
                     target_q = r + self.gamma * d * self.v_model(next_s)
              
              qs = self.q_models(s, a)
              
              loss = [F.mse_loss(q, target_q) for q in qs]
              loss = sum(loss) / len(loss)
              return loss
       
       
       def v_loss(self, s, a):
              with torch.no_grad():
                     target_v = torch.cat(self.q_models_target(s, a), axis=1).min(axis=1, keepdim=True)[0]
              
              v = self.v_model(s)
              loss = self.expectile_loss(target_v - v)
              return loss, v.mean()
       
       
       def policy_loss(self, x0, cond=None):
              loss = self.policy.policy_loss(x0, cond)
              return loss
       
       
       def expectile_loss(self, diff):
              weight = torch.where(diff > 0, self.expectile, (1 - self.expectile))
              return torch.mean(weight * (diff**2))
       
       
       @torch.no_grad()
       def get_action(self, state, from_target=True):
              if from_target:
                     actions = self.policy_target.get_action(state, self.num_sample, clip_sample=True)
              else:
                     actions = self.policy.get_action(state, self.num_sample, clip_sample=True)
              
              state = state.repeat(actions.shape[0], 1)
              qs = torch.cat(self.q_models_target(state, actions), axis=1).min(axis=1)[0]
              idx = torch.argmax(qs)
              
              action = actions[idx]
              return action
              
              
              

if __name__ == '__main__':
       timesteps = 100
       schedule = SCHEDULE['linear']
       print(schedule(timesteps)[0])

       