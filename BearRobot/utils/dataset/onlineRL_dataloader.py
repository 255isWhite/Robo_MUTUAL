import math
import os
import json
import random

import torch
import torch.nn as nn
import numpy as np

from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torchvision import transforms

from utils import ddp

import d4rl
import gym


EPS = 1e-5

class OnlineReplaybuffer():
       """
       Replay buffer used for online RL training
       """
       def __init__(
              self,
              env_name: str,
              max_size: int=int(1e+6),
              batch_size: int=64,
       ):
              
              self.env_name = env_name
              env = gym.make(env_name)
              self.dataset = d4rl.qlearning_dataset(env)  # dict with np.array
              self.s_dim = self.dataset['observations'].shape[1]
              self.a_dim = self.dataset['actions'].shape[1]
              
              self.max_size = max_size
              self.batch_size = batch_size
              self.ptr = 0  # pointer for buffer
              self.size = 0  # data num
              
              # data buffer
              self.s = np.zeros((max_size, self.s_dim))
              self.a = np.zeros((max_size, self.a_dim))
              self.r = np.zeros((max_size, 1))
              self.next_s = np.zeros((max_size, self.s_dim))
              self.d = np.zeros((max_size, 1))
              
              self._normalize_dataset()
       
       
       def _normalize_dataset(self, norm_s=False, norm_a=False, norm_r=False):
              s_mean = np.zeros_like(self.dataset['observations'].mean(axis=0, keepdims=True))
              s_std = np.ones_like(self.dataset['observations'].std(axis=0, keepdims=True))
                     
              a_mean = np.zeros_like(self.dataset['actions'].mean(axis=0, keepdims=True))
              a_std = np.ones_like(self.dataset['actions'].std(axis=0, keepdims=True))
              
              self.data_statistics = {
                     "dataset_num" : self.dataset['observations'].shape[0],
                     "s_mean" : s_mean,
                     "s_std" : s_std,
                     "a_mean" : a_mean,
                     "a_std" : a_std,            
              }
              
              
       def add_data(self, s, a, r, next_s, d):
              self.s[self.ptr] = s
              self.a[self.ptr] = a
              self.r[self.ptr] = r
              self.next_s[self.ptr] = next_s
              self.d[self.ptr] = d
              
              self.ptr = (self.ptr + 1) % self.max_size
              self.size = min(self.size + 1, self.max_size)


       def sample_batch(self):
              ind = np.random.randint(0, self.size, size=self.batch_size)
              
              s = torch.from_numpy(self.s[ind]).to(torch.float32)
              a = torch.from_numpy(self.a[ind]).to(torch.float32)
              next_s = torch.from_numpy(self.next_s[ind]).to(torch.float32)
              r = torch.from_numpy(self.r[ind]).to(torch.float32)
              d = torch.from_numpy(self.d[ind]).to(torch.float32)
              
              return {"s": s, "a": a, "r": r, "next_s": next_s, "d": d}