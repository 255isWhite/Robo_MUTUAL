import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import numpy as np

import d4rl
import gym


EPS = 1e-5

def return_range(dataset, max_episode_steps):
       returns, lengths = [], []
       ep_ret, ep_len = 0., 0
       for r, d in zip(dataset['rewards'], dataset['terminals']):
              ep_ret += float(r)
              ep_len += 1
              if d or ep_len == max_episode_steps:
                     returns.append(ep_ret)
                     lengths.append(ep_len)
                     ep_ret, ep_len = 0., 0
       # returns.append(ep_ret)    # incomplete trajectory
       lengths.append(ep_len)      # but still keep track of number of steps
       assert sum(lengths) == len(dataset['rewards'])
       return min(returns), max(returns)


class D4RLDataset(Dataset):
       def __init__(
              self,
              env_name: str,
              norm_dict: dict={"norm_s": False,
                               "norm_a": False,
                               "norm_r": True},
              clip_to_eps: bool=True, 
       ):
              super().__init__()
              
              self.env_name = env_name
              env = gym.make(env_name)
              self.dataset = d4rl.qlearning_dataset(env)  # dict with np.array
              self.data_statistics = {"dataset_num" : self.dataset['observations'].shape[0]}
              
              if clip_to_eps:
                     lim = 1 - EPS
                     self.dataset["actions"] = np.clip(self.dataset["actions"], -lim, lim)
              
              dones = np.full_like(self.dataset["rewards"], False, dtype=bool)
              for i in range(len(dones) - 1):
                     if (
                            np.linalg.norm(
                            self.dataset["observations"][i + 1]
                            - self.dataset["next_observations"][i]
                            )
                            > 1e-6
                            or self.dataset["terminals"][i] == 1.0
                     ):
                            dones[i] = True

              dones[-1] = True
              self.dataset["dones"] = dones
              
              self._normalize_dataset(**norm_dict)
              self._tran2tensor()
              
              self.s_dim = self.data_statistics['s_mean'].shape[1]
              self.a_dim = self.data_statistics['a_mean'].shape[1]
       
             
       def _normalize_dataset(self, norm_s=False, norm_a=False, norm_r=False):
              if norm_s:
                     s_mean = self.dataset['observations'].mean(axis=0, keepdims=True)
                     s_std = self.dataset['observations'].std(axis=0, keepdims=True)
              else:
                     s_mean = np.zeros_like(self.dataset['observations'].mean(axis=0, keepdims=True))
                     s_std = np.ones_like(self.dataset['observations'].std(axis=0, keepdims=True))
                     
              if norm_a:
                     a_mean = self.dataset['actions'].mean(axis=0, keepdims=True)
                     a_std = self.dataset['actions'].std(axis=0, keepdims=True)
              else:
                     a_mean = np.zeros_like(self.dataset['actions'].mean(axis=0, keepdims=True))
                     a_std = np.ones_like(self.dataset['actions'].std(axis=0, keepdims=True))
              
              if any(s in self.env_name for s in ('halfcheetah', 'hopper', 'walker2d')):
                     if norm_r:
                            (_, _, episode_returns) = self._trajectory_boundaries_and_returns()
                            self.dataset["rewards"] /= (np.max(episode_returns) - np.min(episode_returns))
                            self.dataset['rewards'] *= 1000
              elif 'antmaze' in self.env_name:
                     self.dataset['rewards'] -= 1.
              else:
                     raise NotImplementedError
                     
              self.dataset['observations'] = (self.dataset['observations'] - s_mean) / (s_std + EPS)
              self.dataset['next_observations'] = (self.dataset['next_observations'] - s_mean) / (s_std + EPS)
              self.dataset['actions'] = (self.dataset['actions'] - a_mean) / (a_std + EPS)
              
              self.data_statistics = {
                     "dataset_num" : self.dataset['observations'].shape[0],
                     "s_mean" : s_mean,
                     "s_std" : s_std,
                     "a_mean" : a_mean,
                     "a_std" : a_std,            
              }

       def _trajectory_boundaries_and_returns(self):
              episode_starts = [0]
              episode_ends = []

              episode_return = 0
              episode_returns = []

              for i in range(len(self)):
                     episode_return += self.dataset["rewards"][i]

                     if self.dataset["dones"][i]:
                            episode_returns.append(episode_return)
                            episode_ends.append(i + 1)
                            if i + 1 < len(self):
                                   episode_starts.append(i + 1)
                                   episode_return = 0.0

              return episode_starts, episode_ends, episode_returns
                     
       def _tran2tensor(self):
              for key in ['observations', 'actions', 'rewards', 'next_observations', 'terminals']:
                     self.dataset[key] = torch.from_numpy(self.dataset[key]).to(torch.float32).view(self.data_statistics["dataset_num"], -1)
       
       def __len__(self):
              return self.data_statistics["dataset_num"]

       def __getitem__(self, idx):
              s = self.dataset['observations'][idx]
              a = self.dataset['actions'][idx]
              r = self.dataset['rewards'][idx]
              next_s = self.dataset['next_observations'][idx]
              d = 1. - self.dataset['terminals'][idx]
              return {"s": s, "a": a, "r": r, "next_s": next_s, "d": d}