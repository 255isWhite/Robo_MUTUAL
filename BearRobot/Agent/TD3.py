import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy

from BearRobot.Agent.base_agent import BaseAgent
       

class TD3_Agent(BaseAgent):
       def __init__(
              self, 
              policy_model: torch.nn.Module,
              v_model: torch.nn.Module, 
              q_models: torch.nn.Module,
              policy_noise: float=0.2,
              noise_clip: float=0.5,
              start_steps: int=int(25e3),
              utd: int=2,
              gamma: float=0.99,
       ):       
              super().__init__(
                     policy_model,
                     v_model,
                     q_models,
                     gamma=gamma,
                     utd=utd,
                     start_steps=start_steps
              )

              self.policy_noise = policy_noise
              self.noise_clip = noise_clip
              self.start_steps = start_steps
              self.device = self.policy.device
              

       def q_loss(self, s, a, r, next_s, d):
              with torch.no_grad():
                     noise = (torch.randn_like(a) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
                     next_a = (self.policy_target(next_s) + noise).clamp(-1., 1.)
                     
                     target_q = torch.cat(self.q_models_target(next_s, next_a), axis=1).min(axis=1, keepdim=True)[0]
                     target_q = r + self.gamma * d * target_q
              
              qs = self.q_models(s, a)
              q_mean = (sum(qs) / len(qs)).mean()
              
              loss = [F.mse_loss(q, target_q) for q in qs]
              loss = sum(loss) / len(loss)
              return loss, q_mean.detach().cpu().numpy()
       
       
       def v_loss(self, s, a):
              raise ValueError(f"TD3 has no value function")
       
       
       def policy_loss(self, s):
              a_pi = self.policy(s)
              q_pi = self.q_models(s, a_pi)[0]
              loss = - q_pi.mean()
              return loss
       
       
       @torch.no_grad
       def get_action(self, state, from_target=False):
              if from_target:
                     actions = self.policy_target(state)
              else:
                     actions = self.policy(state)
              return actions
       
       @torch.no_grad
       def explore_action(self, state):
              if not isinstance(state, torch.Tensor):
                     state = torch.tensor(state, dtype=torch.float32).to(self.device)
              action = self.policy(state)
              
              noise = (torch.randn_like(action) * 0.1)
              action = (action + noise).clamp(-1., 1.)
              return action.cpu().detach().numpy()
              
              

if __name__ == '__main__':
       timesteps = 100
       schedule = SCHEDULE['linear']
       print(schedule(timesteps)[0])

       