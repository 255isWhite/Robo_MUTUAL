import torch
import torch.nn as nn
import numpy as np
import copy

class BaseAgent(nn.Module):
       def __init__(
              self, 
              policy: torch.nn.Module,
              v_model: torch.nn.Module, 
              q_models: torch.nn.Module,
              gamma: float=0.99,
              utd: int=2,
              start_steps: int=int(25e3),
              ac_num: int=1,
       ):       
              super().__init__()
              self.policy = policy
              # self.policy_target = copy.deepcopy(policy)
              self.v_model = v_model
              self.q_models = q_models
              self.q_models_target = copy.deepcopy(q_models)
              
              self.start_steps = start_steps
              self.utd = utd
              self.gamma = gamma
              self.device = policy.device
              
              # action chunking
              self.use_ac = True if ac_num > 1 else False
              self.ac_num = ac_num                    
              
       def get_action(self, state):
              """
              get the action during evaluation
              """
              pass
       
       def explore_action(self, state):
              """
              get the action during online interaction, used only for online RL methods
              """
              pass
       
       def load(self, ckpt_path):
              pass
       
       def policy_loss(self):
              pass
       
       def q_loss(self):
              pass
       
       def v_loss(self):
              pass
       
       def _init_action_chunking(self, eval_horizon: int=600, num_samples: int=1):
              self.constant = 10000
              self.all_time_actions = np.ones([num_samples, eval_horizon, eval_horizon+self.ac_num, 7]) * self.constant
       
       # action_trunking
       def get_ac_action(self, actions, t: int, k: float=0.25):
              """get action trunking + temporal ensemble action

              Args:
                  actions (np.array): the predicted action sequence
                  t (int): current time step
                  k (float, optional): the temperature of temporal ensemble. Defaults to 0.25.
              """
              if self.use_ac:
                     B, N, D = actions.shape
                     # actions = actions.reshape(-1, 7)
                     self.all_time_actions[:, [t], t:t+self.ac_num] = np.expand_dims(actions, axis=1)   # B, horizon, horizon+ac_num, 7
                     actions_for_curr_step = self.all_time_actions[:, :, t]  # B, horizon, 7
                     actions_populated = np.all(actions_for_curr_step != self.constant, axis=-1)  # B, horizon
                     actions_for_curr_step = actions_for_curr_step[actions_populated].reshape(B, -1, D)  # B, N, 7
                     exp_weights = np.exp(-k * np.arange(actions_for_curr_step.shape[1]))  # N, 1
                     exp_weights = (exp_weights / exp_weights.sum()).reshape(1, -1, 1)
                     actions = (actions_for_curr_step * exp_weights).sum(axis=1)
                     return actions
              else:
                     return actions
       
       def get_statistics(self, path):
              from mmengine import fileio
              import json
              def openjson(path):
                     value  = fileio.get_text(path)
                     dict = json.loads(value)
                     return dict

              self.statistics = openjson(path)
              self.a_max = torch.tensor(self.statistics['a_max'])
              self.a_min = torch.tensor(self.statistics['a_min'])
              self.a_mean = torch.tensor(self.statistics['a_mean'])
              self.a_std = torch.tensor(self.statistics['a_std'])

              self.s_mean = torch.tensor(self.statistics['s_mean'])
              self.s_std = torch.tensor(self.statistics['s_std'])
              print('dataset statistics load succ:')
              print_dict = json.dumps(self.statistics, indent=4, sort_keys=True)
              print(print_dict)


       def get_transform(self, img_size=0, transform_list=None):
              from PIL import Image
              from torchvision import transforms
              if transform_list is None:
                     transform_list  = [
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                     ]
                     if img_size != 0:
                            transform_list.insert(0, transforms.CenterCrop(img_size))
                            transform_list.insert(0, transforms.Resize(256, interpolation=Image.BICUBIC))
              else:
                     transform_list = transform_list
              self.transform = transforms.Compose(transform_list)