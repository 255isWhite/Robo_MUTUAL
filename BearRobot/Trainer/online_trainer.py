import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR

from torch.nn.parallel import DistributedDataParallel as NativeDDP

from tqdm import tqdm

from BearRobot.utils.evaluation.base_eval import BaseEval
from BearRobot.utils.logger.base_log import BaseLogger
from BearRobot.Agent.base_agent import BaseAgent
from BearRobot.utils.dataset.onlineRL_dataloader import OnlineReplaybuffer


OPTIMIZER = {"adam": torch.optim.Adam,
             "adamw": torch.optim.AdamW}

LR_SCHEDULE = {"cosine": torch.optim.lr_scheduler.CosineAnnealingLR}

   
class RLTrainer:
       def __init__(
              self,
              agent: BaseAgent,
              train_dataloader: OnlineReplaybuffer,
              env,
              logger: BaseLogger,
              evaluator: BaseEval,
              num_steps: int,
              lr: float=1e-4,
              policy_ema: float=5e-3,
              critic_ema: float=5e-3,
              optimizer: str='adam',
              device: str='cpu',
       ):
              # interact env
              self.env = env
              
              # model
              self.agent = agent
              
              # dataloader
              self.train_dataloader = train_dataloader
              
              # optimizer
              self.policy_optimizer = OPTIMIZER['adamw'](self.agent.policy.parameters(), lr=lr)
              self.q_optimizer = OPTIMIZER[optimizer](self.agent.q_models.parameters(), lr=lr)
              self.policy_ema = policy_ema
              self.critic_ema = critic_ema
              
              # learning rate schedule
              # self.scheduler = LR_SCHEDULE['cosine'](self.policy_optimizer, num_steps * 2)
              
              # logger
              self.logger = logger
              self.device = device
              
              # evaluator
              self.evaluator = evaluator
              
              self.num_steps = num_steps
       
       def train_steps(self):
              """
              train some steps
              """
              ep_steps = 0
              steps = self.num_steps
              self.agent.train()
              
              step_s, step_d = self.env.reset(), False
              for step in tqdm(range(steps)):                   
                     # interact with env
                     ep_steps += 1
                     if step <= self.agent.start_steps:
                            step_a = self.env.action_space.sample()
                     else:
                            step_a = self.agent.explore_action(step_s)
                     step_next_s, step_r, step_d, _ = self.env.step(step_a)
                     d_bool = 1 - float(step_d) if ep_steps < 1000 else 1.
                     
                     # add data in replay buffer
                     self.train_dataloader.add_data(step_s, step_a, step_r, step_next_s, d_bool)
                     step_s = step_next_s
                     
                     if step_d:
                            ep_steps = 0.
                            step_s, step_d = self.env.reset(), False
                     
                     # train agent after collecting some initial data
                     if step <= self.agent.start_steps:
                            continue
                     
                     # sample one batch
                     batch = self.train_dataloader.sample_batch()
                     for key in batch.keys(): 
                            batch[key] = batch[key].to(self.device)
                     s, a, r, next_s, d = batch['s'], batch['a'], batch['r'], batch['next_s'], batch['d']
                     
                     # update q
                     self.q_optimizer.zero_grad()
                     q_loss, q_mean = self.agent.q_loss(s, a, r, next_s, d)
                     q_loss.backward()
                     self.q_optimizer.step()
                     
                     # delayed update policy
                     if (step + 1) % self.agent.utd == 0: 
                            self.policy_optimizer.zero_grad()
                            p_loss = self.agent.policy_loss(s)
                            p_loss.backward()
                            self.policy_optimizer.step()
                            self.ema_update_q()
                            self.ema_update_policy()
                     
                     # log the training process
                     if (step + 1) % self.logger.record_freq == 0:
                            self.logger.log_metrics({"train/policy_loss": p_loss.item(),
                                                 "train/q_loss": q_loss.item(),
                                                 "train/q_mean": q_mean.item()}, step=step)
                     
                     # evaluate
                     if (step + 1) % self.evaluator.eval_freq == 0:
                            self.agent.eval()
                            rewards = self.evaluator.eval_episodes(self.agent, step)
                            print(f"Epoch {step} Average return: {rewards:.4f}")
                            self.agent.train()
              
              self.logger.finish()
       
       
       def ema_update_policy(self):
              for param, target_param in zip(self.agent.policy.parameters(), self.agent.policy_target.parameters()):
                     target_param.data.copy_(self.policy_ema * param.data + (1 - self.policy_ema) * target_param.data)
       
       
       def ema_update_q(self):
              for param, target_param in zip(self.agent.q_models.parameters(), self.agent.q_models_target.parameters()):
                     target_param.data.copy_(self.critic_ema * param.data + (1 - self.critic_ema) * target_param.data)