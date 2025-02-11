import copy
import io
from mmengine import fileio
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR

from torch.nn.parallel import DistributedDataParallel as DDP

from tqdm import tqdm

from BearRobot.utils.evaluation.base_eval import BaseEval
from BearRobot.utils.logger.base_log import BaseLogger
from BearRobot.Agent.base_agent import BaseAgent
from torch.utils.data import DataLoader
from BearRobot.Trainer.lr_schedule import CosineAnnealingWarmUpRestarts


OPTIMIZER = {"adam": torch.optim.Adam,
             "adamw": torch.optim.AdamW}

LR_SCHEDULE = {"cosine": torch.optim.lr_scheduler.CosineAnnealingLR,
               "cosinewarm": CosineAnnealingWarmUpRestarts}


class BCTrainer:
       def __init__(
              self,
              agent: BaseAgent,
              train_dataloader: DataLoader,
              val_dataloader: DataLoader,
              logger: BaseLogger,
              evaluator: BaseEval,
              num_steps: int,
              lr: float=1e-4,
              ema: float=1e-3,
              optimizer: str='adam',
              device: int=0,
              global_rank: int=0,
              save: bool=True,
              save_freq: int=10000,
              save_path: str=None,
              resume_path: str=None,
              img_goal: bool=False,
              args=None,
              **kwargs,
       ):     
              self.args = args
              if args.ddp:
                     torch.distributed.barrier()
              
              # use image goal or not
              self.img_goal = img_goal
              
              # optimizer
              self.optimizer = OPTIMIZER[optimizer](agent.policy.parameters(), lr=lr)
              self.ema = ema
                          
              # ddp & model
              if args.ddp:
                     agent.policy = DDP(agent.policy, device_ids=[device], find_unused_parameters=False)
              self.agent = agent
              
              # dataloader
              self.train_dataloader = train_dataloader
              self.val_dataloader = val_dataloader
              
              # learning rate schedule
              self.scheduler = LR_SCHEDULE['cosinewarm'](self.optimizer, num_steps, eta_min=1e-6)
              
              # logger
              self.logger = logger
              self.device = device
              self.global_rank = global_rank
              
              # evaluator
              self.evaluator = evaluator
              
              # save
              self.save = save
              self.save_freq = save_freq
              self.save_path = save_path
              
              # resume 
              self.init_step = 0
              try:
                     self.load_model(resume_path)
              except:
                     print("train from scratch")
              
              self.num_steps = num_steps
              if args.ddp:
                     torch.distributed.barrier()

              n_parameters = sum(p.numel() for p in agent.parameters() if p.requires_grad)
              print("number of trainable parameters: %.2fM" % (n_parameters/1e6,))
              
       def train_epoch(self):
              """
              train some epochs
              """
              epochs = self.num_steps
              self.agent.train()
              for epoch in range(0, epochs):
                     epoch_loss = 0.
                     self.train_dataloader.sampler.set_epoch(epoch)
                     with tqdm(self.train_dataloader, unit="batch") as pbar:
                            for batch in pbar:
                                   imgs = batch['imgs'].to(self.device)
                                   a = batch['a'].to(self.device)
                                   lang = batch['lang']
                                   try:
                                          img_begin = batch["img_begin"]
                                          img_end =  batch["img_end"]
                                   except:
                                          img_begin = None
                                          img_end = None
                                   
                                   cond = {"lang": lang,
                                           "img_begin": img_begin,
                                           "img_end": img_end
                                          }
                                   
                                   self.optimizer.zero_grad()
                                   loss = self.agent(imgs, cond, a, img_goal=self.img_goal)
                                   loss.backward()
                                   self.optimizer.step()
                                   if self.args.ddp:
                                          torch.cuda.synchronize()
                                   
                                   pbar.set_description(f"Epoch {epoch} Loss: {loss.item():.4f}")
                                   epoch_loss += loss.item()
                                   if self.ema is not None:
                                          self.ema_update()
                            
                            self.scheduler.step()
                            
                     avg_loss = epoch_loss / len(self.train_dataloader)
                     if self.global_rank == 0:
                            self.logger.log_metrics({"train/loss": avg_loss}, step=epoch)
                            print(f"Epoch {epoch} Average Loss: {avg_loss:.4f}")
                     
                     # if (epoch + 1) % self.evaluator.eval_freq == 0:
                     #        rewards = self.evaluator.eval_episodes(self.target_diffusion_agent, epoch+1)
                     #        print(f"Epoch {epoch} Average return: {rewards:.4f}")
              
              self.logger.finish()
                     
       def train_steps(self):
              """
              train some steps
              """
              steps = self.num_steps
              self.agent.train()
              
              epoch = 0
              self.train_dataloader.sampler.set_epoch(epoch)
              iterator = iter(self.train_dataloader)
              with tqdm(range(self.init_step, steps)) as pbar:
                     for step in pbar:
                            # get one batch sample
                            t0 = time.time()
                            try:
                                   batch = next(iterator)
                            except:
                                   epoch += 1
                                   self.train_dataloader.sampler.set_epoch(epoch)
                                   iterator = iter(self.train_dataloader)
                                   batch = next(iterator)
                            
                            t1 = time.time()
                            imgs = batch['imgs'].to(self.device)
                            label = batch['label'].to(self.device)
                            proprio = batch['proprio'].to(self.device)
                            lang = batch['lang']
                            try:
                                   img_begin = batch["img_begin"]
                                   img_end =  batch["img_end"]
                            except:
                                   img_begin = None
                                   img_end = None
                            
                            cond = {"lang": lang,
                                   "img_begin": img_begin,
                                   "img_end": img_end
                                   }

                            # one gradient step
                            self.optimizer.zero_grad()
                            loss = self.agent(imgs, cond, label, proprio, img_goal=self.img_goal)
                            loss['policy_loss'].backward()
                            self.optimizer.step()
                            t2 = time.time()
                            if self.args.ddp:
                                   torch.cuda.synchronize()
                            
                            # if self.ema is not None:
                            #        self.ema_update()
                            self.scheduler.step()
                            
                            # log
                            if self.global_rank == 0:
                                   pbar.set_description(f"Step {step} train Loss: {loss['policy_loss'].item():.4f}")
                                   loss_log = {f"train/{key}": value.item() for key, value in loss.items()}
                                   self.logger.log_metrics(loss_log, step=step)
                                   self.logger.log_metrics({"train/lr": self.scheduler.get_last_lr()[0],
                                                 "time/sample": t1-t0,
                                                 "time/train": t2-t1,}, step=step)
                            
                            # save
                            if self.save:
                                   if (step + 1) % self.save_freq == 0 or (step+1) == steps:
                                          if self.global_rank == 0:
                                                 self.save_model(step, loss_log['train/policy_loss'])
                            
                            # validation
                            if (step + 1) % self.args.val_freq == 0:
                                   self.validation(step)
                                   
                            # evaluation
                            if (step + 1) % self.args.eval_freq == 0:
                                   if self.args.ddp:
                                          evaluate = True if self.global_rank == 0 else False
                                   else:
                                          evaluate = True
                                   
                                   if evaluate:
                                          self.evaluator.eval_episodes(self.agent, step, self.save_path, img_goal=True) 
                                          self.evaluator.eval_episodes(self.agent, step, self.save_path, img_goal=False) 
                            
                            if self.args.ddp:
                                   torch.distributed.barrier()

              self.logger.finish()
       
       
       def validation(self, step):
              t1 = time.time()
              val_loss = 0
              self.agent.eval()
              with torch.no_grad():
                     with tqdm(self.val_dataloader, unit="batch") as pbar:
                            for batch in pbar:
                                   imgs = batch['imgs'].to(self.device)
                                   a = batch['label'].to(self.device)
                                   lang = batch['lang']
                                   proprio = batch['proprio'].to(self.device)
                                   loss = self.agent(imgs, lang, a, proprio)
                                   
                                   if self.global_rank == 0:
                                          pbar.set_description(f"Step {step} val Loss: {loss['policy_loss'].item():.4f}")
                                   val_loss += loss['policy_loss'].item()
              
              avg_loss = val_loss / len(self.val_dataloader)
              
              t2 = time.time()
              if self.global_rank == 0:
                     self.logger.log_metrics({"val/policy_loss": avg_loss, "time/val": t2-t1}, step=step)      
              self.agent.train()
              if self.args.ddp:
                     torch.distributed.barrier()
       

       def ema_update(self):
              if self.args.ddp:
                     for param, target_param in zip(self.agent.module.policy.parameters(), self.agent.module.policy_target.parameters()):
                            target_param.data.copy_(self.ema * param.data + (1 - self.ema) * target_param.data)
              else:
                     for param, target_param in zip(self.agent.policy.parameters(), self.agent.policy_target.parameters()):
                            target_param.data.copy_(self.ema * param.data + (1 - self.ema) * target_param.data)                     
              
       def save_model(self, step, loss, save_optimizer=False, save_schedule=False):
              """
              save the model to path
              """
              save_model = {'model': self.agent.state_dict(), 
                            # 'optimizer': self.optimizer.state_dict() if save_optimizer else None, 
                            # 'schedule': self.scheduler.state_dict() if save_schedule else None,
                            'step': step}

              with io.BytesIO() as f:
                     torch.save(save_model, f)
                     fileio.put(f.getvalue(), f"{self.save_path}/{step}_{loss}.pth")
                     fileio.put(f.getvalue(), f"{self.save_path}/latest.pth")
              
       def load_model(self, path: str):
              """
              load ckpt from path
              """
              print(f"loading ckpt from {path}")

              ckpt = fileio.get(path)
              with io.BytesIO(ckpt) as f:
                     ckpt = torch.load(f, map_location='cpu')
              
              # load model
              try:
                     self.agent.load_state_dict(ckpt['model'])
                     self.agent = self.agent.to(self.device)
              except:
                     self.agent.load_state_dict(ckpt)
                     self.agent = self.agent.to(self.device)                     
              print("Model load done")
              
              # load optimizer
              try:
                     self.optimizer.load_state_dict(ckpt['optimizer'])
                     print("Optimizer load done")
              except:
                     print("no pretrained optimizer found, init one")
                     
              # load schedule
              try:
                     self.scheduler.load_state_dict(ckpt['schedule'])
                     print("Schedule load done")
              except:
                     print("no schedule found, init one")
              
              # load step
              try:
                     self.init_step = ckpt['step']
                     print("Step load done")
              except:
                     self.init_step = 0
              
              
              
class RLTrainer:
       def __init__(
              self,
              agent: BaseAgent,
              train_dataloader: DataLoader,
              val_dataloader: DataLoader,
              logger: BaseLogger,
              evaluator: BaseEval,
              num_steps: int,
              lr: float=1e-4,
              policy_ema: float=1e-3,
              critic_ema: float=5e-3,
              optimizer: str='adam',
              device: str='cpu',
       ):
              # model
              self.agent = agent
              
              # dataloader
              self.train_dataloader = train_dataloader
              self.val_dataloader = val_dataloader
              
              # optimizer
              self.policy_optimizer = OPTIMIZER['adamw'](self.agent.policy.parameters(), lr=lr)
              self.v_optimizer = OPTIMIZER[optimizer](self.agent.v_model.parameters(), lr=lr)
              self.q_optimizer = OPTIMIZER[optimizer](self.agent.q_models.parameters(), lr=lr)
              self.policy_ema = policy_ema
              self.critic_ema = critic_ema
              
              # learning rate schedule
              self.scheduler = LR_SCHEDULE['cosine'](self.policy_optimizer, num_steps * 2)
              
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
              steps = self.num_steps
              # self.agent.eval()
              # self.evaluator.eval_episodes(self.agent, 0)
              self.agent.train()
              
              iterator = iter(self.train_dataloader)
              for step in tqdm(range(steps)):
                     try:
                            batch = next(iterator)
                     except:
                            iterator = iter(self.train_dataloader)
                            batch = next(iterator)
                     for key in batch.keys(): 
                            batch[key] = batch[key].to(self.device)
                     
                     s, a, r, next_s, d = batch['s'], batch['a'], batch['r'], batch['next_s'], batch['d']
                     
                     # update policy
                     self.policy_optimizer.zero_grad()
                     p_loss = self.agent.policy_loss(a, s)
                     p_loss.backward()
                     self.policy_optimizer.step()
                     self.scheduler.step()
                     self.ema_update_policy()
                     
                     # update v
                     self.v_optimizer.zero_grad()
                     v_loss, v_mean = self.agent.v_loss(s[:256], a[:256])
                     v_loss.backward()
                     self.v_optimizer.step()
                     
                     # update q
                     self.q_optimizer.zero_grad()
                     q_loss = self.agent.q_loss(s[:256], a[:256], r[:256], next_s[:256], d[:256])
                     q_loss.backward()
                     self.q_optimizer.step()
                     self.ema_update_q()
                     
                     # log the training process
                     if (step + 1) % self.logger.record_freq == 0:
                            self.logger.log_metrics({"train/policy_loss": p_loss.item(),
                                                 "train/v_loss": v_loss.item(),
                                                 "train/q_loss": q_loss.item(),
                                                 "train/v_mean": v_mean.item(),
                                                 "train/lr": self.scheduler.get_last_lr()[0]}, step=step)
                     
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