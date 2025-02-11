import os

import torch
import torchvision.transforms as transforms
import numpy as np
import random

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv, SubprocVectorEnv

from BearRobot.utils.evaluation.base_eval import BaseEval
from BearRobot.Agent.base_agent import BaseAgent
from BearRobot.utils.logger.base_log import BaseLogger
from BearRobot.utils.dataset.dataloader import openimage
from data.libero.data_process import demo2frames
from data.libero.data_process import get_libero_frame

from tqdm import tqdm
import json

import multiprocessing
from functools import partial
import imageio

import re

EPS = 1e-5
LIBERO_DATASETS = {'libero_goal': ["libero_goal"],
                   "libero_object": ["libero_object"],
                   "libero_spatial": ["libero_spatial"],
                   "libero_10": ["libero_10"],
                   "libero_90": ["libero_90"],
                   "libero30": ["libero_goal", "libero_object", "libero_spatial"],
                   "libero130": ["libero_goal", "libero_object", "libero_spatial", "libero_10", "libero_90"]}


benchmark_dict = benchmark.get_benchmark_dict()

def has_normalize(transform):
       if isinstance(transform, transforms.Compose):
              for t in transform.transforms:
                     if isinstance(t, transforms.Normalize):
                            return True
       return False


class LIBEROEval(BaseEval):
       def __init__(
              self,
              task_suite_name: str, # can choose libero_spatial, libero_goal, libero_object, libero_10, libero30, libero_90, libero130.
              obs_key: list=['agentview_image', 'robot0_eye_in_hand_image', 'robot0_gripper_qpos', 'robot0_eef_pos', 'robot0_eef_quat'],
              data_statistics: dict=None,
              logger: BaseLogger=None,
              eval_horizon: int=600,
              num_episodes: int=10,
              eval_freq: int=10,
              seed: int=42,
              rank: int=0,
              json_path: str='/home/dodo/ljx/BearRobot/data/libero/libero_goal-ac.json',
       ):
              super(BaseEval, self).__init__()   
              self.task_suite_name = task_suite_name
              
              assert self.task_suite_name in LIBERO_DATASETS
              self.task_list = LIBERO_DATASETS[self.task_suite_name]
              self.task_suite_list = [benchmark_dict[task]() for task in self.task_list]
              self.obs_key = obs_key
              self.data_statistics = data_statistics
              
              self.frame_length_dict = demo2frames.frame_counts_dict(json_path=json_path)
              
              self.eval_horizon = eval_horizon
              self.num_episodes = num_episodes
              self.eval_freq = eval_freq
              self.logger = logger
              self.seed = seed
              
              self.rank = rank
                     
       def _make_dir(self, save_path):
              if self.rank == 0:
                     task_suite_name = self.task_suite_name
                     path = os.path.join(save_path, task_suite_name)
                     if not os.path.exists(path):
                            os.makedirs(path)
                     self.base_dir = path
       
       def _init_env(self, task_suite, task_id: int=0):
              # get task information and env args
              task = task_suite.get_task(task_id)
              task_name = task.name
              task_description = task.language
              task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
              print(f"[info] retrieving task {task_id} from suite {self.task_suite_name}, the " + \
                     f"language instruction is {task_description}, and the bddl file is {task_bddl_file}")

              # step over the environment
              env_args = {
                     "bddl_file_name": task_bddl_file,
                     "camera_heights": 128,
                     "camera_widths": 128
              }
              
              # init thesubprocess vector environment
              env_num = self.num_episodes
              env = SubprocVectorEnv(
                     [lambda: OffScreenRenderEnv(**env_args) for _ in range(env_num)]
              )
              
              # environment reset 
              env.seed(self.seed + 100)
              env.reset()
              init_states = task_suite.get_task_init_states(task_id) # for benchmarking purpose, we fix the a set of initial states
              init_state_id = np.arange(self.num_episodes) % init_states.shape[0]
              obs = env.set_init_state(init_states[init_state_id])
              
              ### sample one begin and end image frame to construct image goal
              # Generating paths for img_begin and img_end
              task_name = task_description.replace(" ", "_") + "_demo"
              demo_paths = demo2frames.get_demos_for_task(task_name, self.frame_length_dict)
              demo_path = random.choice(demo_paths)

              # eval with the beginning frame and the endding frame
              env_dict = {}
              transform = transforms.ToTensor()
              base_dir='/data/libero/data_jpg/'
              env_dict['img_begin'] = transform(openimage(os.path.join(base_dir, demo_path, "image0/0.jpg")))
              end_idx = self.frame_length_dict[demo_path] - 1 
              env_dict['img_end'] = transform(openimage(os.path.join(base_dir, demo_path, f"image0/{end_idx}.jpg")))

              # eval with the random frames
              # env_dict = {}
              # frame_length = frame_length_dict[demo_path]
              # frame_gap = 10
              # begin_idx = 0
              # begin_idx = torch.randint(0, frame_length - frame_gap, (1,)).item()
              # base_dir='/home/dodo/ljx/BearRobot/data/libero/dataset/'
              # transform = transforms.ToTensor()
              # env_dict['img_begin'] = transform(openimage(os.path.join(base_dir,"libero/data_jpg/libero_goal/", demo_path, f"image0/{begin_idx}.jpg")))
              # end_idx = torch.randint(begin_idx, frame_length, (1,)).item()
              # env_dict['img_end'] = transform(openimage(os.path.join(base_dir,"libero/data_jpg/libero_goal/", demo_path, f"image0/{end_idx}.jpg")))

              # return the environment
              env_dict['env'] = env
              print("[INFO] Origin task description: ", task_description)
              # task_description = self.get_modified_text(task_description)
              print("[INFO] Modified task description: ", task_description)
              env_dict['language_instruction'] = task_description
              env_dict['obs'] = obs
              
              return env_dict
       
       def _log_results(self, metrics: dict, steps: int):
              if self.logger is None:
                     # just print out and save the results and pass
                     print(metrics)
                     save_name = os.path.join(self.base_dir, 'results.json')
                     with open(save_name, 'a+') as f:
                            line = json.dumps(metrics)
                            f.write(line+'\n')
              else:
                     # log the results to the logger
                     self.logger.log_metrics(metrics, steps)
                     self.logger.save_metrics(metrics, steps, self.base_dir)
       
       def raw_obs_to_stacked_obs(self, obs, lang):
              env_num = len(obs)
              
              data = {
                     "obs": {},
                     "lang": lang,
              }
              
              for key in self.obs_key:
                     data["obs"][key] = []
                     
              for i in range(env_num):
                     for key in self.obs_key:
                            data['obs'][key].append(obs[i][key])
              
              for key in data['obs']:
                     data['obs'][key] = np.stack(data['obs'][key])
              
              return data     
       
       def np_image_to_tensor(self, image: np.array, normalize_img: bool):
              B, H, W, C = image.shape
              image = image / 255. if image.max() > 10 else image
              image = torch.from_numpy(image).permute(0, 3, 1, 2).to(torch.float32)  # B, C, H, W
              
              if normalize_img:
                     norm_mean = torch.tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1).repeat(B, 1, 1, 1)  # B, C, 1, 1
                     norm_std = torch.tensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1).repeat(B, 1, 1, 1)
                     
                     image = (image - norm_mean) / norm_std
              return image  # B, C, H, W

       def _rollout(self, task_suite, policy: BaseAgent, task_id: int=0, img_goal=False):
              """
              rollout one episode and return the episode return
              """
              env = self._init_env(task_suite, task_id)
              lang = env['language_instruction']
              obs = env['obs']
              img_begin = env['img_begin']
              img_end = env['img_end']
              policy._init_action_chunking(self.eval_horizon, self.num_episodes)
              
              images = []
              for t in tqdm(range(self.eval_horizon), desc=f'{lang}'):
                     # data transform
                     ## image
                     data = self.raw_obs_to_stacked_obs(obs, lang)
                     obs, lang = data['obs'], data['lang']
                     
                     normalize_img = has_normalize(policy.transform)
                     agent_view = self.np_image_to_tensor(obs['agentview_image'], normalize_img).unsqueeze(1)
                     wrist_view = self.np_image_to_tensor(obs['robot0_eye_in_hand_image'], normalize_img).unsqueeze(1)
                     image_input = torch.cat([agent_view, wrist_view], dim=1).unsqueeze(1)
                     
                     ### record the video
                     B, H, W, C = obs["agentview_image"].shape
                     images.append(np.flip(np.flip(obs["agentview_image"], 1), 2).reshape(B * H, W, C))
                     
                     ## proprio
                     gripper_qpos = obs['robot0_gripper_qpos']
                     eef_pos = obs['robot0_eef_pos']
                     eef_quat = obs['robot0_eef_quat']
                     
                     try:
                            s_dim = policy.policy.s_dim
                     except:
                            s_dim = policy.policy.module.s_dim
                     state = np.concatenate([gripper_qpos, eef_pos, eef_quat], axis=-1) if s_dim > 0 else None

                     # get action using img_begin and img_end embedding difference
                     if img_goal:
                            # using the current frame as the img_begin   # TODO delete this if not useful
                            # img_begin = torch.from_numpy(obs['agentview_image']).permute(0, 3, 1, 2) / 255
                            
                            action = policy.get_action(image_input, None, state=state, t=t, k=0.25, img_begin=img_begin, img_end = img_end, img_goal=img_goal)
                     else:
                            action = policy.get_action(image_input, lang, state=state, t=t, k=0.25)
                     # reshape
                     action = action.reshape(self.num_episodes, -1)
                     
                     # step
                     obs, reward, done, info = env['env'].step(action)
                     if done.all():
                            break
              save_path = f'{self.base_dir}/{lang}.mp4'
              self._save_video(save_path, images, done, fps=30)
              
              num_success = 0
              for k in range(self.num_episodes):
                     num_success += int(done[k])
              avg_succ_rate = num_success / self.num_episodes
             
              metrics = {f'sim/{self.task_suite_name}/{lang}': avg_succ_rate}
              self._log_results(metrics, self.step)
              
              env['env'].close()
              return avg_succ_rate
       
       def _save_video(self, save_path: str, images: list, done: list, fps=30): 
              imageio.mimsave(save_path, images, fps=fps)
              
       
       def eval_episodes(self, policy: BaseAgent, steps: int, save_path: str, img_goal=False):
              """
              rollout several episodes and log the mean episode return
              """
              self._make_dir(save_path)
              self.step = steps
              
              rews = []
              policy.eval()
              for task_suite in self.task_suite_list:
                     for task_id in tqdm(range(len(task_suite.tasks)), desc="Evaluating..."):
                            rews.append(self._rollout(task_suite, policy, task_id, img_goal))
              eval_rewards = sum(rews) / len(rews)
              metrics = {f'sim_summary/{self.task_suite_name}/all': eval_rewards}
              self._log_results(metrics, self.step)
              return eval_rewards
              
       
       def close_env(self):
              for env in self.env:
                     env['env'].close()
                     
       def modify_text(self, text):
              # 定义可能的语言风格变化
              replacements = [
                     (r'\bthe\b', lambda match: random.choice(['that', 'this', 'a', 'the'])),
                     (r'\bon\b', lambda match: random.choice(['on top of', 'onto', 'over'])),
                     (r'\binto\b', lambda match: random.choice(['inside', 'within'])),
                     (r'\bin\b', lambda match: random.choice(['inside of', 'within', 'in'])),
                     (r'\band\b', lambda match: random.choice(['and then', 'and afterwards']))
              ]
              
              # replacements = [
              # (r'\bthe\b', lambda match: random.choice(['this', 'that', 'those', 'these', 'any', 'a'])),
              # (r'\bon\b', lambda match: random.choice(['upon', 'over', 'across', 'atop'])),
              # (r'\bin\b', lambda match: random.choice(['within', 'inside of', 'inside', 'enclosed by', 'amid'])),
              # (r'\band\b', lambda match: random.choice(['plus', 'along with', 'in addition to', 'also', 'as well as', 'furthermore'])),
              # (r'\bto\b', lambda match: random.choice(['towards', 'into', 'in the direction of', 'for', 'until'])),
              # (r'\bof\b', lambda match: random.choice(['from', 'out of', 'belonging to', 'regarding', 'concerning']))
              # ]

              
              modified_text = text
              for pattern, replacement_func in replacements:
                     modified_text = re.sub(pattern, replacement_func, modified_text)

              return modified_text
       
       # 返回修改后的文本的函数
       def get_modified_text(self, input_text):
              # 修改后的文本映射字典
              modified_texts = {
              'put the bowl on the stove': 'place the bowl onto the stove',
              'put the bowl on the plate': 'place the bowl onto the plate',
              'put the bowl on top of the cabinet': 'place the bowl on the cabinet top',
              'put the cream cheese in the bowl': 'place the cream cheese into the bowl',
              'push the plate to the front of the stove': 'move the plate to the front of the stove',
              'open the top drawer and put the bowl inside': 'open the upper drawer and place the bowl within',
              'open the middle drawer of the cabinet': 'open the central drawer of the cabinet',
              'put the wine bottle on the rack': 'place the wine bottle on the rack',
              'put the wine bottle on top of the cabinet': 'place the wine bottle atop the cabinet',
              'turn on the stove': 'activate the stove'
              }
              return modified_texts.get(input_text, "Text not found in the predefined list")
