import os
from datetime import datetime

import random
import torch
import numpy as np

from BearRobot.Net.my_model.diffusion_model import VisualDiffusion
from BearRobot.Agent.ddpm_bc import VLDDPM_BC

from BearRobot.utils.dataset.dataloader_libero import RT1DataLoader, RT1ValDataLoader, AIRKitchenDataLoader, AIRKitchenValDataLoader
from BearRobot.utils.logger.wandb_log import WandbLogger as Logger
from BearRobot.utils.net.initialization import boolean
from BearRobot.utils import ddp
from BearRobot.utils.evaluation.mp_libero_eval import LIBEROEval
from BearRobot.Trainer.trainer import BCTrainer
from BearRobot.config.basic_args import basic_args, diffusion_args
from torch.nn.parallel import DistributedDataParallel as DDP

import torch.multiprocessing as mp
import argparse
import json

LIBERO_DATASETS = {'libero_goal': ["libero_goal"],
                   "libero_object": ["libero_object"],
                   "libero_spatial": ["libero_spatial"],
                   "libero_10": ["libero_10"],
                   "libero_90": ["libero_90"],
                   "libero30": ["libero_goal", "libero_object", "libero_spatial"],
                   "libero130": ["libero_goal", "libero_object", "libero_spatial", "libero_10", "libero_90"]}

def get_args():
       parser = basic_args()
       # customize your argparser
       parser.add_argument('--img_size', default=128, type=int, help='image size')
       parser.add_argument('--frames', default=1, type=int, help='frames num input to the visual encoder')
       parser.add_argument('--visual_encoder', default='resnet34', type=str, help='visual encoder backbone, support resnet 18/34/50')
       parser.add_argument('--visual_pretrain', default=False, type=boolean, help='whether use visual pretrain')
       parser.add_argument('--ft_vision', default=False, type=boolean, help='whether tune the visual encoder')
       parser.add_argument('--text_encoder', default='DecisionNCE-T', type=str, help='language encoder, support T5, DecisionNCE-T, DecisionNCE-P, DecisionNCE-V')       
       parser.add_argument('--ac_num', default=4, type=int, help='action trunking number')
       parser.add_argument('--norm', default="minmax", type=str, help='whether norm the action or not')
       parser.add_argument('--discretize_actions', default=False, type=boolean, help='whether discretize_actions the action or not')
       parser.add_argument('--s_dim', default=9, type=int, help='qpos dim, 0 means dont use qpos')
       parser.add_argument('--encode_s', default=False, type=boolean, help='whether encode the state (qpos) or not')
       parser.add_argument('--encode_a', default=False, type=boolean, help='whether encode the action or not')
       
       parser.add_argument('--num_blocks', default=3, type=int, help='num blocks for decoder MLP')
       parser.add_argument('--hidden_dim', default=256, type=int, help='hidden dim for decoder MLP')
       parser.add_argument('--norm_type', default="bn", type=str, help='normalization type')
       parser.add_argument('--pooling_type', default="avg", type=str, help='pooling type')
       parser.add_argument('--add_spatial_coordinates', default=False, type=boolean, help='add spatial coordinates to the image')
       parser.add_argument('--film_fusion', default=False, type=boolean, help='add film condition to the decoder')
       
       # cross-modal part
       parser.add_argument('--add_noise', default=False, type=boolean, help='add gaussian noise in uni-modal training')
       parser.add_argument('--noise_data_path', default=None, type=str, help='noise data path')
       parser.add_argument('--minus_mean', default=False, type=boolean, help='minus the mean of embeddings')
       parser.add_argument('--mean_data_path', default=None, type=str, help='mean data path')
       parser.add_argument('--lang_prop', default="", type=str, help='language proportion')
       parser.add_argument('--json_path', default="", type=str, help='json file path')
       parser.add_argument('--cos_noise', default=1.0, type=float, help='cosine noise')
       parser.add_argument('--cos_noise_decay', default=0.0, type=float, help='cosine noise decay')
       parser.add_argument('--manual_img_goal', default=False, type=boolean, help='image goal')
       parser.add_argument('--cut_path', default=None, type=str, help='cut path')
       
       parser = diffusion_args(parser)
       args = parser.parse_args()    
       return args   


def main(args):
       kwargs = vars(args)
       assert kwargs['dataset_name'] in LIBERO_DATASETS
       # seed
       seed = args.seed + ddp.get_rank()
       np.random.seed(seed)
       torch.manual_seed(seed)
       torch.cuda.manual_seed_all(seed)
       random.seed(seed)
       torch.backends.cudnn.deterministic = True
       
       # init ddp
       global_rank, rank, _ = ddp.ddp_setup_universal(True, args)
       kwargs['device'] = rank       
       
       # dataset and dataloader
       view_list = ['D435_image', 'wrist_image']

       # whether use image goal
       img_goal = kwargs['manual_img_goal']
       print("Training with image goal: ", img_goal)

       json_path = kwargs['json_path']
              
       print("Json path: ", json_path)

       rt1dataloader, statistics = AIRKitchenDataLoader(
              base_dir='data',
              datalist=[json_path],
              view_list=view_list,
              img_goal=img_goal,
              **kwargs
       )

       # save 
       if args.save and global_rank==0:
              # your ckpt save path
              save_path = args.save_path
              if not os.path.exists(save_path):
                     os.makedirs(save_path)
              
              # save the statistics for training dataset
              with open(os.path.join(save_path, 'statistics.json'), 'w') as f:
                     json.dump(statistics, f)
       else:
              save_path = None

       # logger
       wandb_logger = Logger(args.project_name, args.dataset_name, args, save_path=args.log_path, rank=global_rank) 

       # agent and the model for agent
       visual_diffusion_policy = VisualDiffusion(view_num=len(view_list), 
                                                 output_dim=int(7 * args.ac_num),
                                                 **kwargs).to(rank)
       agent = VLDDPM_BC(policy=visual_diffusion_policy, **kwargs)      
       agent.get_statistics(os.path.join(args.save_path, 'statistics.json'))
       agent.get_transform(img_size=0)
       
       # evaluator
       evaluator = LIBEROEval(task_suite_name=kwargs['dataset_name'], data_statistics=None, eval_horizon=300, num_episodes=10, logger=wandb_logger, rank=global_rank, json_path=json_path)
       
       # trainer
       test_trainer = BCTrainer(agent, 
                                rt1dataloader, 
                                None, 
                                wandb_logger, 
                                evaluator,
                                num_steps=int(args.steps), 
                                lr=args.lr, 
                                device=rank,
                                global_rank=global_rank,
                                save=args.save,
                                save_freq=args.save_freq, 
                                save_path=save_path,
                                resume_path=args.resume,
                                img_goal=img_goal,
                                args=args
                     )
       test_trainer.train_steps()


if __name__ == '__main__':
       # get log
       args = get_args()
       main(args)