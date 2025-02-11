from mmengine import fileio
import io
import os
import json
import yaml

import torch
from torch.nn.parallel import DistributedDataParallel as DDP

from BearRobot.Agent import *
from BearRobot.Agent.ddpm_bc import VLDDPM_BC
from BearRobot.Agent.ACT import ACTAgent 
from BearRobot.Net.my_model.diffusion_model import VisualDiffusion, VisualDiffusion_pretrain
from BearRobot.Net.my_model.ACT_model import ACTModel

from BearRobot.Net.my_model.Align_net import AlignNet

def openjson(path):
       value  = fileio.get_text(path)
       dict = json.loads(value)
       return dict


def convert_str(item):
        try:
                return int(item)
        except:
                try:
                        return float(item)
                except:
                        return item


def wandb_args2dict(ckpt_path, wandb_name: str=None):
        if wandb_name is None:
                wandb_name = 'latest-run'
        try:
                wandb_path = os.path.join('/'.join(ckpt_path.split('.pth')[0].split('/')[:-1]), f'wandb/{wandb_name}/files/wandb-metadata.json')
                meta_data = openjson(wandb_path)
                args = [convert_str(arg.split('--')[-1]) for arg in meta_data['args']]
                config_dict = dict(zip(args[::2], args[1::2]))
                print("-------load meta data from wandb succ!------------")
                print_dict = json.dumps(config_dict, indent=4, sort_keys=True)
                print(print_dict)
                return config_dict
        except:
                print("Automatically load wandb meta data fail, please provide your meta data mannually")
                return {}


def wandb_yaml2dict(ckpt_path, wandb_name: str=None, wandb_path: str=None):
        if wandb_name is None:
                wandb_name = 'latest-run'
        try:
                if wandb_path is None:
                        wandb_path = os.path.join('/'.join(ckpt_path.split('.pth')[0].split('/')[:-1]), f'wandb/{wandb_name}/files/config.yaml')
                with open(wandb_path, 'r') as stream:
                        config = yaml.safe_load(stream)
                del config['wandb_version']
                del config['_wandb']
                config_dict = {key: value['value'] for key, value in config.items()}
                print("-------load meta data from wandb succ!------------")
                print_dict = json.dumps(config_dict, indent=4, sort_keys=True)
                print(print_dict)
                return config_dict
        except:
                print("Automatically load wandb meta data fail, please provide your meta data mannually")
                return {}


def load_ckpt(agent, ckpt_path):
        from collections import OrderedDict
        
        # os.environ["MASTER_ADDR"] = "localhost"
        # os.environ["MASTER_PORT"] = '12346'
        # torch.cuda.set_device(0)
        # torch.distributed.init_process_group(backend='nccl', rank=0, world_size=1)
        # agent.policy = DDP(agent.policy, device_ids=[0], find_unused_parameters=False)

        ckpt = fileio.get(ckpt_path)
        with io.BytesIO(ckpt) as f:
                ckpt = torch.load(f, map_location='cpu')
        
        new_ckpt = OrderedDict()
        for key in ckpt['model'].keys():
                new_key = key.replace(".module", '')
                new_ckpt[new_key] = ckpt['model'][key]

        ckpt['model'] = new_ckpt
        agent.load_state_dict(ckpt['model'], strict=False)
        agent.eval()
        return agent.to(0)


def build_ACT(ckpt_path: str, statistics_path: str, wandb_name: str=None, wandb_path: str=None):
        kwargs = wandb_yaml2dict(ckpt_path, wandb_name, wandb_path=wandb_path)
        model = ACTModel(output_dim=7,
                         dim_feedforward=3200,
                         **kwargs).to(0)
        agent = ACTAgent(model)
        agent.get_statistics(statistics_path)
        agent.get_transform(kwargs['img_size'])
        return load_ckpt(agent, ckpt_path)


def build_visual_diffsuion(ckpt_path: str, statistics_path: str, wandb_name: str=None, wandb_path: str=None):
        kwargs = wandb_yaml2dict(ckpt_path, wandb_name, wandb_path=wandb_path)
        model = VisualDiffusion(view_num=2,
                                output_dim=7 * kwargs['ac_num'],
                                **kwargs).to(0)
        agent = VLDDPM_BC(model, T=kwargs['T'], beta=kwargs['beta'], ac_num=kwargs['ac_num'], text_encoder=kwargs['text_encoder'], add_noise=False) 
        agent.get_statistics(statistics_path)
        agent.get_transform(kwargs['img_size'])
        return load_ckpt(agent, ckpt_path)

def build_visual_diffusion_mmpretrain(ckpt_path: str, statistics_path: str, wandb_name: str=None, wandb_path: str=None):
        kwargs = wandb_yaml2dict(ckpt_path, wandb_name, wandb_path=wandb_path)
        model = VisualDiffusion_pretrain(view_num=2,
                                output_dim=7 * kwargs['ac_num'],
                                **kwargs).to(0)
        agent = VLDDPM_BC(model, T=kwargs['T'], beta=kwargs['beta'], ac_num=kwargs['ac_num'], text_encoder=kwargs['mm_encoder']) 
        agent.get_statistics(statistics_path)
        
        import torchvision.transforms as T
        transform_list  = [
              T.ToTensor(),
        ]
        agent.get_transform(img_size=0, transform_list=transform_list)
        return load_ckpt(agent, ckpt_path)


def build_visual_diffsuion_c3(ckpt_path: str, statistics_path: str, cross_modal: bool=True , align_net: str=None, wandb_name: str=None, wandb_path: str=None):
        kwargs = wandb_yaml2dict(ckpt_path, wandb_name, wandb_path=wandb_path)
        kwargs['device'] = 'cuda:0'
        if cross_modal and kwargs['add_noise'] and not kwargs['minus_mean']:
                kwargs['minus_mean'] = True
                kwargs['lang_fit_img'] = True
        model = VisualDiffusion(view_num=2,
                                output_dim=7 * kwargs['ac_num'],
                                **kwargs).to(0)
        agent = VLDDPM_BC(model, **kwargs)
        # agent.align_net = AlignNet(1024).to('cuda:0')
        # agent.align_net.load_state_dict(torch.load(align_net, map_location='cuda:0'))
        # agent.align_net.eval() 
        agent.get_statistics(statistics_path)
        agent.get_transform(kwargs['img_size'])
        return load_ckpt(agent, ckpt_path)


# def build_visual_diffusino(ckpt_path: str, statistics_path: str, wandb_name: str=None):
#         model = VisualDiffusion(img_size=224, 
#                                 view_num=2, 
#                                 output_dim=28, 
#                                 device=0, 
#                                 vision_pretrained=False, 
#                                 vision_encoder='resnet50', 
#                                 add_spatial_coordinates=True,
#                                 norm_type='bn',
#                                 encode_a=True,
#                                 encode_s=True,
#                                 s_dim=7).to(0)
#         agent = VLDDPM_BC(model, num_timesteps=25, schedule='vp')      
#         agent.get_statistics(statistics_path)
#         agent.get_transform()
#         agent = load_ckpt(agent, ckpt_path)
#         return agent


if __name__ == '__main__':      
        # agent = build_ACT("/home/dodo/ljx/BearRobot/experiments/airkitchen/ACT/40W_v0/latest.pth",
        #                   "/home/dodo/ljx/BearRobot/experiments/airkitchen/diffusion/ACT_20W_T25_bn_spatial_normtrans_saencoder/statistics.json")
        

        agent = build_visual_diffsuion("/home/dodo/ljx/BearRobot/experiments/airkitchen/diffusion/ACT_40W_T25_bn_spatial_normtrans_salinear/399999_0.2852.pth",
                                       "/home/dodo/ljx/BearRobot/experiments/airkitchen/diffusion/ACT_20W_T25_bn_spatial_normtrans_saencoder/statistics.json",
                                       wandb_path="/home/dodo/ljx/BearRobot/experiments/airkitchen/diffusion/ACT_40W_T25_bn_spatial_normtrans_salinear/offline-run-20240424_160423-u1r3yiwo/files/config.yaml")