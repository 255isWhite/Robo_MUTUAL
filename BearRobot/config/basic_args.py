import argparse
from BearRobot.utils.net.initialization import boolean

def basic_args():
       # basic argument
       parser = argparse.ArgumentParser(description='An example to use this model')
       parser.add_argument('--device', default='cuda', help='cuda or cpu')
       parser.add_argument('--project_name', default='Bearobot_pytorch_example', help='your project name')
       parser.add_argument('--algo_name', default='YOUR_ALGO', help='your algo name')
       parser.add_argument('--dataset_name', default='bridge', help='choose your d4rl env or some realrobot dataset')

       # basic training argument
       parser.add_argument('--steps', default=int(1e+6), type=int, help='train steps')
       parser.add_argument('--val_freq', default=int(5e+3), type=int, help='val frequency')
       parser.add_argument('--eval_freq', default=100, type=int, help='evaluation frequency, deploy the agent in sim env to evaluate')
       parser.add_argument("--seed", default=42, type=int)  # Sets PyTorch and Numpy seeds
       parser.add_argument('--batch_size', default=16, type=int)
       parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
       
       # save log parameters
       parser.add_argument('--save', default=True, type=boolean, help='save ckpt or not')
       parser.add_argument('--save_path', default='../experiments/test', type=str, help='ckpt save path')
       parser.add_argument('--save_freq', default=int(1e+4), type=int, help='save ckpt frequency')
       parser.add_argument('--log_path', default='../experiments/test', type=str, help='ckpt save path')
       parser.add_argument('--resume', default=None, type=str, help='resume path')
       parser.add_argument('--wandb', default=False, type=boolean, help='use wandb or not')
       
       # DataLoader parameters
       parser.add_argument('--num_workers', default=8, type=int)
       parser.add_argument('--pin-mem', action='store_true', help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
       parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem', help='')
       parser.set_defaults(pin_mem=True)
       
       # distributed training parameters
       parser.add_argument('--ddp', default=False, type=boolean, help='use ddp or not')
       parser.add_argument('--world_size', default=3, type=int, help='number of distributed processes')
       parser.add_argument('--port', default='22323', type=str, help='port')   
       return parser


def diffusion_args(parser: argparse.ArgumentParser):
       parser.add_argument('--beta', default='vp', type=str, help='noise schedule')
       parser.add_argument('--time_embed', default='learned', type=str, help='learned or fixed type time embedding')
       parser.add_argument('--T', default=15, type=int, help='diffusion max time step')
       parser.add_argument('--time_dim', default=32, type=int, help='time fourier feature dim')
       parser.add_argument('--time_hidden_dim', default=256, type=int, help='time fourier feature hidden dim')
       parser.add_argument('--sampler', default='ddpm', type=str, help='sampler, ddpm or ddim')
       return parser