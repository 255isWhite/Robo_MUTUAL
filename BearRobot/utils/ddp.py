import os
import torch
import torch.distributed as dist
from torch.distributed import init_process_group, destroy_process_group
import subprocess


def ddp_setup_universal(verbose=False, args=None):
       if args.ddp == False:
              print(f"do not use ddp, train on GPU 0")
              return 0, 0, 1
       
       if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
              rank = int(os.environ["RANK"])
              world_size = int(os.environ['WORLD_SIZE'])
              gpu = int(os.environ['LOCAL_RANK'])
              os.environ['MASTER_PORT'] = str(getattr(args, 'port', '29529'))
              os.environ["MASTER_ADDR"] = "localhost"
       elif 'SLURM_PROCID' in os.environ:
              rank = int(os.environ['SLURM_PROCID'])
              gpu = rank % torch.cuda.device_count()
              world_size = int(os.environ['SLURM_NTASKS'])
              node_list = os.environ['SLURM_NODELIST']
              num_gpus = torch.cuda.device_count()
              addr = subprocess.getoutput(f'scontrol show hostname {node_list} | head -n1')
              os.environ['MASTER_PORT'] = str(args.port)
              os.environ['MASTER_ADDR'] = addr
       else:
              print("Not using DDP mode")
              return 0, 0, 1

       os.environ['WORLD_SIZE'] = str(world_size)
       os.environ['LOCAL_RANK'] = str(gpu)
       os.environ['RANK'] = str(rank)              

       torch.cuda.set_device(gpu)
       dist_backend = 'nccl'
       dist_url = "env://"
       print('| distributed init (rank {}): {}, gpu {}'.format(rank, dist_url, gpu), flush=True)
       init_process_group(backend=dist_backend, world_size=world_size, rank=rank)
       torch.distributed.barrier()
       if verbose:
              setup_for_distributed(rank == 0)
       return rank, gpu, world_size
      

def ddp_setup(rank: int, world_size: int, verbose: bool=False, port: str="12355"):
       """
       Args:
       rank: Unique identifier of each process
       world_size: Total number of processes
       """
       os.environ["MASTER_ADDR"] = "localhost"
       os.environ["MASTER_PORT"] = port
       torch.cuda.set_device(rank)
       init_process_group(backend="nccl", rank=rank, world_size=world_size)
       torch.distributed.barrier()
       if verbose:
              setup_for_distributed(rank == 0)
       print(f"{rank}"*88)
       return rank, rank, world_size


def ddp_setup_slurm(verbose=False, args=None):
       rank = int(os.environ['SLURM_PROCID'])
       gpu = rank % torch.cuda.device_count()
       world_size = int(os.environ['SLURM_NTASKS'])
       node_list = os.environ['SLURM_NODELIST']
       num_gpus = torch.cuda.device_count()
       addr = subprocess.getoutput(f'scontrol show hostname {node_list} | head -n1')
       os.environ['MASTER_PORT'] = str(args.port)
       os.environ['MASTER_ADDR'] = addr
       os.environ['WORLD_SIZE'] = str(world_size)
       os.environ['LOCAL_RANK'] = str(gpu)
       os.environ['RANK'] = str(rank)

       torch.cuda.set_device(gpu)
       dist_backend = 'nccl'
       dist_url = "env://"
       print('| distributed init (rank {}): {}, gpu {}'.format(rank, dist_url, gpu), flush=True)
       init_process_group(backend=dist_backend, world_size=world_size, rank=rank)
       torch.distributed.barrier()
       if verbose:
              setup_for_distributed(rank == 0)
       return rank, gpu, world_size
       

def init_distributed_mode(args, verbose=False):
       print('1'*88)
       if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
              args.rank = int(os.environ["RANK"])  # global gpu id
              args.world_size = int(os.environ['WORLD_SIZE']) # process num
              args.gpu = int(os.environ['LOCAL_RANK']) # 0~7 local gpu id
              os.environ['MASTER_PORT'] = "12355"
              os.environ['MASTER_ADDR'] = 'localhost'
              print("2"*88)
       else:
              print('Not using distributed mode')
              args.distributed = False
              return
       print("3"*88)
       args.distributed = True

       torch.cuda.set_device(args.rank)
       args.dist_backend = 'nccl'
       print('| distributed init (rank {}): {}, gpu {}'.format(
              args.rank, args.dist_url, args.gpu), flush=True)
       init_process_group(backend=args.dist_backend, world_size=args.world_size, rank=args.rank)
       print("4"*88)
       torch.distributed.barrier()
       if verbose:
              setup_for_distributed(args.rank == 0)
       print("5"*88)
              

def setup_for_distributed(is_master):
       """
       This function disables printing when not in master process
       """
       import builtins as __builtin__
       builtin_print = __builtin__.print

       def print(*args, **kwargs):
              force = kwargs.pop('force', False)
              if is_master or force:
                     builtin_print(*args, **kwargs)

       __builtin__.print = print


def is_dist_avail_and_initialized():
       if not dist.is_available():
              return False
       if not dist.is_initialized():
              return False
       return True

   
def get_world_size():
       if not is_dist_avail_and_initialized():
              return 1
       return dist.get_world_size()


def get_rank():
       if not is_dist_avail_and_initialized():
              return 0
       return dist.get_rank()


def is_main_process():
       return get_rank() == 0