import wandb
    
from BearRobot.utils.logger.base_log import BaseLogger

class WandbLogger(BaseLogger):
    def __init__(self, project_name, run_name, args, save_path, record_freq: int=100, rank=0):
              super(BaseLogger).__init__()
              """
              project_name (str): wandb project name
              config: dict or argparser
              """              
              if rank == 0:
                wandb.init(project=project_name, name=run_name)
                wandb.config.update(args)
              else:
                wandb.init(mode="disabled")

              self.record_freq = record_freq

    def log_metrics(self, metrics: dict, step: int):
        """
            metrics (dict):
            step (int, optional): epoch or step
        """
        wandb.log(metrics, step=step)

    def finish(self):
        wandb.finish()
