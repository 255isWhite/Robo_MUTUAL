from BearRobot.utils.logger.base_log import BaseLogger

class BaseEval():
       def __init__(
              self,
              num_episodes: int,
              eval_freq: int,
              data_statistics: dict,
              logger: BaseLogger,
       ):
              super().__init__()
              self.num_episodes = num_episodes
              self.eval_freq = eval_freq
              self.data_statistics = data_statistics
              self.logger = logger
              
       def eval_episodes(self):
              """
              evaluation for num_episodes
              """
              raise NotImplementedError

       def _log_results(self, metrics: dict, steps: int):
              """
              log the results to the logger or print out
              """
              raise NotImplementedError