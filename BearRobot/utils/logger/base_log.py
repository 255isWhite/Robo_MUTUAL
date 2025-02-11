class BaseLogger():
       def __init__(self, args):
            self.args = args
       
       def log_metrics(self, metrics: dict, step: int):
              """
              log the evaluation results
              """
              raise NotImplementedError
       
       def save_metrics(self, metrics: dict, step: int, save_path: str):
              """
              save the evaluation results to the save_path
              """
              raise NotImplementedError

       def finish(self):
              """
              close the logger
              """
              raise NotImplementedError