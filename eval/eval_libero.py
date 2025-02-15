from BearRobot.utils.evaluation.mp_libero_eval import LIBEROEval
from BearRobot.Agent import build_visual_diffsuion_c3, build_visual_diffsuion
import argparse
import os

def get_args():
        parser = argparse.ArgumentParser(description='args for err evaluation')
        
        # crucial eval parameters
        parser.add_argument('--img_goal', default=True, type=bool, help='whether to use goal for evaluation')
        
        parser.add_argument('--basic_path', default='', type=str, help='path to the checkpoint and statistics and wandb')
        parser.add_argument('--statistic_name', default='statistics.json', type=str, help='statistics file name')
        parser.add_argument('--ckpt_name', default='', type=str, help='checkpoint file name')
        parser.add_argument('--save_path', default='', type=str, help='path to save the evaluation results')
        parser.add_argument('--json_path', default='', type=str, help='path to the json file')
        parser.add_argument('--task_suite_name', default='libero130', type=str, help='task suite name')
        parser.add_argument('--num_episodes', default=10, type=int, help='number of episodes to evaluate')
        parser.add_argument('--eval_horizon', default=300, type=int, help='horizon for evaluation')
            
        args = parser.parse_args()    
        return args   
    
    

def main(args):
    ckpt_path = os.path.join(args.basic_path, args.ckpt_name)
    statistic_path = os.path.join(args.basic_path, args.statistic_name)

    policy = build_visual_diffsuion_c3(ckpt_path, statistic_path, img_goal= args.img_goal, wandb_path=os.path.join(args.basic_path,"wandb/latest-run/files/config.yaml"))
    evaluator = LIBEROEval(task_suite_name=args.task_suite_name, data_statistics=None, eval_horizon=args.eval_horizon, \
                            num_episodes=args.num_episodes, json_path=args.json_path)

    evaluator.eval_episodes(policy, 0, save_path=args.basic_path, img_goal=args.img_goal)

if __name__ == '__main__':
    args = get_args()
    main(args)