import os
import json
import re,random
from collections import defaultdict

def frame_counts_dict(json_path='/home/dodo/ljx/BearRobot/data/libero/libero_goal-ac.json'):
    with open(json_path) as json_file:
        data = json.load(json_file)
    demo_count = defaultdict(int)
    pattern = re.compile(r'.+/(libero_.+)/(.+_demo)/demo_(\d+)/')
    for item in data:
        match = pattern.search(item['D435_image'])
        # match = pattern.search(item['wrist_image'])
        if match:
            task_suite = match.group(1)
            instruction = match.group(2)
            demo_id = match.group(3)
            key = f"{task_suite}/{instruction}/demo_{demo_id}"
            demo_count[key] += 1
    return demo_count

def get_demos_for_task(task_name, frame_length_dict):
    pattern = re.compile(rf'.+{task_name}/demo_\d+')
    matched_demos = [match.group(0) for key in frame_length_dict for match in [pattern.search(key)] if match]
    return matched_demos

def get_random_demo_img_path(demo_path, idx, base_dir='/home/dodo/ljx/BearRobot/data/libero/dataset/'):
    path = os.path.join(base_dir, demo_path, f"image0/{idx}.jpg")
    return path

def test_func():
    frame_length_dict = frame_counts_dict()
    task_description = "open the middle drawer of the cabinet" 
    task_name = task_description.replace(" ", "_")+ "_demo"
    demo_paths = get_demos_for_task(task_name, frame_length_dict)
    demo_path = random.choice(demo_paths)

if __name__ == "__main__":
    test_func()