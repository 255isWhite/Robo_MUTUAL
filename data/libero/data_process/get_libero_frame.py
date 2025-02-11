import re
import os
import torch
from mmengine import fileio
import json
from data.libero.data_process import demo2frames

import random


def get_demofixed_begin_end_frame(step, frame_length_dic):
    """return a fixed begin and fixed end frame

    Args:
        step (dict): one dict contains one step info of the libero demo, such as D435_image, wrist_image....
        frame_length_dic (dict): _description_

    Returns:
        _type_: _description_
    """
    
    pattern = re.compile(r'.+/(libero_.+)/(.+_demo)/demo_(\d+)/')
    match = pattern.search(step['D435_image'])
    
    if match:
            task_suite = match.group(1)
            instruction = match.group(2)
            demo_id = match.group(3)
            key = f"{task_suite}/{instruction}/demo_{demo_id}"
            frame_length = frame_length_dic.get(key, 100) - 1 # Default to 1000 if not found

    img_begin_path = step['D435_image'].replace(step['D435_image'].split("/")[-1], "0.jpg")
    img_end_path = step['D435_image'].replace(step['D435_image'].split("/")[-1], f"{frame_length}.jpg")

    return img_begin_path, img_end_path

def get_demofixed_random_frame(step,base_dir,frame_length_dic,img_base_dir = "libero/data_jpg/libero_goal/"):
    '''
    return a random begin and random end frame
    '''
    pattern = re.compile(r'([^/]+_demo)/demo_(\d+)/')

    match = pattern.search(step['D435_image'])
    
    if match:
            instruction = match.group(1)
            demo_id = match.group(2)
            key = f"{instruction}/demo_{demo_id}"
            frame_length = frame_length_dic.get(key, 100) - 1 # Default to 1000 if not found

    pattern_step_idx = re.compile(r'/(\d+)\.jpg$')
    match = pattern_step_idx.search(step['D435_image'])
    
    step_idx = int(match.group(1))
    if step_idx == 0:
        idx0 = step_idx
    else:
        idx0 = torch.randint(0, step_idx, (1,)).item()
    
    if step_idx == frame_length:
        idx1 = step_idx
    else:
        idx1 = torch.randint(step_idx, frame_length, (1,)).item()

    img_begin_path = os.path.join(base_dir,img_base_dir,key,f"image0/{idx0}.jpg")
    img_end_path = os.path.join(base_dir,img_base_dir,key,f"image0/{idx1}.jpg")

    return img_begin_path,img_end_path

def get_demofixed_idx_begin_frame(step,base_dir,frame_length_dic,img_base_dir = "libero/data_jpg/libero_goal/"):
    '''
    return current frame as begin frame and a random end frame
    '''
    pattern = re.compile(r'([^/]+_demo)/demo_(\d+)/')

    match = pattern.search(step['D435_image'])
    
    if match:
            instruction = match.group(1)
            demo_id = match.group(2)
            key = f"{instruction}/demo_{demo_id}"
            frame_length = frame_length_dic.get(key, 100) - 1 # Default to 1000 if not found

    pattern_step_idx = re.compile(r'/(\d+)\.jpg$')
    match = pattern_step_idx.search(step['D435_image'])
    
    step_idx = int(match.group(1))

    if step_idx == frame_length:
        idx1 = step_idx
    else:
        idx1 = torch.randint(step_idx, frame_length, (1,)).item()

    img_begin_path = os.path.join(base_dir,img_base_dir,key,f"image0/{step_idx}.jpg")
    img_end_path = os.path.join(base_dir,img_base_dir,key,f"image0/{idx1}.jpg")

    return img_begin_path, img_end_path

def get_demorandom_begin_end_frame(step, base_dir, frame_length_dic, img_base_dir = "libero/data_jpg/libero_goal/"):

    instruction = step['instruction']
    task_name = instruction.replace(" ", "_") + "_demo"
    demo_paths = demo2frames.get_demos_for_task(task_name, frame_length_dic)
    demo_path = random.choice(demo_paths)
    end_idx = frame_length_dic[demo_path] - 1
    
    img_begin_path = os.path.join(base_dir, img_base_dir, demo_path, "image0/0.jpg")
    img_end_path = os.path.join(base_dir, img_base_dir, demo_path, f"image0/{end_idx}.jpg")
    
    return img_begin_path, img_end_path
    

def get_demo_length(step,base_dir,frame_length_dic,img_base_dir = "libero/data_jpg/libero_goal/"):
    '''
    return demo length
    '''
    pattern = re.compile(r'([^/]+_demo)/demo_(\d+)/')

    match = pattern.search(step['D435_image'])
    
    frame_length = 100
    
    if match:
            instruction = match.group(1)
            demo_id = match.group(2)
            key = f"{instruction}/demo_{demo_id}"
            frame_length = frame_length_dic.get(key, 100) - 1 # Default to 1000 if not found

    return frame_length


def test_func():
    global base_dir,datalist
    base_dir='/home/dodo/ljx/BearRobot/data/libero/dataset/'
    datalist=['/home/dodo/ljx/BearRobot/data/libero/libero_spatial-ac.json']
    datalist_img = []  

    def openjson(path):
           value  = fileio.get_text(path)
           dict = json.loads(value)
           return dict
    
    for one_list in datalist: 
        datalist = openjson(one_list)
        datalist_img += datalist
        
    random_index = torch.randint(0, 10000, (1,)).item()
    step = datalist_img[random_index]

    return step

if __name__ == "__main__":
    json_path = '/home/dodo/ljx/BearRobot/data/libero/libero_spatial-ac.json'
    get_demofixed_begin_end_frame(test_func(), demo2frames.frame_counts_dict(json_path))