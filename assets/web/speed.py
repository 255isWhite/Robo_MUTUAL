import os
from moviepy.editor import VideoFileClip

# 定义原始根文件夹和目标根文件夹
root_dirs = ['t2v']
target_root = 'speed3'

def process_videos(src_folder, dst_folder):
    for subdir in os.listdir(src_folder):
        src_subdir_path = os.path.join(src_folder, subdir)
        dst_subdir_path = os.path.join(dst_folder, subdir)
        
        # 如果目标子文件夹不存在，创建它
        os.makedirs(dst_subdir_path, exist_ok=True)
        
        if os.path.isdir(src_subdir_path):
            videos = os.listdir(src_subdir_path)
            videos.sort()  # 确保按照文件名排序
            for video in videos:
                src_video_path = os.path.join(src_subdir_path, video)
                dst_video_path = os.path.join(dst_subdir_path, video)
                
                # 加载视频并施加3倍速
                with VideoFileClip(src_video_path) as clip:
                    sped_up_clip = clip.speedx(factor=3)  # 正确调用 speedx
                    sped_up_clip.write_videofile(dst_video_path, codec="libx264", audio_codec="aac")
                
                print(f"Processed {src_video_path} to {dst_video_path}")

# 处理两个根文件夹
for root_dir in root_dirs:
    src_folder = root_dir
    dst_folder = os.path.join(target_root, root_dir)
    process_videos(src_folder, dst_folder)