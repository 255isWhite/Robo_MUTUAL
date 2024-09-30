import os

# 定义根文件夹路径
root_dirs = ['v2t', 't2v']

def rename_videos(folder):
    for subdir in os.listdir(folder):
        subdir_path = os.path.join(folder, subdir)
        if os.path.isdir(subdir_path):
            videos = os.listdir(subdir_path)
            videos.sort()  # 确保按照文件名排序
            for idx, video in enumerate(videos):
                video_path = os.path.join(subdir_path, video)
                new_video_name = f"{idx}.mp4"
                new_video_path = os.path.join(subdir_path, new_video_name)
                os.rename(video_path, new_video_path)
                print(f"Renamed {video_path} to {new_video_path}")

# 处理两个文件夹
for root_dir in root_dirs:
    rename_videos(root_dir)