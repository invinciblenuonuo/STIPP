import os
import random
import shutil
from tqdm import tqdm
import multiprocessing

source_dir='/mnt/HDD/dataset/processed_data'
train_dir='/mnt/HDD/dataset/waymo_train'
test_dir='/mnt/HDD/dataset/waymo_valid'

def copy_file(src, dst):
    shutil.copy2(src, dst)

def split_files(source_dir, ratio=0.8):
    """
    将文件夹中的文件按场景随机分成两份

    Args:
        source_dir: 源文件夹路径
        ratio: 训练集比例 (0-1)
    """

    # 获取所有文件
    files = os.listdir(source_dir)
    scenes = set([f.split('_')[0] for f in files])
    random.shuffle(list(scenes))
    split_index = int(len(scenes) * ratio)
    train_scenes = list(scenes)[:split_index]
    test_scenes = list(scenes)[split_index:]

    # with multiprocessing.Pool() as p:
    #     for file in tqdm(files,total=len(files)):
    #         scene = file.split('_')[0]

    #         src = os.path.join(source_dir, file)
    #         dst = train_dir if scene in train_scenes else test_dir
            
    #         p.starmap(copy_file, args=(src, dst))
    #         # if scene in train_scenes:
    #         #     shutil.copyfileobj(os.path.join(source_dir, file), train_dir)
    #         # else:
    #         #     shutil.copyfileobj(os.path.join(source_dir, file), test_dir)
    # p.close()
    # p.join()

    for file in tqdm(files,total=len(files)):
        scene = file.split('_')[0]
        # src = os.path.join(source_dir, file)
        # dst = train_dir if scene in train_scenes else test_dir            
        if scene in train_scenes:
            shutil.copy2(os.path.join(source_dir, file), train_dir)
        else:
            shutil.copy2(os.path.join(source_dir, file), test_dir)


split_files(source_dir)