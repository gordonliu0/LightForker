import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import  Dataset
from torchvision import transforms
import numpy as np
import json
import os
from skimage import io
from skimage.transform import resize
from pathlib import Path

img_dir = [ "/Users/gordonliu/Documents/ml_projects/LightForker/data_label/data/Kaggle_Dataset/dayTrain/dayTrain/dayClip1/train",
            "/Users/gordonliu/Documents/ml_projects/LightForker/data_label/data/Kaggle_Dataset/dayTrain/dayTrain/dayClip2/train",]
landmarks_frame = None
for sub_path in img_dir:
    print(f"On sub_path {sub_path}")
    json_files = sorted(os.path.join(sub_path, x) for x in os.listdir(sub_path) if (x.endswith('.json')))
    print("json_files is ", json_files)
    for x in json_files:
        print("Now on json file x", x)
        with open(x, 'r') as opened_file:
            sample = json.load(opened_file)
            if landmarks_frame is None:
                landmarks_frame = sample
                root_dir_list = len(sample) * [Path(sub_path)]
            else:
                landmarks_frame += sample
                root_dir_list += len(sample) * [Path(sub_path)]
root_dir = img_dir

print(landmarks_frame)
print(root_dir_list)
