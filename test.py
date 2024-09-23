import sys
sys.path.append('.')
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from dataset.dataset import LightFormerDataset
from pathlib import Path
from functools import partial


# Test LightFormer Dataset

def test_dataset():
    image_norm = [(0.485, 0.456, 0.406), (0.229, 0.224, 0.225)]
    transform = transforms.Normalize(mean=image_norm[0], std=image_norm[1])
    train_set = LightFormerDataset( [
                    "/Users/gordonliu/Documents/ml_projects/LightForker/data_label/data/Kaggle_Dataset/dayTrain/dayTrain/dayClip1",
                    "/Users/gordonliu/Documents/ml_projects/LightForker/data_label/data/Kaggle_Dataset/dayTrain/dayTrain/dayClip2",], transform)

def test_squeeze():
     x = torch.zeros(2, 1, 2, 1, 2)
     x = x.squeeze()
     print(x.size())

def main():
    test_squeeze()

if __name__ == '__main__':
    main()
