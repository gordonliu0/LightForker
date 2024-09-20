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


class LightFormerDataset(Dataset):
    """
    Custom LightFormer Dataset.
    https://pytorch.org/tutorials/beginner/basics/data_tutorial.html#creating-a-custom-dataset-for-your-files
    """

    def __init__(self, img_dir, transform=None):
        """
        Args:
            img_dir (list): Paths to the image directories, which should include a .json with annotations.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        # Properties:
        #   img_dir = original list of dataset directories
        #   landmarks_frame = samples
        #   root_dir_list = which root dir the sample is in
        #   transform = transform to be applied to each image in each sample

        self.img_dir = img_dir
        self.landmarks_frame = []
        self.root_dir_list = []
        self.transform = transform

        # handle a list of img_dir
        for sub_path in self.img_dir:
            json_files = sorted(os.path.join(sub_path, x) for x in os.listdir(sub_path) if (x.endswith('.json')))
            for x in json_files:
                with open(x, 'r') as opened_file:
                    sample = json.load(opened_file)
                    self.landmarks_frame += sample
                    self.root_dir_list += len(sample) * [Path(sub_path)]

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_names = self.landmarks_frame[idx]['images']
        root_dir = self.root_dir_list[idx]
        images = torch.from_numpy(np.zeros((10,3,512,960),dtype='float32'))
        for i,img_name in enumerate(img_names):
            image = io.imread(os.path.join(root_dir, '../frames',img_name))
            image = resize(image, (512, 960), anti_aliasing=True)
            # numpy image: H x W x C
            # torch image: C x H x W
            image = image.transpose((2, 0, 1)) / 255.0
            image = image.astype('float32')
            image = torch.from_numpy(image)
            image = self.transform(image)
            images[i] = image

        label = self.landmarks_frame[idx]['label'][:4]
        label = np.array([label])
        label = label.astype('float32')
        label = torch.from_numpy(label)
        label = label.squeeze()
        sample = {
            'images': images,
            'label': label,
            'name': img_names[0],
        }

        return sample

# class LightFormerDataset2(Dataset):
#     def __init__(self, json_path, image_norm):
#         self.all_json_files_path = []
#         self.all_samples = None
#         self.root_dir_list = None
#         self.transform = transforms.Normalize(mean=image_norm[0], std=image_norm[1])

#         if type(json_path) is list:
#             for i, sub_path in enumerate(json_path):
#                 # if i == 0:
#                 #     upper_level_path = sub_path
#                 #     all_folders = os.listdir(upper_level_path)
#                 #     for folder in all_folders:
#                 #         folder_path = os.path.join(upper_level_path, folder)
#                 #         sub_folders = os.listdir(folder_path)
#                 #         for sub_folder in sub_folders:
#                 #             sub_folder_path = os.path.join(folder_path, sub_folder)
#                 #             if os.path.exists(sub_folder_path + '/train'):
#                 #                 json_path = sub_folder_path + '/train' + '/new_samples.json'
#                 #                 self.all_json_files_path.append(json_path)
#                 # else:
#                 self.json_files = sorted(os.path.join(sub_path, x) for x in os.listdir(sub_path) if (x.endswith('.json')))
#                 for x in self.json_files:
#                     sample = json.load(open(x, 'r'))
#                     if self.all_samples is None:
#                         self.all_samples = sample
#                         self.root_dir_list = len(sample) * [Path(sub_path)]
#                     else:
#                         self.all_samples += sample
#                         self.root_dir_list += len(sample) * [Path(sub_path)]

#         for json_file_path in self.all_json_files_path:
#             sample = json.load(open(json_file_path, 'r'))
#             if self.all_samples is None:
#                 self.all_samples = sample
#                 self.root_dir_list = len(sample) * [Path(json_file_path)]
#             else:
#                 self.all_samples += sample
#                 self.root_dir_list += len(sample) * [Path(json_file_path)]


#     def __len__(self):
#         return len(self.all_samples)

#     def __getitem__(self, idx):
#         if torch.is_tensor(idx):
#             idx = idx.tolist()
#         img_names = self.all_samples[idx]['images']
#         root_dir = self.root_dir_list[idx]
#         # print(root_dir)
#         # print(os.path.dirname(os.path.dirname(root_dir)))
#         images = torch.from_numpy(np.zeros((10, 3, 256,480),dtype='float32'))
#         for i,img_name in enumerate(img_names):
#             if not os.path.exists(os.path.join(root_dir, '../frames')):
#                 # img =  io.imread(os.path.join(root_dir, '../../image_30_crop', img_name))
#                 if i%2 != 0:
#                     img =  io.imread(os.path.join(root_dir, '../../image_30_crop', img_name))
#                 else:
#                     img = io.imread(os.path.join(root_dir, '../../image_120_resize', img_name))

#             else:
#                 img = io.imread(os.path.join(root_dir, '../frames', img_name))
#                 img = resize(img,(512,960), anti_aliasing=True)
#             # print(img.shape)
#             # resized_img = resize(img,(1080,1920), anti_aliasing=True)
#             # croped_img = resized_img[362:618,720:1200]
#             img = img.transpose((2, 0, 1))/255.0
#             img = img.astype('float32')
#             img = torch.from_numpy(img)
#             img = self.transform(img)
#             images[i] = img

#         label = self.all_samples[idx]['label'][:4]
#         label = np.array([label])
#         label = label.astype('float32')

#         sample = {
#             'images': images,
#             'label': torch.from_numpy(label),
#             'name': img_names[0]
#         }

#         return sample
