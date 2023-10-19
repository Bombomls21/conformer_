import glob
import os
import pickle

import numpy as np
import torch.utils.data as data
from PIL import Image

class PedesAttr(data.Dataset):
    def __init__(self, cfg, path, transform=None, target_transform=None):
        data_path = path
        print("which pickle: ", data_path)
        dataset_info = pickle.load(open(data_path, 'rb+'))

        img_id = dataset_info['image_name']
        attr_label = dataset_info['label']
        attr_label[attr_label == 2] = 0

        self.attr_id = dataset_info['attr_name']
        self.attr_num = len(self.attr_id)

        self.dataset = cfg.dataset
        self.transform = transform
        self.target_transform = target_transform

        self.root_path = '/home/server12gb/Desktop/Bach/UparChallenge_baseline/data'

        if self.target_transform:
            self.attr_num = len(self.target_transform)
            print(f'target_label: {self.target_transform}')
        else:
            self.attr_num = len(self.attr_id)
            print('target_label: all')

        self.img_num = len(img_id)
        self.img_id = img_id
        self.label = attr_label

    def __getitem__(self, index):
        imgname, gt_label = self.img_id[index], self.label[index]

        imgpath = os.path.join(self.root_path, imgname)
        img = Image.open(imgpath)

        if self.transform is not None:
            img = self.transform(img)

        gt_label = gt_label.astype(np.float32)

        if self.target_transform:
            gt_label = gt_label[self.target_transform]

        return img, gt_label, imgname

    def __len__(self):
        return len(self.img_id)

class PedesAttrVal(data.Dataset):
    def __init__(self, cfg, transform=None, idx=None):
        val_path = cfg.test_pickle
        print("which pickle: ", val_path) 
        val_info = pickle.load(open(val_path, 'rb+'))

        self.img_path = val_info.image_path
        self.attribute = val_info.attribute
        self.transform = transform
        self.root_path = '/home/server12gb/Desktop/Bach/UparChallenge_baseline/data'

    def __getitem__(self, index):
        val_image_name = self.img_path[index]
        val_image_path = os.path.join(self.root_path, val_image_name)
        img = Image.open(val_image_path)
        if self.transform is not None:
            img = self.transform(img)
        return img, val_image_name,  # noisy_weight

    def __len__(self):
        return len(self.img_path)