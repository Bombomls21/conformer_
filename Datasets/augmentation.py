import random
import torch
import numpy as np
import torchvision.transforms as T
from PIL import Image

def get_transform(cfg):
    height = 224
    width = 224
    # height = 384
    # width = 192
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


    train_transform = T.Compose([
        T.Resize((height, width)),
        T.Pad(10),
        T.RandomCrop((height, width)),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        normalize,
    ])

    valid_transform = T.Compose([
        T.Resize((height, width)),
        T.ToTensor(),
        normalize
    ])


    return train_transform, valid_transform
