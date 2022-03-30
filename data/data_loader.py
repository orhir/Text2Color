from io import BytesIO

import numpy as np
import lmdb
from PIL import Image
from skimage import color
import torch
from torch.utils.data import Dataset
import torchvision.datasets as dset
import torchvision.transforms as transforms
import random


def RGB2Lab(inputs):
    return color.rgb2lab(inputs)


def Normalize(inputs):
    # output l [-50,50] ab[-128,128]
    l = inputs[:, :, 0:1]
    ab = inputs[:, :, 1:3]
    l = l - 50
    # ab = ab
    lab = np.concatenate((l, ab), 2)

    return lab.astype('float32')


def selfnormalize(inputs):
    d = torch.max(inputs) - torch.min(inputs)
    out = (inputs) / d
    return out


def to_gray(inputs):
    img_gray = np.clip((np.concatenate((inputs[:, :, :1], inputs[:, :, :1], inputs[:, :, :1]), 2) + 50) / 100 * 255, 0,
                       255).astype('uint8')

    return img_gray


def numpy2tensor(inputs):
    out = torch.from_numpy(inputs.transpose(2, 0, 1))
    return out


class MultiResolutionDataset(Dataset):
    def __init__(self, transform, resolution=256):
        self.resolution = resolution
        self.transform = transform
        self.dataset = dset.ImageFolder(root='datasets/train',
                                     transform=transform)
        # self.dataset = dset.CocoCaptions(root='datasets/train2014',
        #                                  annFile='datasets/annotations/captions_train2014.json',
        #                                  transform=transform)
        self.length = len(self.dataset)
        print("{} Images in dataset".format(self.length))
    def __len__(self):
        return self.length

    def __getitem__(self, index):
        rand = random.randint(0, self.length)
        img, caption = self.dataset[rand]
        img_src = np.array(img)  # [0,255] uint8

        ## add gaussian noise
        img_lab = Normalize(RGB2Lab(img_src))  # l [-50,50] ab [-128, 128]

        img = img_src.astype('float32')  # [0,255] float32 RGB

        img = numpy2tensor(img)
        img_lab = numpy2tensor(img_lab)

        return img, caption, img_lab
