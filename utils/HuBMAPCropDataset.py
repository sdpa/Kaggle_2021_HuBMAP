from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import tifffile as tiff
from .config import options
from . import joint_transforms
import os
import torch
from torchvision import transforms
import random
from PIL import Image


def rle_to_mask(_encoding, size):
    # encoding is a string. convert it to int.
    mask = np.zeros(size[0] * size[1], dtype=np.uint8)
    encoding_np = np.fromstring(_encoding, dtype=int, sep=' ')
    starts = np.array(encoding_np)[::2]
    lengths = np.array(encoding_np)[1::2]
    for start, length in zip(starts, lengths):
        mask[start:start + length] = 1
    return mask.reshape(size, order='F')


class HuBMAPCropDataset(Dataset):
    def __init__(self, base_dir, mode, patient=None):
        self.base_dir = base_dir
        self.mode = mode
        self.slice_indexes = []
        self.images = []
        self.masks = []  # The masks for 9 training patients.
        self.global_img = None
        self.train_transform = transforms.Compose([
            joint_transforms.JointRandomCrop(options.train_window),
            joint_transforms.JointRandomHorizontalFlip(),
            joint_transforms.JointRandomVerticalFlip(),
            joint_transforms.JointRandomRotation()
        ])

        if mode == "train":
            # Get all training patients.
            train_pats = pd.read_csv(options.kaggle_data_path + "/tiffs/trainData/train.csv")
            leave_out_name = train_pats.iloc[options.test_tiff_value]['id']
            images = os.listdir(options.kaggle_data_path + "/improved_crops/ImgCrops")
            masks = os.listdir(options.kaggle_data_path + "/improved_crops/maskCrops")
            self.images = [x for x in images if leave_out_name not in x]
            self.masks = [x for x in masks if leave_out_name not in x]
            del images, masks
        elif mode == "val":
            # Open tiff of validation tiff file
            train_pats = pd.read_csv(options.kaggle_data_path + "/tiffs/trainData/train.csv")
            self.valid_tiff_name = train_pats.iloc[options.test_tiff_value]['id']
            tiff_file = tiff.imread(options.kaggle_data_path + "/tiffs/trainData/" + self.valid_tiff_name + '.tiff')

            if len(tiff_file.shape) > 3:
                tiff_file = tiff_file.squeeze(0).squeeze(0)
                tiff_file = np.moveaxis(tiff_file, 0, -1)

            grid = self.make_grid((tiff_file.shape[0], tiff_file.shape[1]), window=options.test_window)
            if len(self.slice_indexes) > 0:
                self.slice_indexes = np.concatenate((self.slice_indexes, grid), axis=0)
            else:
                self.slice_indexes = grid
            self.global_img = tiff_file
            self.val_tiff_shape = tiff_file.shape  # tuple (height, width, channels)

            # Load global mask for validation image
            encoding = train_pats.loc[train_pats["id"] == self.valid_tiff_name]['encoding'].iloc[0]
            self.global_mask = rle_to_mask(encoding, (self.val_tiff_shape[0], self.val_tiff_shape[1]))

        elif mode == "test":
            tiff_file = tiff.imread("/home/cougarnet.uh.edu/srizvi7/Desktop/Kaggle_2021_HuBMAP/testData/"
                                    + patient + '.tiff')
            if len(tiff_file.shape) > 3:
                tiff_file = tiff_file.squeeze(0).squeeze(0)
                tiff_file = np.moveaxis(tiff_file, 0, -1)
            grid = self.make_grid((tiff_file.shape[0], tiff_file.shape[1]), window=options.test_window)
            if len(self.slice_indexes) > 0:
                self.slice_indexes = np.concatenate((self.slice_indexes, grid), axis=0)
            else:
                self.slice_indexes = grid
            self.global_img = tiff_file

    def make_grid(self, shape, window=options.test_window, min_overlap=0):
        # y = rows, x = cols
        y, x = shape
        nx = x // (window - min_overlap) + 1
        x1 = np.linspace(0, x, num=nx, endpoint=False, dtype=np.int64)
        x1[-1] = x - window
        x2 = (x1 + window).clip(0, x)
        ny = y // (window - min_overlap) + 1
        y1 = np.linspace(0, y, num=ny, endpoint=False, dtype=np.int64)
        y1[-1] = y - window
        y2 = (y1 + window).clip(0, y)
        slices = np.zeros((nx, ny, 4), dtype=np.int64)
        for i in range(nx):
            for j in range(ny):
                slices[i, j] = x1[i], x2[i], y1[j], y2[j]
        return slices.reshape(nx * ny, 4)

    def get_global_image_size(self):
        # Returns rows, columns
        return self.global_img.shape[0], self.global_img.shape[1]

    def __getitem__(self, index):
        if self.mode == "train":
            file_name = self.images[index]
            img = Image.open(options.kaggle_data_path + "/improved_crops/ImgCrops/" + file_name)
            mask = Image.open(options.kaggle_data_path + "/improved_crops/maskCrops/" + file_name)

            # Augmentations (and need to crop 1024x1024 to 512x512)
            if random.random() < 0.5:  # Probabilities are from training notebook of Kidney Inference (Score 0.865)
                img = transforms.ColorJitter(brightness=0.2, contrast=0.2, hue=.05, saturation=.05)(img)
            if random.random() < 0.1:
                img = transforms.GaussianBlur(kernel_size=(3, 3))(img)

            aug_arr = self.train_transform([img, mask])  # RandomCrop, horiz and vertical flip, random rotation
            img, mask = aug_arr[0], aug_arr[1]

            # Image normalization
            img = np.array(img)
            if img.max() != img.min():
                img = (img - img.min()) / (img.max() - img.min())

            # img = np.array(img)
            img = np.moveaxis(img, -1, 0)
            mask = np.array(mask)

            return {'image': torch.Tensor(img), 'mask': torch.Tensor(mask).unsqueeze(0)}

        elif self.mode == "val":
            coordinate = self.slice_indexes[index]
            x1, x2, y1, y2 = coordinate[0], coordinate[1], coordinate[2], coordinate[3]
            img = self.global_img[y1:y2, x1:x2, :]  # (512, 512, 3)
            mask = self.global_mask[y1:y2, x1:x2]  # (512, 512)
            mask = np.expand_dims(mask, axis=2)  # (512,512) -> (512,512,1)

            img = transforms.ToTensor()(img)  # [3, 512, 512]
            mask = transforms.ToTensor()(mask) * 255  # [1, 512, 512]
            coordinate = torch.tensor([int(x1), int(x2), int(y1), int(y2)])

            return {'image': img, 'mask': mask, 'coords': coordinate}

        elif self.mode == "test":
            coordinate = self.slice_indexes[index]
            x1, x2, y1, y2 = coordinate[0], coordinate[1], coordinate[2], coordinate[3]
            img = self.global_img[y1:y2, x1:x2, :]
            coordinate = torch.tensor([int(x1), int(x2), int(y1), int(y2)])
            img = transforms.ToTensor()(img)

            # TTA (Test Time Augmentations) for Kaggle predictions
            img_horiz = transforms.RandomHorizontalFlip(p=1)(img)
            img_vert = transforms.RandomVerticalFlip(p=1)(img)

            return img, coordinate, img_horiz, img_vert

    def __len__(self):
        if self.mode == "train":
            return len(self.images)
        elif self.mode == "test" or self.mode == 'val':
            return len(self.slice_indexes)


"""
import matplotlib.pyplot as plt
import numpy as np

plt.imshow(np.moveaxis(pred_mask[0], 0, -1))
plt.show()
"""
