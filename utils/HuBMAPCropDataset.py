from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import tifffile as tiff
from .config import options
from . import joint_transforms
import os
import torch
from torchvision import transforms
# import cv2
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
            joint_transforms.JointRandomHorizontalFlip()
        ])

        if mode == "train":
            # Get all training patients.
            train_pats = pd.read_csv("/home/cougarnet.uh.edu/srizvi7/Desktop/Kaggle_2021_HuBMAP/trainData/train.csv")
            leave_out_name = train_pats.iloc[options.test_tiff_value]['id']
            images = os.listdir(self.base_dir + "/ImgCrops")
            masks = os.listdir(self.base_dir + "/maskCrops")
            if mode == "train":
                self.images = [x for x in images if leave_out_name not in x]
                self.masks = [x for x in masks if leave_out_name not in x]
                del images, masks
            elif mode == "val":
                self.images = [x for x in images if leave_out_name in x]
                self.masks = [x for x in masks if leave_out_name in x]
                del images, masks
        elif mode == "val":
            # Open tiff of validation tiff file
            train_pats = pd.read_csv("/home/cougarnet.uh.edu/srizvi7/Desktop/Kaggle_2021_HuBMAP/trainData/train.csv")
            self.valid_tiff_name = train_pats.iloc[options.test_tiff_value]['id']
            tiff_file = tiff.imread("/home/cougarnet.uh.edu/srizvi7/Desktop/Kaggle_2021_HuBMAP/trainData/" +
                                    self.valid_tiff_name + '.tiff')

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
            tiff_file = tiff.imread("/home/cougarnet.uh.edu/srizvi7/Desktop/Kaggle_221_HuBMAP/testData/"
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
            img = Image.open(self.base_dir + "/ImgCrops/" + file_name)
            mask = Image.open(self.base_dir + "/maskCrops/" + file_name)

            # Augmentations (and need to crop 1024x1024 to 512x512
            aug_arr = self.train_transform([img, mask])
            img, mask = aug_arr[0], aug_arr[1]

            # Image normalization
            img = np.asarray(img)
            if img.max() != img.min():
                img = (img - img.min()) / (img.max() - img.min())
            # img = Image.fromarray((img * 255).astype('uint8'))
            img = img * 255

            # img = np.array(img)
            img = np.moveaxis(img, -1, 0)
            mask = np.array(mask)

            return {'image': torch.from_numpy(img), 'mask': torch.from_numpy(mask).unsqueeze(0)}

            # img = transforms.ColorJitter(brightness=0.5, contrast=0.5, hue=.05, saturation=.05).forward(img)
            # img = transforms.GaussianBlur(kernel_size=(3,3)).forward(img)
            # img = transforms.ToTensor()(img)
            # mask = transforms.ToTensor()(mask)
            # merged = torch.cat((img, mask), 0)
            # crop = transforms.RandomCrop((options.train_window, options.train_window),
            #                              pad_if_needed=True).forward(merged)
            # crop = transforms.RandomHorizontalFlip().forward(crop)
            # crop = transforms.RandomVerticalFlip().forward(crop)
            # crop = transforms.RandomRotation(90).forward(crop)
            # # # Split them back.
            # img = crop[:3, :, :]
            # mask = crop[-1:, :, :]
            # mask = mask * 255
            # return img, mask
        elif self.mode == "val":
            coordinate = self.slice_indexes[index]
            x1, x2, y1, y2 = coordinate[0], coordinate[1], coordinate[2], coordinate[3]
            img = self.global_img[y1:y2, x1:x2, :]
            mask = self.global_mask[y1:y2, x1:x2]

            img = transforms.ToTensor()(img)
            mask = transforms.ToTensor()(mask) * 255
            coordinate = torch.tensor([int(x1), int(x2), int(y1), int(y2)])

            return {'image': img, 'global_mask': mask, 'coords': coordinate}

        elif self.mode == "test":
            coordinate = self.slice_indexes[index]
            x1, x2, y1, y2 = coordinate[0], coordinate[1], coordinate[2], coordinate[3]
            img = self.global_img[y1:y2,x1:x2,:]
            coordinate = torch.tensor([int(x1), int(x2), int(y1), int(y2)])
            img = transforms.ToTensor()(img)
            return img, coordinate

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
