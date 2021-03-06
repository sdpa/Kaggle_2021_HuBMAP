from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import tifffile as tiff
from config import options

import os
import torch
from torchvision import transforms
import cv2
from PIL import Image

class HuBMAPCropDataset(Dataset):
    def __init__(self, base_dir, mode, patient=None):
        self.base_dir = base_dir
        self.mode = mode
        self.slice_indexes = []
        self.images = []
        self.masks = []  # The masks for 9 training patients.
        self.global_img = None

        if mode == "train":
            # Get all training patients.
            train_patients = pd.read_csv(base_dir + "/train.csv")
            leave_out_name = train_patients.iloc[options.test_tiff_value]['id']
            if mode == "train":
                images = os.listdir(self.base_dir + "/train/ImgCrops")
                #
                masks = os.listdir(self.base_dir + "/train/maskCrops")
                self.images = [x for x in images if leave_out_name not in x]
                self.masks = [x for x in masks if leave_out_name not in x]
                del images, masks
        elif mode == "val" or mode == "test":
            tiff_file = tiff.imread(self.base_dir + "/" + patient + '.tiff')
            #print("Tiff file shape: ", tiff_file.shape)
            if len(tiff_file.shape) == 3 and tiff_file.shape[0] == 3:
                tiff_file = np.moveaxis(tiff_file, 0, -1)
            if len(tiff_file.shape) > 3:
                tiff_file = tiff_file.squeeze(0).squeeze(0)
                tiff_file = np.moveaxis(tiff_file, 0, -1)
            if mode == "val":
                grid = self.make_grid((tiff_file.shape[0], tiff_file.shape[1]), window=options.test_window)
            elif mode == "test":
                grid = self.make_grid((tiff_file.shape[0], tiff_file.shape[1]), window=options.val_window)
            if len(self.slice_indexes) > 0:
                self.slice_indexes = np.concatenate((self.slice_indexes, grid), axis=0)
            else:
                self.slice_indexes = grid
            # print("Fixed file size: ", tiff_file.shape)
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
            img = Image.open(self.base_dir + "/train/ImgCrops/" + file_name)
            mask = Image.open(self.base_dir + "/train/maskCrops/" + file_name)

            img = transforms.ColorJitter(brightness=0.5, contrast=0.5, hue=.05, saturation=.05).forward(img)
            img = transforms.GaussianBlur(kernel_size=(3,3)).forward(img)

            img = transforms.ToTensor()(img)
            mask = transforms.ToTensor()(mask)

            merged = torch.cat((img, mask), 0)

            crop = transforms.RandomCrop((options.train_window, options.train_window), pad_if_needed=True).forward(merged)
            crop = transforms.RandomHorizontalFlip().forward(crop)
            crop = transforms.RandomVerticalFlip().forward(crop)
            crop = transforms.RandomRotation(90).forward(crop)

            # # Split them back.
            img = crop[:3, :, :]
            mask = crop[-1:, :, :]

            mask = mask * 255

            return img, mask
        elif self.mode == "val" or self.mode == "test":
            coordinate = self.slice_indexes[index]
            x1, x2, y1, y2 = coordinate[0], coordinate[1], coordinate[2], coordinate[3]
            img = self.global_img[y1:y2, x1:x2, :]
            coordinate = torch.tensor([int(x1), int(x2), int(y1), int(y2)])
            img = transforms.ToTensor()(img)
            return img, coordinate

    def __len__(self):
        if self.mode == "train":
            return len(self.images)
        elif self.mode == "val" or self.mode == "test":
            return len(self.slice_indexes)
