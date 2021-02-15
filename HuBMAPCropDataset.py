from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import tifffile as tiff
from config import options
from PIL import Image
import os
import torch
from torchvision import transforms
from numpy import asarray

class HuBMAPCropDataset(Dataset):
    def __init__(self, base_dir, mode, patient=None):
        self.base_dir = base_dir
        self.mode = mode
        self.slice_indexes = []
        self.images = []
        self.masks = []  # The masks for 9 training patients.
        self.global_img = None

        if mode == "train" or mode == "val":
            # Get all training patients.
            train_patients = pd.read_csv(base_dir + "/train.csv")
            leave_out_name = train_patients.iloc[options.test_tiff_value]['id']
            if mode == "train":
                images = os.listdir(self.base_dir + "/ImgCrops")
                masks = os.listdir(self.base_dir + "/maskCrops")
                self.images = [x for x in images if leave_out_name not in x]
                self.masks = [x for x in masks if leave_out_name not in x]
                del images, masks
            elif mode == "val":
                images = os.listdir(self.base_dir + "/ImgCrops")
                masks = os.listdir(self.base_dir + "/maskCrops")
                self.images = [x for x in images if leave_out_name in x]
                self.masks = [x for x in masks if leave_out_name in x]
                del images, masks
        elif mode == "test":
            tiff_file = tiff.imread(self.base_dir + "/" + patient + '.tiff')
            if len(tiff_file.shape) > 3:
                tiff_file = tiff_file.squeeze(0).squeeze(0)
                tiff_file = np.moveaxis(tiff_file, 0, -1)
            grid = self.make_grid((tiff_file.shape[0], tiff_file.shape[1]), window=512)
            if len(self.slice_indexes) > 0:
                self.slice_indexes = np.concatenate((self.slice_indexes, grid), axis=0)
            else:
                self.slice_indexes = grid
            self.global_img = tiff_file

    def make_grid(self, shape, window=512, min_overlap=0):
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
        if self.mode == "train" or self.mode == "val":
            file_name = self.images[index]
            img = Image.open(self.base_dir + "/ImgCrops/" + file_name)
            mask = Image.open(self.base_dir + "/maskCrops/" + file_name)

            img = transforms.ToTensor()(img)
            mask = transforms.ToTensor()(mask)
            mask = mask * 255

            img = transforms.Resize((512, 512), Image.BILINEAR)(img)
            mask = transforms.Resize((512, 512), Image.BILINEAR)(mask)

            return img, mask
        elif self.mode == "test":
            coordinate = self.slice_indexes[index]
            x1, x2, y1, y2 = coordinate[0], coordinate[1], coordinate[2], coordinate[3]
            img = self.global_img[y1:y2,x1:x2,:]
            coordinate = torch.tensor([x1, x2, y1, y2])
            img = transforms.ToTensor()(img)
            return img, coordinate

    def __len__(self):
        if self.mode == "train" or self.mode == 'val':
            return len(self.images)
        elif self.mode == "test":
            return len(self.slice_indexes)
