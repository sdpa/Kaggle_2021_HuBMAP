from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import tifffile as tiff
from config import options
import torch
import os
import cv2
# import matplotlib.pyplot as plt


class HuBMAPDataset(Dataset):
    def __init__(self, base_dir, mode, patient=None):
        self.base_dir = base_dir
        self.mode = mode
        self.slice_indexes = []
        self.images = {}
        self.masks = {}  # The masks for 9 training patients.

        if mode == "train" or "val":
            train_patients = pd.read_csv(base_dir + "/train.csv")
            metadata = pd.read_csv(base_dir + "/HuBMAP-20-dataset_information.csv")
            if mode == "train":
                patients = train_patients.drop(options.test_tiff_value)
            else:
                patients = train_patients.iloc[options.test_tiff_value]

            for i in range(len(patients)):
                patientName = train_patients['id'].iloc[i]
                patient_metadata = metadata.loc[metadata["image_file"] == patientName + ".tiff"]
                height = patient_metadata["height_pixels"].iloc[0]
                width = patient_metadata["width_pixels"].iloc[0]
                self.masks[patientName] = self.rle_to_mask(train_patients['encoding'].iloc[i], (height, width, 1))
                self.images[patientName] = tiff.imread(self.base_dir + "/" + patientName + '.tiff')
                if len(self.images[patientName].shape) > 3:
                    self.images[patientName] = self.images[patientName].squeeze(0).squeeze(0)
                    self.images[patientName] = np.moveaxis(self.images[patientName], 0, -1)
                grid = self.make_grid((height, width), window=256)
                _id = [patientName]*grid.shape[0]
                _id = np.array(_id).reshape(grid.shape[0], 1)
                # Concatenate patient name
                grid = np.concatenate((grid, np.array(_id)), axis = 1)
                if len(self.slice_indexes) > 0:
                    self.slice_indexes = np.concatenate((self.slice_indexes, grid), axis = 0)
                else:
                    self.slice_indexes = grid

        elif mode == "test":
            self.imges[patient] = tiff.imread(self.base_dir + "/" + patient + '.tiff')
            if len(self.images[patient].shape) > 3:
                self.images[patient] = self.images[patient].squeeze(0).squeeze(0)
                self.images[patient] = np.moveaxis(self.images[patient], 0, -1)
            grid = self.make_grid((self.imges[patient].shape[0], self.imges[patient].shape[1]), window=256)
            _id = [patient] * grid.shape[0]
            _id = np.array(_id).reshape(grid.shape[0], 1)
            # Concatenate patient name
            grid = np.concatenate((grid, np.array(_id)), axis=1)
            if len(self.slice_indexes) > 0:
                self.slice_indexes = np.concatenate((self.slice_indexes, grid), axis=0)
            else:
                self.slice_indexes = grid
        # for i,slice in enumerate(self.slice_indexes):
        #     x1, x2, y1, y2, patient = slice
        #     if int(np.sum(self.masks[patient][int(x1):int(x2), int(y1):int(y2)])) == 0:
        #         self.slice_indexes = np.delete(self.slice_indexes, i,axis=0)


    def rle_to_mask(self, encoding, size):
        # encoding is a string. convert it to int.
        mask = np.zeros(size[0] * size[1], dtype=np.uint8)
        encoding = np.fromstring(encoding, dtype=int, sep=' ')
        starts = np.array(encoding)[::2]
        lengths = np.array(encoding)[1::2]
        for start, length in zip(starts, lengths):
            mask[start:start + length] = 1
        return mask.reshape(size, order='F')


    def make_grid(self, shape, window=260, min_overlap=32):
        """
            Return Array of size (N,4), where N - number of tiles,
            2nd axis represente slices: x1,x2,y1,y2
        """
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

    def __getitem__(self, index):
        if self.mode == "train" or self.mode == "val":
            x1, x2, y1, y2, patient = self.slice_indexes[index]
            slice_img = self.images[patient][int(y1):int(y2),int(x1):int(x2)]
            slice_mask = self.masks[patient][int(y1):int(y2),int(x1):int(x2)]
            return slice_img, slice_mask, (x1, x2, y1, y2)
        elif self.mode == "test":
            x1, x2, y1, y2, patient = self.slice_indexes[index]
            slice_img = self.images[patient][int(y1):int(y2), int(x1):int(x2)]
            return slice_img, (x1, x2, y1, y2)

    def __len__(self):
        return len(self.slice_indexes)
