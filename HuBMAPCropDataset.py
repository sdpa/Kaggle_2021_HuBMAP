from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import tifffile as tiff
from config import options
from PIL import Image
import os
from torchvision import transforms
from numpy import asarray

class HuBMAPCropDataset(Dataset):
    def __init__(self, base_dir, mode, patient=None):
        self.base_dir = base_dir
        self.mode = mode
        self.slice_indexes = []
        self.images = []
        self.masks = []  # The masks for 9 training patients.

        # Get all patients masks.
        train_patients = pd.read_csv(base_dir + "/train.csv")
        metadata = pd.read_csv(base_dir + "/HuBMAP-20-dataset_information.csv")
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
            self.images, self.masks = {}, {}
            self.images[patient] = tiff.imread(self.base_dir + "/" + patient + '.tiff')
            if len(self.images[patient].shape) > 3:
                self.images[patient] = self.images[patient].squeeze(0).squeeze(0)
                self.images[patient] = np.moveaxis(self.images[patient], 0, -1)
            grid = self.make_grid((self.images[patient].shape[0], self.images[patient].shape[1]), window=1024)
            _id = [patient] * grid.shape[0]
            _id = np.array(_id).reshape(grid.shape[0], 1)
            # Concatenate patient name
            grid = np.concatenate((grid, np.array(_id)), axis=1)
            if len(self.slice_indexes) > 0:
                self.slice_indexes = np.concatenate((self.slice_indexes, grid), axis=0)
            else:
                self.slice_indexes = grid

    def make_grid(self, shape, window=1024, min_overlap=0):
        x, y = shape
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
            x1, x2, y1, y2, patient = self.slice_indexes[index]
            slice_img = self.images[patient][int(y1):int(y2), int(x1):int(x2)]
            return slice_img, (x1, x2, y1, y2)

    def __len__(self):
        if self.mode == "train" or self.mode == 'val':
            return len(self.images)
        elif self.mode == "test":
            return len(self.slice_indexes)
