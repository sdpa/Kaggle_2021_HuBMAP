import pandas as pd
import json
import numpy as np
import os
import sys
import tifffile as tiff
import matplotlib.pyplot as plt
from PIL import Image
import random
import cv2
import math

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
train_patients = pd.read_csv(BASE_DIR + "/trainData/train.csv")
train_patient_names = train_patients['id']
metadata_all = pd.read_csv(BASE_DIR + "/trainData/HuBMAP-20-dataset_information.csv")


def rle_to_mask(_encoding, size):
    # encoding is a string. convert it to int.
    mask = np.zeros(size[0] * size[1], dtype=np.uint8)
    encoding_np = np.fromstring(_encoding, dtype=int, sep=' ')
    starts = np.array(encoding_np)[::2]
    lengths = np.array(encoding_np)[1::2]
    for start, length in zip(starts, lengths):
        mask[start:start + length] = 1
    return mask.reshape(size, order='F')


largest = [0, 0]
for i, name in enumerate(train_patient_names):
    print(name)
    img = tiff.imread(BASE_DIR + "/trainData/" + name + ".tiff")

    # Removes extra dimensions in the image.
    if len(img.shape) > 3:
        img = img.squeeze(0).squeeze(0)
        img = np.moveaxis(img, 0, -1)
    print('Tiff file shape: ', img.shape)

    # Get encoding from dataframe to generate the global mask.
    encoding = train_patients.loc[train_patients["id"] == name]['encoding'].iloc[0]
    global_mask = rle_to_mask(encoding, (img.shape[0], img.shape[1]))
    print("Mask shape: ", global_mask.shape)

    x = img.shape[1] - 1024
    y = img.shape[0] - 1024

    length = len(os.listdir(BASE_DIR + '/trainData/ImgCrops'))//len(train_patient_names)

    for j in range(length):
        x1 = random.randint(0,x)
        x2 = x1 + 1024

        y1 = random.randint(0,y)
        y2 = y1 + 1024

        img_crop = img[y1:y2, x1:x2, :]
        mask_crop = global_mask[y1:y2, x1:x2]

        # im = Image.fromarray(img_crop)
        # im.save(BASE_DIR + "/trainData/ImgCrops/{}_{}_{}.png".format(name, x1, y1))
        #
        # im = Image.fromarray(mask_crop)
        # im.save(BASE_DIR + "/trainData/maskCrops/{}_{}_{}.png".format(name, x1, y1))

    break

print('Success!!')

