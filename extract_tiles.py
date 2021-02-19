import pandas as pd
import json
import numpy as np
import os
import sys
import tifffile as tiff
import matplotlib.pyplot as plt
from PIL import Image
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
    json_data = None
    with open(BASE_DIR + '/trainData/' + name + '.json') as f:
        json_data = json.load(f)

    for j in range(len(json_data)):
        polygon = json_data[j]['geometry']['coordinates'][0]
        x1 = sys.maxsize
        y1 = sys.maxsize
        x2 = 0
        y2 = 0
        shape = (0, 0)
        # print(polygon)
        for point in polygon:
            if point[0] < x1:
                x1 = point[0]
            if point[0] > x2:
                x2 = point[0]
            if point[1] > y2:
                y2 = point[1]
            if point[1] < y1:
                y1 = point[1]
        shape = (y2 - y1, x2 - x1)

        if shape[0] > largest[0]:
            largest[0] = shape[0]
        if shape[1] > largest[1]:
            largest[1] = shape[1]

        verticalPadding = (1024 - shape[0])
        horizontalPadding = (1024 - shape[1])

        # Create necessary padding in all 4 sides
        if verticalPadding % 2 != 0:
            paddingTop = verticalPadding * random.random()
            paddingBottom = verticalPadding - paddingTop
        else:
            paddingTop, paddingBottom = verticalPadding // 2, verticalPadding // 2

        if horizontalPadding % 2 != 0:
            paddingLeft = horizontalPadding * random.random()
            paddingRight = horizontalPadding - paddingLeft
        else:
            paddingLeft, paddingRight = horizontalPadding // 2, horizontalPadding // 2

        new_y1 = y1 - paddingTop
        new_y2 = y2 + paddingBottom

        new_x1 = x1 - paddingLeft
        new_x2 = x2 + paddingRight

        img_crop = img[new_y1:new_y2, new_x1:new_x2, :]
        mask_crop = global_mask[new_y1:new_y2, new_x1:new_x2]

        im = Image.fromarray(img_crop)
        im.save(BASE_DIR + "/ImgCrops/{}_{}_{}.png".format(name, new_x1, new_y1))

        im = Image.fromarray(mask_crop)
        im.save(BASE_DIR + "/maskCrops/{}_{}_{}.png".format(name, new_x1, new_y1))


images_count = len(os.listdir(BASE_DIR+"/ImgCrops"))
mask_count = len(os.listdir(BASE_DIR+"/maskCrops"))
print('Images count: ', images_count)
print('Masks count: ', mask_count)
print('Done padding')
