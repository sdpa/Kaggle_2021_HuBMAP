import pandas as pd
import json
import numpy as np
# import os
import sys
import tifffile as tiff
import matplotlib.pyplot as plt
from PIL import Image
import random
from shapely.geometry import Polygon
from utils.rle_to_mask import rle_to_mask
from utils.config import options

# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
train_patients = pd.read_csv(options.kaggle_data_path + "/tiffs/trainData/train.csv")
train_patient_names = train_patients['id']
metadata_all = pd.read_csv(options.kaggle_data_path + "/tiffs/HuBMAP-20-dataset_information.csv")

medulla_percent = 0.45
cortex_no_glom_percent = 0.45
background_percent = 0.10

largest = [0, 0]
for i, name in enumerate(train_patient_names):
    print(name)
    img = tiff.imread(options.kaggle_data_path + "/tiffs/trainData/" + name + ".tiff")
    patient_meta = metadata_all[metadata_all['image_file'] == name + '.tiff']
    rows = patient_meta['height_pixels'].iloc[0]
    cols = patient_meta['width_pixels'].iloc[0]
    print(rows, cols)
    # # Removes extra dimensions in the image.
    if len(img.shape) > 3:
        img = img.squeeze(0).squeeze(0)
        img = np.moveaxis(img, 0, -1)
    # print('Tiff file shape: ', img.shape)

    # Get encoding from dataframe to generate the global mask.
    encoding = train_patients.loc[train_patients["id"] == name]['encoding'].iloc[0]
    global_mask = rle_to_mask(encoding, (rows, cols))

    print("Mask shape: ", global_mask.shape)
    with open(options.kaggle_data_path + '/tiffs/trainData/' + name + '.json') as \
            f1, open(options.kaggle_data_path + '/tiffs/trainData/' + name + '-anatomical-structure.json') as f2:
        glom_data = json.load(f1)
        anatomical_data = json.load(f2)

    max_glom_count = len(glom_data)
    max_medulla = round(medulla_percent * max_glom_count)
    max_cortex_no_glom = round(cortex_no_glom_percent * max_glom_count)
    max_background = round(background_percent * max_glom_count)
    total = max_glom_count * 2

    glom_created = 0

    fig, ax = plt.subplots()
    ax.set(xlim=(0, cols), ylim=(0, rows))

    # create crops of gloms.
    while glom_created < max_glom_count:
        polygon = glom_data[glom_created]['geometry']['coordinates'][0]
        x1, y1 = sys.maxsize, sys.maxsize
        x2, y2 = 0, 0
        for point in polygon:
            x = point[0]
            y = point[1]
            if y <= y1:
                y1 = y
            if x <= x1:
                x1 = x
            if y > y2:
                y2 = y
            if x > x2:
                x2 = x
        shape = (y2 - y1, x2 - x1)

        if shape[0] > largest[0]:
            largest[0] = shape[0]
        if shape[1] > largest[1]:
            largest[1] = shape[1]

        verticalPadding = (1024 - shape[0])
        horizontalPadding = (1024 - shape[1])

        # Create necessary padding in all 4 sides
        if verticalPadding % 2 != 0:
            paddingTop = int(verticalPadding * random.random())
            paddingBottom = verticalPadding - paddingTop
        else:
            paddingTop, paddingBottom = verticalPadding // 2, verticalPadding // 2

        if horizontalPadding % 2 != 0:
            paddingLeft = int(horizontalPadding * random.random())
            paddingRight = horizontalPadding - paddingLeft
        else:
            paddingLeft, paddingRight = horizontalPadding // 2, horizontalPadding // 2

        new_y1 = y1 - paddingTop
        new_y2 = y2 + paddingBottom

        new_x1 = x1 - paddingLeft
        new_x2 = x2 + paddingRight
        mask_crop = global_mask[new_y1:new_y2, new_x1:new_x2]
        save_name = name + '_glom_' + str(new_y1) + '_' + str(new_x1)
        img_crop = img[new_y1:new_y2, new_x1:new_x2, :]
        im = Image.fromarray(img_crop)
        im.save(options.kaggle_data_path + "/improved_crops/ImgCrops/{}.png".format(save_name))

        im = Image.fromarray(mask_crop)
        im.save(options.kaggle_data_path + "/improved_crops/maskCrops/{}.png".format(save_name))

        # offset to plot on matplotlib
        new_y1 = rows - new_y1
        new_y2 = rows - new_y2

        glm_pts = [[new_x1, new_y1], [new_x1, new_y2], [new_x2, new_y2], [new_x2, new_y1]]
        glom_sh = Polygon(glm_pts)
        ax.plot(*glom_sh.exterior.xy, color="red")
        glom_created += 1

    # create crops from cortex, medulla and background.
    # There is medulla,cortex or cortex, medulla
    cortex_polygon, cortex_polygon_2, medulla_polygon, = [], [], []

    # Get only cortex and background
    if len(anatomical_data) == 1:
        cortex_polygon = anatomical_data[0]['geometry']['coordinates'][0]
        max_cortex_no_glom = max_cortex_no_glom + max_medulla
        max_medulla = 0
    elif len(anatomical_data) == 2:
        for k in range(len(anatomical_data)):
            if anatomical_data[k]['properties']['classification']['name'] == 'Cortex':
                cortex_polygon = np.array(anatomical_data[k]['geometry']['coordinates'][0])
                cortex_polygon = np.squeeze(cortex_polygon)
                cortex_polygon = cortex_polygon.tolist()
            else:
                medulla_polygon = anatomical_data[k]['geometry']['coordinates'][0]
    elif len(anatomical_data) == 3:
        cortex_polygon = anatomical_data[0]['geometry']['coordinates'][0]
        cortex_polygon_2 = anatomical_data[1]['geometry']['coordinates'][0]
        medulla_polygon = anatomical_data[2]['geometry']['coordinates'][0]

    medulla_created, cortex_no_glom_created, background_created = 0, 0, 0,

    # offset cortex and medulla.
    for point in cortex_polygon:
        point[1] = rows - point[1]
    for point in medulla_polygon:
        point[1] = rows - point[1]
    for point in cortex_polygon_2:
        point[1] = rows - point[1]

    # create shapely shapes for cortex, medulla and image.
    cortex_sh = Polygon(cortex_polygon)
    medulla_sh = Polygon(medulla_polygon)
    img_sh = Polygon([[0, 0],
                      [cols, 0],
                      [cols, rows],
                      [0, rows]])
    cortex_2_sh = None
    if len(cortex_polygon_2) != 0:
        cortex_2_sh = Polygon(cortex_polygon_2)
        ax.plot(*cortex_2_sh.exterior.xy, color="green")

    ax.plot(*img_sh.exterior.xy, color="blue")

    if len(medulla_polygon) != 0:
        ax.plot(*medulla_sh.exterior.xy, color="orange")
    ax.plot(*cortex_sh.exterior.xy, color="green")

    while cortex_no_glom_created < max_cortex_no_glom or background_created < max_background \
            or medulla_created < max_medulla:
        x, y = random.randint(0, cols-1024), random.randint(0, rows-1024)
        corners = [[x, y], [x + 1024, y], [x + 1024, y + 1024], [x, y + 1024]]
        box = Polygon(corners)
        exists = False
        if len(anatomical_data) == 3:
            if (cortex_sh.contains(box) or cortex_2_sh.contains(box)) and cortex_no_glom_created < max_cortex_no_glom:
                exists = True
        elif cortex_sh.contains(box) and cortex_no_glom_created < max_cortex_no_glom:
            exists = True
        if exists:
            # Offset y points back to top-left origin config to get mask.
            x1, y1 = corners[1][0], corners[1][1]
            y1 = rows - y1
            x2, y2 = x1 + 1024, y1 + 1024
            mask_crop = global_mask[y1:y2, x1:x2]
            if np.sum(mask_crop) <= 0:
                cortex_no_glom_created += 1
                ax.plot(*box.exterior.xy, color='green')
                save_name = name + '_noglom_' + str(y1) + '_' + str(x1)
                img_crop = img[y1:y2, x1:x2, :]
                im = Image.fromarray(img_crop)
                im.save(options.kaggle_data_path + "/improved_crops/ImgCrops/{}.png".format(save_name))

                im = Image.fromarray(mask_crop)
                im.save(options.kaggle_data_path + "/improved_crops/maskCrops/{}.png".format(save_name))

        if medulla_sh.contains(box) and medulla_created < max_medulla:
            medulla_created += 1
            ax.plot(*box.exterior.xy, color='orange')
            x1, y1 = corners[1][0], corners[1][1]
            y1 = rows - y1
            x2, y2 = x1 + 1024, y1 + 1024
            mask_crop = global_mask[y1:y2, x1:x2]
            save_name = name + '_medulla_' + str(y1) + '_' + str(x1)
            img_crop = img[y1:y2, x1:x2, :]
            im = Image.fromarray(img_crop)
            im.save(options.kaggle_data_path + "/improved_crops/ImgCrops/{}.png".format(save_name))
            im = Image.fromarray(mask_crop)
            im.save(options.kaggle_data_path + "/improved_crops/maskCrops/{}.png".format(save_name))
        if not cortex_sh.contains(box) and img_sh.contains(box) and not medulla_sh.contains(box) \
                and background_created < max_background:
            background_created += 1
            ax.plot(*box.exterior.xy, color='blue')
            x1, y1 = corners[1][0], corners[1][1]
            y1 = rows - y1
            x2, y2 = x1 + 1024, y1 + 1024
            mask_crop = global_mask[y1:y2, x1:x2]
            save_name = name + '_bg_' + str(y1) + '_' + str(x1)
            img_crop = img[y1:y2, x1:x2, :]
            im = Image.fromarray(img_crop)
            im.save(options.kaggle_data_path + "/improved_crops/ImgCrops/{}.png".format(save_name))
            im = Image.fromarray(mask_crop)
            im.save(options.kaggle_data_path + "/improved_crops/maskCrops/{}.png".format(save_name))
    plt.axis('equal')
    plt.title(name)
    plt.show()

    print("glom: ", glom_created)
    print("No gloms cortex: ", cortex_no_glom_created)
    print("medulla: ", medulla_created)
    print("background: ", background_created)
    print("**"*20)
