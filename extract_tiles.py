import pandas as pd
import json
import numpy as np
import os
import sys
import tifffile as tiff
import matplotlib.pyplot as plt
from PIL import Image
import random
import pprint
import cv2
import math
from shapely.geometry import Polygon
from utils import rle_to_mask

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
train_patients = pd.read_csv(BASE_DIR + "/HuBMAP_Dataset/train.csv")
train_patient_names = train_patients['id']
metadata_all = pd.read_csv(BASE_DIR + "/HuBMAP_Dataset/HuBMAP-20-dataset_information.csv")

medulla_percent = 0.45
cortex_no_glom_percent = 0.45
background_percent  = 0.10

largest = [0, 0]
all_polygons = {}
for i, name in enumerate(train_patient_names):
    img = tiff.imread(BASE_DIR + "/HuBMAP_Dataset/train/" + name + ".tiff")
    patient_meta =  metadata_all[metadata_all['image_file'] == name + '.tiff']
    rows = patient_meta['height_pixels'].iloc[0]
    cols = patient_meta['width_pixels'].iloc[0]
    print(rows, cols)
    # # Removes extra dimensions in the image.
    if len(img.shape) > 3:
        img = img.squeeze(0).squeeze(0)
        img = np.moveaxis(img, 0, -1)
    print('Tiff file shape: ', img.shape)

    # Get encoding from dataframe to generate the global mask.
    encoding = train_patients.loc[train_patients["id"] == name]['encoding'].iloc[0]
    global_mask = rle_to_mask(encoding, (rows, cols))

    print("Mask shape: ", global_mask.shape)
    glom_data = None
    anatomical_data = None
    with open(BASE_DIR + '/HuBMAP_Dataset/train/' + name + '.json') as f1, open(BASE_DIR + '/HuBMAP_Dataset/train/' + name + '-anatomical-structure'+ '.json') as f2:
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

    # create crops of all gloms of size 1024x1024
    while glom_created < max_glom_count:
        polygon = glom_data[glom_created]['geometry']['coordinates'][0]
        x1,y1 = sys.maxsize, sys.maxsize
        x2,y2 = 0,0
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

        new_y1 = int(y1 - paddingTop)
        new_y2 = int(y2 + paddingBottom)

        new_x1 = int(x1 - paddingLeft)
        new_x2 = int(x2 + paddingRight)

        mask_crop = global_mask[new_y1:new_y2, new_x1:new_x2]
        save_name = name + '_glom_' + str(new_y1) + '_' + str(new_x1)
        img_crop = img[new_y1:new_y2, new_x1:new_x2, :]
        im = Image.fromarray(img_crop)
        im.save(BASE_DIR + "/HuBMAP_Dataset/train/ImgCrops/{}.png".format(save_name))

        im = Image.fromarray(mask_crop)
        im.save(BASE_DIR + "/HuBMAP_Dataset/train/maskCrops/{}.png".format(save_name))

        glm_pts = [[new_x1, new_y1], [new_x1, new_y2], [new_x2, new_y2], [new_x2, new_y1]]
        glom_sh = Polygon(glm_pts)
        ax.plot(*glom_sh.exterior.xy, color="green")
        glom_created += 1

    medulla_created, cortex_no_glom_created, background_created = 0, 0, 0,

    all_polygons[name] = {}
    patient_meta = metadata_all[metadata_all['image_file'] == name + '.tiff']
    rows = patient_meta['height_pixels'].iloc[0]
    cols = patient_meta['width_pixels'].iloc[0]
    print("=="*10)
    print(name)
    ax.set(xlim=(0, cols), ylim=(rows, 0))

    #Reads json data and creates dictionary of anatomical structures along with their coordinates
    for k in range(len(anatomical_data)):
        structure_name = anatomical_data[k]['properties']['classification']['name']
        if name == 'aaa6a05cc' and k == 1:
            coordinates = np.array(anatomical_data[k]['geometry']['coordinates'][0][0])
            coordinates = np.squeeze(coordinates)
            coordinates = coordinates.tolist()
            polygon_sh = Polygon(coordinates)
            all_polygons[name][structure_name] = coordinates
            ax.plot(*polygon_sh.exterior.xy, color="red")
            print(structure_name)
        else:
            coordinates = anatomical_data[k]['geometry']['coordinates'][0]
            if (name == 'cb2d976f4' or name == 'b2dc8411c') and k == 1:
                structure_name = 'Cortex_2'
            print(structure_name)
            all_polygons[name][structure_name] = coordinates
            if structure_name == 'Cortex' or structure_name == 'Cortex_2':
                color = 'red'
            elif structure_name == 'Medulla':
                color = 'orange'
            elif structure_name == 'Outer Stripe':
                color = 'yellow'
            elif structure_name == 'Inner medulla':
                color = 'cyan'
            elif structure_name == 'Outer Medulla':
                color = 'magenta'
            polygon_sh = Polygon(coordinates)
            ax.plot(*polygon_sh.exterior.xy, color=color)

    # makes Polygon objects out of all the structural features.
    if len(all_polygons[name].keys()) == 4:
        cortex_sh = Polygon(all_polygons[name]['Cortex'])
        outer_stripe_sh = Polygon(all_polygons[name]['Outer Stripe'])
        inner_medulla_sh = Polygon(all_polygons[name]['Inner medulla'])
        outer_medulla_sh = Polygon(all_polygons[name]['Outer Medulla'])
    elif len(all_polygons[name].keys()) == 3:
        cortex_sh = Polygon(all_polygons[name]['Cortex'])
        cortex_2_sh = Polygon(all_polygons[name]['Cortex_2'])
        medulla_sh = Polygon(all_polygons[name]['Medulla'])
    elif len(all_polygons[name].keys()) == 2:
        # print(all_polygons[name]['Cortex'])
        cortex_sh = Polygon(all_polygons[name]['Cortex'])
        medulla_sh = Polygon(all_polygons[name]['Medulla'])
    else:
        cortex_sh = Polygon(all_polygons[name]['Cortex'])
        max_medulla = 0
    img_sh = Polygon([[0, 0], [cols,0], [cols,rows], [0,rows]])

    print(all_polygons[name].keys())

    #Creates crops based on the structural features.
    while cortex_no_glom_created < max_cortex_no_glom or background_created < max_background or medulla_created < max_medulla:
        x, y = random.randint(0, cols-1024), random.randint(0, rows-1024)
        corners = [[x, y], [x + 1024, y], [x + 1024, y + 1024], [x, y + 1024]]
        box = Polygon(corners)
        exists = False

        # Check where the box is located.
        no_glom_exists, medulla_exists, bg_exists, = False, False, False
        if len(all_polygons[name]) == 4:
            if cortex_sh.contains(box) and cortex_no_glom_created < max_cortex_no_glom:
                no_glom_exists = True
            elif inner_medulla_sh.contains(box) or outer_medulla_sh.contains(box) and medulla_created < max_medulla:
                medulla_exists = True
            elif img_sh.contains(box) and background_created < max_background:
                bg_exists = True
        elif len(all_polygons[name]) == 3:
            if cortex_sh.contains(box) or cortex_2_sh.contains(box) and cortex_no_glom_created < max_cortex_no_glom:
                no_glom_exists = True
            elif medulla_sh.contains(box) and medulla_created < max_medulla:
                medulla_exists = True
            elif img_sh.contains(box) and background_created < max_background:
                bg_exists = True
        elif len(all_polygons[name]) == 2:
            if cortex_sh.contains(box) and cortex_no_glom_created < max_cortex_no_glom:
                no_glom_exists = True
            elif medulla_sh.contains(box) and medulla_created < max_medulla:
                medulla_exists = True
            elif img_sh.contains(box) and background_created < max_background:
                bg_exists = True
        else:
            if cortex_sh.contains(box) and cortex_no_glom_created < max_cortex_no_glom:
                no_glom_exists = True
            elif img_sh.contains(box) and background_created < max_background:
                bg_exists = True


        if no_glom_exists or medulla_exists or bg_exists:
            # Offset y points back to top-left origin config to get mask.
            x1, y1 = corners[0][0], corners[0][1]
            # y1 = rows - y1
            x2, y2 = x1 + 1024, y1 + 1024
            mask_crop = global_mask[y1:y2, x1:x2]
            if no_glom_exists and np.sum(mask_crop) <= 0:
                cortex_no_glom_created += 1
                ax.plot(*box.exterior.xy, color='red')
                save_name = name + '_noglom_' + str(y1) + '_' + str(x1)
                img_crop = img[y1:y2, x1:x2, :]
                im = Image.fromarray(img_crop)
                im.save(BASE_DIR + "/HUBMAP_Dataset/train/ImgCrops/{}.png".format(save_name))
                im = Image.fromarray(mask_crop)
                im.save(BASE_DIR + "/HUBMAP_Dataset/train/maskCrops/{}.png".format(save_name))
            elif medulla_exists:
                medulla_created += 1
                ax.plot(*box.exterior.xy, color='orange')
                save_name = name + '_medulla_' + str(y1) + '_' + str(x1)
                img_crop = img[y1:y2, x1:x2, :]
                im = Image.fromarray(img_crop)
                im.save(BASE_DIR + "/HuBMAP_Dataset/train/ImgCrops/{}.png".format(save_name))

                im = Image.fromarray(mask_crop)
                im.save(BASE_DIR + "/HuBMAP_Dataset/train/maskCrops/{}.png".format(save_name))
            elif bg_exists:
                background_created += 1
                ax.plot(*box.exterior.xy, color='blue')
                save_name = name + '_bg_' + str(y1) + '_' + str(x1)
                img_crop = img[y1:y2, x1:x2, :]
                im = Image.fromarray(img_crop)
                im.save(BASE_DIR + "/HuBMAP_Dataset/train/ImgCrops/{}.png".format(save_name))

                im = Image.fromarray(mask_crop)
                im.save(BASE_DIR + "/HuBMAP_Dataset/train/maskCrops/{}.png".format(save_name))
    plt.axis('equal')
    plt.title(name)
    plt.show()

    print("glom: ", glom_created)
    print("No gloms cortex: ", cortex_no_glom_created)
    print("medulla: ", medulla_created)
    print("background: ", background_created)
    print("**"*20)
