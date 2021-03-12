import os
import json
import pprint
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from shapely.geometry import Polygon

#Get all train images.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

#Get train patients.
all_files = os.listdir(BASE_DIR + "/HuBMAP_Dataset/train")
train_names = [x[:-26] for x in all_files if '-structure.json' in x]
meta_data = pd.read_csv(BASE_DIR + "/HuBMAP_Dataset/HuBMAP-20-dataset_information.csv")
# print(train_names)
#
all_polygons = {}
print(train_names)

for i,name in enumerate(train_names):
    glom_data = None
    anatomical_data: None
    fig, ax = plt.subplots()
    with open(BASE_DIR + '/HuBMAP_Dataset/train/' + name + '.json') as f1, open(BASE_DIR + '/HuBMAP_Dataset/train/' + name + '-anatomical-structure'+ '.json') as f2:
        glom_data = json.load(f1)
        anatomical_data = json.load(f2)
        all_polygons[name] = {}
        patient_meta = meta_data[meta_data['image_file'] == name + '.tiff']
        rows = patient_meta['height_pixels'].iloc[0]
        cols = patient_meta['width_pixels'].iloc[0]
        print("=="*10)#
        print(name)
        ax.set(xlim=(0, cols), ylim=(rows, 0))
        for k in range(len(anatomical_data)):
            structure_name = anatomical_data[k]['properties']['classification']['name']
            if name == 'aaa6a05cc' and k == 1:
                coordinates = np.array(anatomical_data[k]['geometry']['coordinates'][0][0])
                coordinates = np.squeeze(coordinates)
                coordinates = coordinates.tolist()
                polygon_sh = Polygon(coordinates)
                all_polygons[name][structure_name] = coordinates[0]
                ax.plot(*polygon_sh.exterior.xy, color="red")
                print(structure_name)
            else:
                print(structure_name)
                coordinates = anatomical_data[k]['geometry']['coordinates'][0]
                if name == 'cb2d976f4' and k == 1:
                    structure_name = 'Cortex_2'
                all_polygons[name][structure_name] = coordinates[0]
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
            # print(structure_name)
        # Visualize gloms also.
        for g in range(len(glom_data)):
            glom_coords = glom_data[g]['geometry']['coordinates'][0]
            glom_sh = Polygon(glom_coords)
            ax.plot(*glom_sh.exterior.xy, color='green')
    # plt.axis('equal')
    # plt.title(name)
    # plt.show()
pp = pprint.PrettyPrinter()
pp.pprint(all_polygons)
# print(all_polygons)
