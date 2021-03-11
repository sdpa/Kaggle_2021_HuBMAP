from shapely.geometry import Point, Polygon
import matplotlib.pyplot as plt
import random
# points = [[12791, 2451], [11717, 2695], [10985, 3281], [10106, 4550], [10058, 4697], [10009, 4794], [9960, 4794], [9179, 8309], [9130, 10262], [10497, 9578], [12986, 8748], [16110, 9334], [16891, 10750], [16843, 11970], [17038, 14704], [17673, 14606], [17673, 14655], [17673, 15631], [16599, 16266], [14548, 18706], [11327, 22026], [10253, 23295], [10253, 23344], [10204, 23344], [10106, 23978], [10497, 24759], [10497, 24808], [10546, 24857], [10594, 24857], [11522, 25345], [11571, 25345], [12596, 24271], [15964, 28762], [16110, 28762], [16159, 28860], [16208, 28909], [17770, 29397], [18551, 29446], [19283, 28616], [19332, 28567], [19381, 28567], [19918, 26175], [19967, 26029], [19967, 25980], [19967, 25931], [20699, 23930], [20699, 23881], [20748, 23881], [22163, 21050], [22212, 20952], [22505, 19390], [22505, 17437], [22505, 17389], [22456, 15631], [22554, 14753], [22407, 12946], [21871, 10945], [21382, 9188], [21041, 7577], [18161, 4746], [15427, 3330], [13963, 2793], [12791, 2451]]
#
# #
# # for point in points:
# #     point[1] = 31278 - point[1]
#
# ys = [point[1] for point in points]
# xs = [point[0] for point in points]
#
# # make a shape.
# cortex = Polygon(points)
# img = Polygon([[0,0],[25794,0],[25794,31278],[0,31278]])
# fig2, ax2 = plt.subplots()
# ax2.set(xlim=(0,25794), ylim=(0,31278))
# ax2.plot(*img.exterior.xy)
# ax2.plot(*cortex.exterior.xy)

# cortex_count, bg_count = 0, 0
# max_cortex_count, max_bg_count = 10, 10
# while cortex_count < max_cortex_count or bg_count < max_bg_count:
#     x,y = random.randint(0,25754), random.randint(0,31278)
#     box = Polygon([[x,y],[x+1024,y],[x+1024,y+1024], [x,y+1024]])
#     if cortex.contains(box):
#         cortex_count += 1
#         ax2.plot(*box.exterior.xy, color='green')
#     if not cortex.contains(box) and img.contains(box):
#         bg_count += 1
#         ax2.plot(*box.exterior.xy, color='blue')
#     # if not cortex.contains(box) and
# plt.axis('equal')
# plt.show()

import os
from PIL import Image
import numpy
import tifffile as tiff
Image.MAX_IMAGE_PIXELS = 17895697000

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

print('Train files')
all_files = os.listdir(BASE_DIR + "/HuBMAP_Dataset/train")
patient_files = [x for x in all_files if '.tiff' in x]
# patient_files = ['095bf7a1f.tiff']
# patient_files = ['4ef6695ce.tiff']

for patient_file in patient_files:
    name = patient_file[:-5]
    # print(name)
    tiff_file = tiff.imread(BASE_DIR + "/HuBMAP_Dataset/train/" + patient_file)
    # imarray = numpy.array(tiff_file)
    print("{} : {}".format(name, tiff_file.shape))

print('**'*20)
print('Test files')
all_files = os.listdir(BASE_DIR + "/HuBMAP_Dataset/test")
patient_files = [x for x in all_files if '.tiff' in x]

for patient_file in patient_files:
    name = patient_file[:-5]
    # print(name)
    tiff_file = tiff.imread(BASE_DIR + "/HuBMAP_Dataset/test/" + patient_file)
    # imarray = numpy.array(tiff_file)
    print("{} : {}".format(name, tiff_file.shape))

