#Import models and run tests.
import os
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import numpy as np
from config import *
from train import global_shift_mask, rle_encode_less_memory
from HuBMAPCropDataset import HuBMAPCropDataset
from torch.utils.data import DataLoader
from PIL import Image
import pandas as pd
from models.EfficientUnet.efficient_unet import *

os.environ['CUDA_VISIBLE_DEVICES'] = '2'

#Get best model and run test.

def predict():
    subm = {}
    all_files = os.listdir(BASE_DIR + "/testData")
    patient_files = [x for x in all_files if '.tiff' in x]
    model.eval()
    for patient_file in patient_files:
        name = patient_file[:-5]
        print('--' * 40)
        print("Predicting for patient: {}".format(name))
        test_dataset = HuBMAPCropDataset(BASE_DIR + "/testData", mode='test', patient=name)
        test_loader = DataLoader(test_dataset, batch_size=4,
                                 shuffle=False, num_workers=options.workers, drop_last=False)
        height, width = test_dataset.get_global_image_size()
        global_mask = torch.zeros((height, width), dtype=torch.int8)
        #global_mask = global_mask.to(device, dtype=torch.int8)
        with torch.no_grad():
            for i, data in enumerate(test_loader):
                img_batch, coordinates_batch = data
                img_batch = img_batch.to(device, dtype=torch.float)
                coordinates_batch = coordinates_batch.to(device, dtype=torch.float)
                pred_mask_batch = model(img_batch)

                # Converts mask to 0/1.
                pred_mask_batch = (pred_mask_batch > options.threshold).type(torch.int8)
                pred_mask_batch = pred_mask_batch * 255

                # Loop through each img,mask in batch.
                for each_mask, coordinate in zip(pred_mask_batch, coordinates_batch):
                    each_mask = torch.squeeze(each_mask)
                    # xs = columns, ys = rows. (x1,y1) --> Top Left. (x2,y2) --> bottom right.
                    x1, x2, y1, y2 = coordinate
                    global_mask[int(y1):int(y2),int(x1):int(x2)] = each_mask
        global_mask = global_mask.numpy()

        # Apply a shift on global mask.
        global_mask = global_shift_mask(global_mask, options.y_shift, options.x_shift)
        mask_img = Image.fromarray(global_mask)
        mask_img.save(save_dir + "/predictions/{}_mask.png".format(name))
        rle_pred = rle_encode_less_memory(global_mask)

        subm[i] = {'id': name, 'predicted': rle_pred}
        del global_mask, rle_pred
        print("processed {}".format(name))
    df_sub = pd.DataFrame(subm).T
    df_sub.to_csv(save_dir + "/predictions/submission.csv", index=False)
    print("Done Testing")

if __name__ == '__main__':
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # save dir
    save_dir = BASE_DIR + '/save/20210220_171241'

    model = get_efficientunet_b2(out_channels=1, pretrained=False)
    cudnn.benchmark = True
    model.cuda()
    model = nn.DataParallel(model)
    checkpoint = torch.load(options.load_model)
    state_dict = checkpoint['state_dict']  # Need this if saved multiple things like global step, etc in the checkpoint.
    model.load_state_dict(state_dict)

    predict()


