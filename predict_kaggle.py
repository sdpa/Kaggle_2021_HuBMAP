import os
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from utils.config import options
from train_hubmap import global_shift_mask, rle_encode_less_memory
from utils.HuBMAPCropDataset import HuBMAPCropDataset
from torch.utils.data import DataLoader
from torchvision import transforms
import pandas as pd
import matplotlib.pyplot as plt
from models.EfficientUnet.efficient_unet import *
from datetime import datetime

os.environ['CUDA_VISIBLE_DEVICES'] = options.gpu


def log_string(out_str):
    LOG_FOUT.write(out_str + '\n')
    LOG_FOUT.flush()
    print(out_str)


def predict():
    subm = {}
    all_files = os.listdir(BASE_DIR + "/testData")
    patient_files = [x for x in all_files if '.tiff' in x]
    model.eval()

    vert_flip = transforms.RandomVerticalFlip(p=1)
    horiz_flip = transforms.RandomHorizontalFlip(p=1)

    for patient_file in patient_files:
        name = patient_file[:-5]
        print('--' * 40)
        print("Predicting for patient: {}".format(name))
        test_dataset = HuBMAPCropDataset(BASE_DIR + "/testData", mode='test', patient=name)
        test_loader = DataLoader(test_dataset, batch_size=options.batch_size, shuffle=False,
                                 num_workers=options.workers, drop_last=False)
        height, width = test_dataset.get_global_image_size()
        global_mask = torch.zeros((height, width), dtype=torch.int8)
        with torch.no_grad():
            for i, data in enumerate(test_loader):
                # Test Time Augmentation (TTA) - Normal, Horiz flip, Vert flip
                img_batch, coordinates_batch, imgs_horiz, imgs_vert = data

                img_batch = img_batch.to(device, dtype=torch.float)
                coordinates_batch = coordinates_batch.to(device, dtype=torch.float)
                imgs_horiz = imgs_horiz.to(device, dtype=torch.float)
                imgs_vert = imgs_vert.to(device, dtype=torch.float)

                pred_mask_batch = model(img_batch)
                pred_mask_horiz = model(imgs_horiz)
                pred_mask_vert = model(imgs_vert)

                # Flip horizontally and vertically-flipped masks back
                for k in range(len(pred_mask_horiz)):
                    pred_mask_horiz[k] = horiz_flip(pred_mask_horiz[k])  # Flip horizontal masks back
                for m in range(len(pred_mask_vert)):
                    pred_mask_vert[m] = vert_flip(pred_mask_vert[m])  # Flip vertical masks back

                # Combine three masks back into pred_mask_batch
                pred_mask_batch = (pred_mask_batch + pred_mask_horiz + pred_mask_vert) / 3

                # Converts mask to 0/1.
                pred_mask_batch = torch.sigmoid(pred_mask_batch)
                pred_mask_batch = (pred_mask_batch > options.threshold).type(torch.int8)

                # Loop through each img,mask in batch.
                for each_mask, coordinate in zip(pred_mask_batch, coordinates_batch):
                    each_mask = torch.squeeze(each_mask)
                    # xs = columns, ys = rows. (x1,y1) --> Top Left. (x2,y2) --> bottom right.
                    x1, x2, y1, y2 = coordinate
                    global_mask[int(y1):int(y2), int(x1):int(x2)] = each_mask
        global_mask = global_mask.numpy()

        # Apply a shift on global mask.
        if name == "afa5e8098":
            global_mask = global_shift_mask(global_mask, options.y_shift, options.x_shift)
        # Plot global mask imgs for viewing on sciview/figure
        plt.imsave(SAVE_DIR + '/' + name + "_mask.png", global_mask)
        rle_pred = rle_encode_less_memory(global_mask)

        subm[i] = {'id': name, 'predicted': rle_pred}
        del global_mask, rle_pred
        print("processed {}".format(name))
    df_sub = pd.DataFrame(subm).T
    df_sub.to_csv(SAVE_DIR + "/submission.csv", index=False)
    print("Done Testing")


if __name__ == '__main__':
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # save dir
    SAVE_DIR = BASE_DIR + '/predictions/' + datetime.now().strftime('%Y_%m_%d_%H%M%S')
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    LOG_FOUT = open(os.path.join(SAVE_DIR, 'info.txt'), 'w')
    log_string("Loaded Model: " + options.load_model_path)
    log_string("Threshold used: " + str(options.threshold))
    log_string("Kaggle Score: ")

    model = get_efficientunet_b2(out_channels=1, pretrained=False)
    cudnn.benchmark = True
    model.cuda()
    # model = nn.DataParallel(model)
    checkpoint = torch.load(options.load_model_path)
    state_dict = checkpoint['state_dict']  # Need this if saved multiple things like global step, etc in the checkpoint.
    model.load_state_dict(state_dict)
    print('Successfully loaded model from', options.load_model_path)

    predict()

"""
import matplotlib.pyplot as plt
import numpy as np

for i in range(4):
    plt.imshow((pred_mask_batch[i].detach().cpu().squeeze(0).numpy()) > 0.39)
    plt.show()
"""
