import os
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from datetime import datetime
from HuBMAPCropDataset import HuBMAPCropDataset
from models.EfficientUnet.efficientnet import EfficientNet
from models.EfficientUnet.efficient_unet import *
import pandas as pd
from config import options
import numpy as np
from PIL import Image


def rle_encode_less_memory(img):
    pixels = img.T.flatten()
    pixels[0] = 0
    pixels[-1] = 0
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 2
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


def global_shift_mask(maskpred1, y_shift, x_shift):
    """
    applies a global shift to a mask by
    padding one side and cropping from the other
    """
    maskpred3 = None
    if y_shift < 0 and x_shift >= 0:
        maskpred2 = np.pad(maskpred1,
                           [(0, abs(y_shift)), (abs(x_shift), 0)],
                           mode='constant', constant_values=0)
        maskpred3 = maskpred2[abs(y_shift):, :maskpred1.shape[1]]
    elif y_shift >= 0 and x_shift < 0:
        maskpred2 = np.pad(maskpred1,
                           [(abs(y_shift), 0), (0, abs(x_shift))],
                           mode='constant', constant_values=0)
        maskpred3 = maskpred2[:maskpred1.shape[0], abs(x_shift):]
    elif y_shift >= 0 and x_shift >= 0:
        maskpred2 = np.pad(maskpred1,
                           [(abs(y_shift), 0), (abs(x_shift), 0)],
                           mode='constant', constant_values=0)
        maskpred3 = maskpred2[:maskpred1.shape[0], :maskpred1.shape[1]]
    elif y_shift < 0 and x_shift < 0:
        maskpred2 = np.pad(maskpred1,
                           [(0, abs(y_shift)), (0, abs(x_shift))],
                           mode='constant', constant_values=0)
        maskpred3 = maskpred2[abs(y_shift):, abs(x_shift):]
    return maskpred3


def predict():
    subm = {}
    all_files = os.listdir(BASE_DIR + "/testData")
    patient_files = [x for x in all_files if '.tiff' in x]
    for patient_file in patient_files:
        name = patient_file[:-5]
        print('--' * 40)
        print("Predicting for patient: {}".format(name))
        test_dataset = HuBMAPCropDataset(BASE_DIR + "/testData", mode='test', patient=name)
        test_loader = DataLoader(test_dataset, batch_size=options.batch_size,
                                 shuffle=False, num_workers=options.workers, drop_last=False)
        height, width = test_dataset.get_global_image_size()
        global_mask = torch.zeros((height, width), dtype=torch.int8)
        # global_mask = global_mask.to(device, dtype=torch.int8)
        model.eval()
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
                    global_mask[int(y1):int(y2), int(x1):int(x2)] = each_mask
        global_mask = global_mask.numpy()

        # Apply a shift on global mask.
        global_mask = global_shift_mask(global_mask, options.y_shift, options.x_shift)
        mask_img = Image.fromarray(global_mask)
        mask_img.save(BASE_DIR + "/predictions/{}_mask.png".format(name))
        rle_pred = rle_encode_less_memory(global_mask)

        subm[i] = {'id': name, 'predicted': rle_pred}
        del global_mask, rle_pred
        print("processed {}".format(name))
    df_sub = pd.DataFrame(subm).T
    df_sub.to_csv(BASE_DIR + "/predictions/submission.csv", index=False)
    print("Done Testing")


if __name__ == '__main__':
    ##################################
    # Initialize saving directory
    ##################################
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    predictions_dir = os.path.join(BASE_DIR, 'predictions')
    if not os.path.exists(predictions_dir):
        os.makedirs(predictions_dir)

    os.system('cp {}/train.py {}'.format(BASE_DIR, predictions_dir))
    os.system('cp {}/HuBMAPCropDataset.py {}'.format(BASE_DIR, predictions_dir))

    ##################################
    # Create the model
    ##################################
    model = get_efficientunet_b2(out_channels=1, pretrained=False)
    # model = EfficientNet.from_pretrained('efficientnet-b2')

    print('{} model Generated.'.format(options.model))
    print("Number of trainable parameters: {}".format(sum(param.numel() for param in model.parameters())))

    ##################################
    # Use cuda
    ##################################
    cudnn.benchmark = True
    model.cuda()
    model = nn.DataParallel(model)

    ##################################
    # Load dataset
    ##################################
    train_dataset = HuBMAPCropDataset(BASE_DIR + "/trainData", mode="train")
    train_loader = DataLoader(train_dataset, batch_size=options.batch_size,
                              shuffle=True, num_workers=options.workers, drop_last=False)

    val_dataset = HuBMAPCropDataset(BASE_DIR + "/trainData", mode="val")
    val_loader = DataLoader(val_dataset, batch_size=options.batch_size,
                            shuffle=False, num_workers=options.workers, drop_last=False)

    ##################################
    # PREDICTING
    ##################################
    print('')
    print('Starting Predictions on Global Mask')
    predict()
