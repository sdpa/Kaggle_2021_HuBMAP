import os
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Adam
from datetime import datetime

import tifffile as tiff
from HuBMAPCropDataset import HuBMAPCropDataset
from models.EfficientUnet.efficientnet import EfficientNet
from models.EfficientUnet.efficient_unet import *
import pandas as pd
from config import options
import numpy as np
from PIL import Image
from utils import global_shift_mask, rle_encode_less_memory, rle_to_mask
from torchvision import transforms

os.environ['CUDA_VISIBLE_DEVICES'] = '1'


def log_string(out_str):
    LOG_FOUT.write(out_str + '\n')
    LOG_FOUT.flush()
    print(out_str)


def get_dice_coeff(pred, targs):
    '''
    Calculates the dice coeff of a single or batch of predicted mask and true masks.

    Args:
        pred : Batch of Predicted masks (b, w, h) or single predicted mask (w, h)
        targs : Batch of true masks (b, w, h) or single true mask (w, h)

    Returns: Dice coeff over a batch or over a single pair.
    '''
    pred = torch.sigmoid(pred)
    # Converts mask to 0/1.
    pred = (pred > options.threshold)
    return (2.0 * (pred * targs).sum()) / ((pred + targs).sum() + 0.0001)


class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=0.0001):
        # comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)

        targets = targets.contiguous()
        targets = targets.view(-1)

        # Groudn truth is all zeros.
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)

        return 1 - dice


def train():
    log_string('lr is ' + str(options.lr))

    best_loss = 100
    best_acc = 0

    model.train()
    losses = 0
    count = 0
    global_step = 0

    for epoch in range(options.epochs):
        log_string('**' * 40)
        log_string('Training Epoch %03d' % (epoch + 1))
        for i, data in enumerate(train_loader):
            # with torch.cuda.device(0):
            img_batch, mask_batch = data
            img_batch, mask_batch = img_batch.to(device, dtype=torch.float), mask_batch.to(device, dtype=torch.float)
            pred_mask_batch = model(img_batch)

            loss = criterion(pred_mask_batch, mask_batch)
            # print("Loss",loss)
            losses += loss.item()
            # dice_coeff = get_dice_coeff(torch.squeeze(pred_mask_btch), slice_mask)
            # dice_coeff_list += dice_coeff

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            count += 1
            global_step += 1
            if (i + 1) % options.disp_freq == 0:
                log_string("epoch: {0}, batch_id:{1} train_dice_loss: {2:.4f}".format(epoch + 1, i + 1, losses / count))
                losses = 0
                count = 0
                best_loss, best_acc = evaluate(best_loss=best_loss, best_acc=best_acc, global_step=global_step)
        log_string('--' * 40)
        log_string('Evaluating at epoch #{}'.format(epoch + 1))
        # best_loss, best_acc = evaluate(best_loss=best_loss, best_acc=best_acc, global_step=global_step)
        model.train()


def evaluate(**kwargs):
    best_loss = kwargs['best_loss']
    best_acc = kwargs['best_acc']
    global_step = kwargs['global_step']
    model.eval()
    height, width = val_dataset.get_global_image_size()
    global_mask_pred = torch.zeros((height, width), dtype=torch.float).to(device)
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(val_loader):
            img_batch, coordinates_batch = data
            img_batch = img_batch.to(device, dtype=torch.float)
            coordinates_batch = coordinates_batch.to(device, dtype=torch.float)
            pred_mask_batch = model(img_batch)

            # Loop through each img,mask in batch.
            for each_mask, coordinate in zip(pred_mask_batch, coordinates_batch):
                each_mask = torch.squeeze(each_mask)
                # xs = columns, ys = rows. (x1,y1) --> Top Left. (x2,y2) --> bottom right.
                x1, x2, y1, y2 = coordinate
                global_mask_pred[int(y1):int(y2), int(x1):int(x2)] = each_mask

    # pass through criterion to get loss.
    val_loss = criterion(global_mask_pred, global_mask_target)
    val_acc = get_dice_coeff(global_mask_pred, global_mask_target)

    # check for improvement
    loss_str, acc_str = '', ''
    improved = False
    if val_loss <= best_loss:
        loss_str, best_loss = '(improved)', val_loss
        improved = True
    if val_acc >= best_acc:
        acc_str, best_acc = '(improved)', val_acc
    if val_loss <= best_loss or val_acc >= best_acc:
        # save checkpoint model
        state_dict = model.state_dict()
        for key in state_dict.keys():
            state_dict[key] = state_dict[key].cpu()
        save_path = os.path.join(model_dir, '{}.ckpt'.format(global_step))
        torch.save({
            'global_step': global_step,
            'loss': val_loss,
            'acc': val_acc,
            'save_dir': model_dir,
            'state_dict': state_dict},
            save_path)
        log_string('Model saved at: {}'.format(save_path))
    # display
    log_string("validation_loss: {0:.4f} {1}".format(val_loss, loss_str))
    log_string("validation_accuracy: {0:.4f} {1}".format(val_acc, acc_str))
    log_string('--' * 40)
    return best_loss, best_acc


# Run test.py for test predictions
# def predict():
#     subm = {}
#     all_files = os.listdir(BASE_DIR + "/testData")
#     patient_files = [x for x in all_files if '.tiff' in x]
#     best_model = os.listdir(model_dir)[-1]
#     model.load_state_dict(torch.load(save_dir + '/models/{}'.format(best_model)))
#     model.eval()
#     for patient_file in patient_files:
#         name = patient_file[:-5]
#         log_string('--' * 40)
#         log_string("Predicting for patient: {}".format(name))
#         test_dataset = HuBMAPCropDataset(BASE_DIR + "/HuBMAP_Dataset/test", mode='test', patient=name)
#         test_loader = DataLoader(test_dataset, batch_size=8,
#                                  shuffle=False, num_workers=options.workers, drop_last=False)
#         height, width = test_dataset.get_global_image_size()
#         global_mask = torch.zeros((height, width), dtype=torch.int8)
#         # global_mask = global_mask.to(device, dtype=torch.int8)
#         with torch.no_grad():
#             for i, data in enumerate(test_loader):
#                 img_batch, coordinates_batch = data
#                 img_batch = img_batch.to(device, dtype=torch.float)
#                 coordinates_batch = coordinates_batch.to(device, dtype=torch.float)
#                 pred_mask_batch = model(img_batch)
#                 pred_mask_batch = torch.sigmoid(pred_mask_batch)
#
#                 # Loop through each img,mask in batch.
#                 for each_mask, coordinate in zip(pred_mask_batch, coordinates_batch):
#                     each_mask = torch.squeeze(each_mask)
#                     # xs = columns, ys = rows. (x1,y1) --> Top Left. (x2,y2) --> bottom right.
#                     x1, x2, y1, y2 = coordinate
#                     global_mask[int(y1):int(y2), int(x1):int(x2)] = each_mask
#         global_mask = global_mask.numpy()
#
#         # Apply a shift on global mask.
#         global_mask = global_shift_mask(global_mask, options.y_shift, options.x_shift)
#         mask_img = Image.fromarray(global_mask)
#         mask_img.save(save_dir + "/predictions/{}_mask.png".format(name))
#         rle_pred = rle_encode_less_memory(global_mask)
#
#         subm[i] = {'id': name, 'predicted': rle_pred}
#         del global_mask, rle_pred
#         log_string("processed {}".format(name))
#     df_sub = pd.DataFrame(subm).T
#     df_sub.to_csv(save_dir + "/predictions/submission.csv", index=False)
#     log_string("Done Testing")


if __name__ == '__main__':
    ##################################
    # Initialize saving directory
    ##################################
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    save_dir = options.save_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_dir = os.path.join(save_dir, datetime.now().strftime('%Y%m%d_%H%M%S'))
    os.makedirs(save_dir)

    LOG_FOUT = open(os.path.join(save_dir, 'log_train.txt'), 'w')
    LOG_FOUT.write(str(options) + '\n')
    print(str(options) + '\n')

    model_dir = os.path.join(save_dir, 'models')
    logs_dir = os.path.join(save_dir, 'tf_logs')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    predictions_dir = os.path.join(save_dir, 'predictions')
    if not os.path.exists(predictions_dir):
        os.makedirs(predictions_dir)

    os.system('cp {}/train.py {}'.format(BASE_DIR, save_dir))
    os.system('cp {}/HuBMAPCropDataset.py {}'.format(BASE_DIR, save_dir))

    ##################################
    # Create the model
    ##################################
    model = get_efficientunet_b2(out_channels=1, pretrained=False)

    # model = EfficientNet.from_pretrained('efficientnet-b2')

    log_string('{} model Generated.'.format(options.model))
    log_string("Number of trainable parameters: {}".format(sum(param.numel() for param in model.parameters())))

    ##################################
    # Use cuda
    ##################################
    cudnn.benchmark = True
    model.cuda()
    model = nn.DataParallel(model)
    ##################################
    # Loss and Optimizer
    ##################################
    criterion = DiceLoss()
    optimizer = Adam(model.parameters(), lr=options.lr)

    ##################################
    # Load dataset
    ##################################
    # os.system('cp {}/dataset/dataset.py {}'.format(BASE_DIR, save_dir))

    train_dataset = HuBMAPCropDataset(BASE_DIR + "/HuBMAP_Dataset/", mode="train")
    train_loader = DataLoader(train_dataset, batch_size=options.batch_size,
                              shuffle=True, num_workers=options.workers, drop_last=False)

    name = 'aaa6a05cc' # Make this patient the validation patient.
    encoding_df = pd.read_csv(BASE_DIR + '/HuBMAP_Dataset/train.csv')
    encoding = encoding_df.loc[encoding_df["id"] == name]['encoding'].iloc[0]
    metadata_all = pd.read_csv(BASE_DIR + "/HuBMAP_Dataset/HuBMAP-20-dataset_information.csv")
    width = metadata_all.loc[metadata_all["image_file"] == name + '.tiff']['width_pixels'].iloc[0]
    height = metadata_all.loc[metadata_all["image_file"] == name + '.tiff']['height_pixels'].iloc[0]
    global_mask_target = rle_to_mask(encoding, (height, width))
    global_mask_target = torch.from_numpy(global_mask_target)
    global_mask_target = global_mask_target.to(device, dtype=torch.float)

    val_dataset = HuBMAPCropDataset(BASE_DIR + "/HuBMAP_Dataset/", mode="val", patient=name)
    val_loader = DataLoader(val_dataset, batch_size=options.batch_size,
                            shuffle=False, num_workers=options.workers, drop_last=False)

    ##################################
    # TRAINING
    ##################################
    log_string('')
    log_string('Start training: Total epochs: {}, Batch size: {}, Training size: {}, Validation size: {}'.
               format(options.epochs, options.batch_size, len(train_dataset), len(val_dataset)))

    train()
    # predict()
