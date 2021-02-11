import os
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Adam

from HuBMAPDataset import HuBMAPDataset
from models.EfficientUnet.efficientnet import EfficientNet
from models.EfficientUnet.efficient_unet import *
from config import options
import pandas as pd

os.environ['CUDA_VISIBLE_DEVICES'] = '1'


def log_string(out_str):
    LOG_FOUT.write(out_str + '\n')
    LOG_FOUT.flush()
    print(out_str)


#Convert mask to rle string.
def rle_encode_less_memory(img):
    pixels = img.T.flatten()
    pixels[0] = 0
    pixels[-1] = 0
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 2
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


def get_dice_coeff(pred, targs):
    '''
    Calculates the dice coeff of a single or batch of predicted mask and true masks.

    Args:
        pred : Batch of Predicted masks (b, w, h) or single predicted mask (w, h)
        targs : Batch of true masks (b, w, h) or single true mask (w, h)

    Returns: Dice coeff over a batch or over a single pair.
    '''

    pred = (pred > 0).float()
    return 2.0 * (pred * targs).sum() / ((pred + targs).sum() + 1.0)


class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        # comment out if your model contains a sigmoid or equivalent activation layer
        # log_string(inputs)
        # inputs = F.sigmoid(inputs)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)

        return 1 - dice

def train():
    model.train()
    losses = 0
    dice_coeff_list = []
    for epoch in range(options.epochs):
        log_string("Epoch " + str(epoch) + " starting...")
        for i, data in enumerate(train_loader):
            slice_img, slice_mask, _ = data
            slice_img, slice_mask = slice_img.to(device, dtype=torch.float), slice_mask.to(device, dtype=torch.float)
            slice_img = slice_img.permute(0, 3, 1, 2)
            pred_mask_btch = model(slice_img)

            loss = criterion(pred_mask_btch, slice_mask)
            # log_string("Loss",loss)
            losses += loss.item()
            # dice_coeff = get_dice_coeff(torch.squeeze(pred_mask_btch), slice_mask)
            # dice_coeff_list += dice_coeff

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % options.disp_freq == 0:
                log_string("epoch: {0}, batch_id:{1} train_dice_loss: {2:.4f}".format(epoch + 1, i + 1, losses/(i + 1)))
                losses = 0

            if (i + 1) % options.val_freq == 0:
                log_string('**' * 40)
                log_string("Evaluating at batch: " + str(i + 1))
                evaluate()
                log_string('**' * 40)


def evaluate():
    model.eval()
    losses = 0
    meta_data = pd.read_csv(BASE_DIR + '/trainData/HuBMAP-20-dataset_information.csv')
    width = meta_data.iloc[options.test_tiff_value]['width_pixels']
    height = meta_data.iloc[options.test_tiff_value]['height_pixels']
    entire_mask = torch.zeros((width, height))

    with torch.no_grad():
        batch_count = 0
        total_loss = 0
        for i, data in enumerate(val_loader):
            batch_count += 1
            slice_img, slice_mask, _ = data
            slice_img, slice_mask = slice_img.to(device, dtype=torch.float), slice_mask.to(device, dtype=torch.float)
            slice_img = slice_img.permute(0, 3, 1, 2)
            pred_mask_batch = model(slice_img)

            loss = criterion(pred_mask_batch, slice_mask)
            total_loss += loss.item()
        log_string("Evaluation Loss " + str(total_loss / batch_count))


def predict():
    subsmission = {}
    model.eval()
    meta_data = pd.read_csv(BASE_DIR + '/trainData/HuBMAP-20-dataset_information.csv')
    all_files = os.listdir(BASE_DIR + "testData/")
    patients = [file[:-5] for file in all_files if file[-5:] == ".tiff"]
    with torch.no_grad():
        for i, patient in enumerate(patients):
            patient_meta = meta_data.loc[meta_data['image_file'] == patients[0]+".tiff"]
            width, height = patient_meta['width'], patient_meta['height']
            entire_mask = torch.zeros((height, width))
            test_dataest = HuBMAPDataset(BASE_DIR + "/testData", mode="test", patient=patient)
            test_loader = DataLoader(test_dataest, batch_size=options.batch_size,
                                     shuffle=False, num_workers=options.workers, drop_last=False)
            for data in test_loader:
                slice_img, sliceIdxBatch = data
                slice_img = slice_img.to(device, dtype=torch.float),
                slice_img = slice_img.permute(0, 3, 1, 2)
                pred_mask_batch = model(slice_img)

                # Crate global mask
                for mask,sliceIdx in zip(pred_mask_batch, sliceIdxBatch):
                    x1, x2, y1, y2 = sliceIdx
                    entire_mask[x1:x2, y1:y2] = pred_mask_batch
            # Encode mask to rle and submit.
            rle_pred = rle_encode_less_memory[entire_mask]
            subsmission[i] = {'id' : patient+".tiff", 'predicted' : rle_pred}
    df_sub = pd.DataFrame(subsmission).T
    df_sub.to_csv('submission.csv', index=False)


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
    log_string('Options:')
    for key, val in options.__dict__.items():
        log_string(key + ": " + str(val))

    model_dir = os.path.join(save_dir, 'models')
    logs_dir = os.path.join(save_dir, 'tf_logs')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    ##################################
    # Create the model
    ##################################
    model = get_efficientunet_b2(out_channels=1, pretrained=False)
    # model = EfficientNet.from_pretrained('efficientnet-b2')

    log_string('\n{} model Generated.'.format(options.model))
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

    train_dataset = HuBMAPDataset(BASE_DIR + "/trainData", mode="train")
    train_loader = DataLoader(train_dataset, batch_size=options.batch_size,
                              shuffle=True, num_workers=options.workers, drop_last=False)
    val_dataset = HuBMAPDataset(BASE_DIR + "/trainData", mode='val')
    val_loader = DataLoader(val_dataset, batch_size=options.batch_size,
                            shuffle=False, num_workers=options.workers, drop_last=False)

    ##################################
    # TRAINING
    ##################################
    log_string('')
    log_string('Start training: Total epochs: {}, Batch size: {}, Training size: {}, Validation size: ...'.
               format(options.epochs, options.batch_size, len(train_dataset)))
    train()
    predict()
