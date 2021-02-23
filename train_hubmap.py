import os
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Adam
from datetime import datetime

import tifffile as tiff
from utils.HuBMAPCropDataset import HuBMAPCropDataset
from models.EfficientUnet.efficientnet import EfficientNet
from models.EfficientUnet.efficient_unet import *
import pandas as pd
from hubmap_config import options
import numpy as np
from PIL import Image
from torchvision import transforms
from prettytable import PrettyTable

os.environ['CUDA_VISIBLE_DEVICES'] = '{}'.format(options.gpu)


def log_string(out_str):
    LOG_FOUT.write(out_str + '\n')
    LOG_FOUT.flush()
    print(out_str)


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


def get_dice_coeff(pred, targs):
    """
    Calculates the dice coeff of a single or batch of predicted mask and true masks.

    Args:
        pred : Batch of Predicted masks (b, w, h) or single predicted mask (w, h)
        targs : Batch of true masks (b, w, h) or single true mask (w, h)

    Returns: Dice coeff over a batch or over a single pair.
    """
    pred = (pred > options.threshold).float()
    return 2.0 * (pred * targs).sum() / ((pred + targs).sum() + 0.0001)


class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=0.0001):
        # comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2. * intersection) / (inputs.sum() + targets.sum() + smooth)

        return 1 - dice


def train():
    log_string('lr is ' + str(options.lr))
    global_step = 0
    best_loss = 10000
    best_acc = 0

    for epoch in range(options.epochs):
        log_string('**' * 40)
        log_string('Training Epoch %03d, Learning Rate %g' % (epoch + 1, optimizer.param_groups[0]['lr']))
        model.train()
        losses = 0
        count = 0

        for i, data in enumerate(train_loader):
            img_batch, true_masks = data
            img_batch, true_masks = img_batch.to(device, dtype=torch.float), true_masks.to(device, dtype=torch.float)

            pred_mask = model(img_batch)

            loss = criterion(pred_mask, true_masks)
            losses += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            count += 1
            global_step += 1
            if (i + 1) % options.disp_freq == 0:
                log_string("epoch: {0}, batch_id:{1} train_dice_loss: {2:.4f}".format(epoch + 1, i + 1, losses/count))
                losses = 0
                count = 0
        log_string('--' * 40)
        log_string('Evaluating at epoch #{}'.format(epoch+1))
        best_loss, best_acc = evaluate(best_loss=best_loss, best_acc=best_acc, global_step=global_step)


def evaluate(**kwargs):
    best_loss = kwargs['best_loss']
    best_acc = kwargs['best_acc']
    global_step = kwargs['global_step']
    model.eval()
    val_loss = 0
    val_acc = 0
    with torch.no_grad():
        for i, data in enumerate(val_loader):
            img_batch, true_masks = data
            img_batch, true_masks = img_batch.to(device, dtype=torch.float), true_masks.to(device, dtype=torch.float)
            pred_mask = model(img_batch)

            loss = criterion(pred_mask, true_masks)
            val_loss += loss.item()
            val_acc += get_dice_coeff(pred_mask, true_masks)

        val_loss = val_loss/(i+1)
        val_acc = val_acc/(i+1)

    # check for improvement
    loss_str, acc_str = '', ''
    if val_loss <= best_loss:
        loss_str, best_loss = '(improved)', val_loss

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
    if val_acc >= best_acc:
        acc_str, best_acc = '(improved)', val_acc
    # display
    log_string("validation_loss: {0:.4f} {1}".format(val_loss, loss_str))
    log_string("validation_accuracy: {0:.4f} {1}".format(val_acc, acc_str))
    log_string('--' * 40)
    return best_loss, best_acc


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
    model = get_efficientunet_b2(out_channels=options.num_classes, pretrained=False)

    # model = EfficientNet.from_pretrained('efficientnet-b2')

    log_string('{} model Generated.'.format(options.model))
    log_string("Number of trainable parameters: {}\n".format(sum(param.numel() for param in model.parameters())))

    log_string('Options:')
    table = PrettyTable(['Name', 'Value'])
    table.add_row(['Parameters', sum(param.numel() for param in model.parameters())])
    table.add_row(['Log/Checkpoint Files', save_dir + '/log_train.txt'])
    table.add_row(['TB Files', logs_dir])
    for k in range(len(list(options.__dict__.keys()))):
        if list(options.__dict__.keys())[k] == 'load_model_path':
            continue
        else:
            table.add_row([list(options.__dict__.keys())[k], str(list(options.__dict__.values())[k])])
    log_string(str(table) + '\n')

    ##################################
    # Use cuda
    ##################################
    cudnn.benchmark = True
    model.cuda()
    model = nn.DataParallel(model)

    ##################################
    # Loss and Optimizer
    ##################################
    # ToDo: Consider using nn.BCEWithLogitsLoss as criterion
    criterion = DiceLoss()
    optimizer = Adam(model.parameters(), lr=options.lr)

    ##################################
    # Load dataset
    ##################################
    # os.system('cp {}/dataset/dataset.py {}'.format(BASE_DIR, save_dir))

    train_dataset = HuBMAPCropDataset(BASE_DIR + "/trainData", mode="train")
    train_loader = DataLoader(train_dataset, batch_size=options.batch_size,
                              shuffle=True, num_workers=options.workers, drop_last=False)

    val_dataset = HuBMAPCropDataset(BASE_DIR + "/trainData", mode="val")
    val_loader = DataLoader(val_dataset, batch_size=options.batch_size,
                            shuffle=False, num_workers=options.workers, drop_last=False)

    ##################################
    # TRAINING
    ##################################
    log_string('')
    log_string('Start training: Total epochs: {}, Batch size: {}, Training size: {}, Validation size: {}'.
               format(options.epochs, options.batch_size, len(train_dataset), len(val_dataset)))

    train()