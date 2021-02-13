import os
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Adam

from HuBMAPCropDataset import HuBMAPCropDataset
from models.EfficientUnet.efficientnet import EfficientNet
from models.EfficientUnet.efficient_unet import *
from config import options

os.environ['CUDA_VISIBLE_DEVICES'] = '1'


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
        print("Predicted final layer shape: ", inputs.shape)
        print(inputs)
        inputs = F.sigmoid(inputs)

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
        for i, data in enumerate(train_loader):
            slice_img, slice_mask = data
            slice_img, slice_mask = slice_img.to(device, dtype=torch.float), slice_mask.to(device, dtype=torch.float)
            pred_mask_btch = model(slice_img)

            loss = criterion(pred_mask_btch, slice_mask)
            #print("Loss",loss)
            losses += loss.item()
            #dice_coeff = get_dice_coeff(torch.squeeze(pred_mask_btch), slice_mask)
            #dice_coeff_list += dice_coeff

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % options.disp_freq == 0:
                print("epoch: {0}, batch_id:{1} train_dice_loss: {2:.4f}".format(epoch + 1, i + 1, losses/(i + 1)))
                losses = 0

def predict():
    pass



if __name__ == '__main__':
    ##################################
    # Initialize saving directory
    ##################################
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # save_dir = options.save_dir
    # if not os.path.exists(save_dir):
    #     os.makedirs(save_dir)
    # save_dir = os.path.join(save_dir, datetime.now().strftime('%Y%m%d_%H%M%S'))
    # os.makedirs(save_dir)
    #
    # print(str(options) + '\n')
    #
    # model_dir = os.path.join(save_dir, 'models')
    # logs_dir = os.path.join(save_dir, 'tf_logs')
    # if not os.path.exists(model_dir):
    #     os.makedirs(model_dir)

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
    # Loss and Optimizer
    ##################################
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
    # test_dataset = data(mode='test', data_len=options.data_len)
    # test_loader = DataLoader(test_dataset, batch_size=options.batch_size,
    #                          shuffle=False, num_workers=options.workers, drop_last=False)

    ##################################
    # TRAINING
    ##################################
    print('')
    print('Start training: Total epochs: {}, Batch size: {}, Training size: {}, Validation size: ...'.
               format(options.epochs, options.batch_size, len(train_dataset)))
    train()
    # predict()
