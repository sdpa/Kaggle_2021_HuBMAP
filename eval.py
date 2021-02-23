import torch
import torch.nn.functional as F
from utils.config import options
import numpy as np
import pandas as pd
# from tqdm import tqdm

from dice_loss import dice_coeff


def rle_to_mask(_encoding, size):
    # encoding is a string. convert it to int.
    mask = np.zeros(size[0] * size[1], dtype=np.uint8)
    encoding_np = np.fromstring(_encoding, dtype=int, sep=' ')
    starts = np.array(encoding_np)[::2]
    lengths = np.array(encoding_np)[1::2]
    for start, length in zip(starts, lengths):
        mask[start:start + length] = 1
    return mask.reshape(size, order='F')


def eval_net(net, loader, val_shape, device):
    """Evaluation without the densecrf with the dice coefficient"""
    net.eval()
    mask_type = torch.float32 if options.n_classes == 1 else torch.long
    n_val = len(loader)  # the number of batch
    tot = 0
    tIOU = 0
    SMOOTH = 1e-6

    # global_mask = np.zeros(val_shape, dtype=np.uint8)
    pred_global_mask = torch.zeros(val_shape, dtype=torch.int8)

    with torch.no_grad():
        for batch in loader:
            imgs, true_masks, coords = batch['image'], batch['mask'], batch['coords']
            imgs = imgs.to(device=device, dtype=torch.float32)
            true_masks = true_masks.to(device=device, dtype=mask_type)

            if (options.arch == 'HRNetM') or (options.arch == 'DeepLabv3+x71') or (options.arch == 'PanDeepLabx71'):
                inputs = {'images': imgs, 'gts': true_masks}
                out = net(inputs)
                mask_pred = out['pred']
            else:
                mask_pred = net(imgs)

            if options.n_classes > 1:  # Changed from net.n_classes
                tot += F.cross_entropy(mask_pred, true_masks).item()
                sol = torch.argmax(mask_pred, dim=1)
                if options.data_name == 'CamVid':
                    sol[torch.where(true_masks == 11)] = 11
                elif options.data_name == 'Cityscapes':
                    sol[torch.where(true_masks == 19)] = 19
                intersection = (sol & true_masks).float().sum((1, 2))
                union = (sol | true_masks).float().sum((1, 2))
                iou = (intersection + SMOOTH) / (union + SMOOTH)
                # thresholded = torch.clamp(20 * (iou - 0.5), 0, 10).ceil() / 10
                tIOU += iou.mean()
            else:
                pred = torch.sigmoid(mask_pred)
                pred = (pred > options.threshold).float()
                tot += dice_coeff(pred, true_masks).item()
                # pbar.update()

                # Assembling global mask
                for mask, coordinate in zip(pred, coords):
                    each_mask = torch.squeeze(each_mask)
                    # xs = columns, ys = rows. (x1,y1) --> Top Left. (x2,y2) --> bottom right.
                    x1, x2, y1, y2 = coordinate
                    pred_global_mask[int(y1):int(y2),int(x1):int(x2)] = each_mask
    pred_global_mask = pred_global_mask.numpy()

    # Load global mask for validation image
    train_pats = pd.read_csv("/home/cougarnet.uh.edu/srizvi7/Desktop/Kaggle_2021_HuBMAP/trainData/train.csv")
    valid_tiff_name = train_pats.iloc[options.test_tiff_value]['id']
    encoding = train_pats.loc[train_pats["id"] == valid_tiff_name]['encoding'].iloc[0]
    true_global_mask = rle_to_mask(encoding, val_shape)

    net.train()
    return (tot / n_val), (tIOU / n_val), dice_coeff(pred_global_mask, true_global_mask).item()


def eval_train(options, pred, masks):
    """Evaluation on training batch"""
    tIOU = 0
    SMOOTH = 1e-6

    sol = torch.argmax(pred, dim=1)
    if options.data_name == 'CamVid':
        sol[torch.where(masks == 11)] = 11
    elif options.data_name == 'Cityscapes':
        sol[torch.where(masks == 19)] = 19
    intersection = (sol & masks).float().sum((1, 2))
    union = (sol | masks).float().sum((1, 2))
    iou = (intersection + SMOOTH) / (union + SMOOTH)
    tIOU += iou.mean()

    return tIOU
