import torch
import torch.nn.functional as F
from utils.config import options
# from tqdm import tqdm

from dice_loss import dice_coeff


def eval_net(net, loader, device):
    """Evaluation without the densecrf with the dice coefficient"""
    net.eval()
    mask_type = torch.float32 if net.n_classes == 1 else torch.long
    n_val = len(loader)  # the number of batch
    tot = 0
    tIOU = 0
    SMOOTH = 1e-6

    # with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
    for batch in loader:
        imgs, true_masks = batch['image'], batch['mask']
        imgs = imgs.to(device=device, dtype=torch.float32)
        true_masks = true_masks.to(device=device, dtype=mask_type)

        with torch.no_grad():
            if (options.arch == 'HRNetM') or (options.arch == 'DeepLabv3+x71') or (options.arch == 'PanDeepLabx71'):
                inputs = {'images': imgs, 'gts': true_masks}
                out = net(inputs)
                mask_pred = out['pred']
            else:
                mask_pred = net(imgs)

        if net.n_classes > 1:
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
            pred = (pred > 0.5).float()
            tot += dice_coeff(pred, true_masks).item()
            # pbar.update()

    net.train()
    return (tot / n_val), (tIOU / n_val)


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
