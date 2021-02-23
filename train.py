import os
from utils.config import options
import torch.backends.cudnn as cudnn
from torchvision import transforms

import sys
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from utils.logger_utils import Logger
from utils.data_loader import get_train_valid_loader, get_test_loader
from utils.camvid import Cam_LabelTensorToPILImage, cam_class_color, cam_class_weight, cam_mean, cam_std
from utils.cityscapes import City_LabelTensorToPILImage, CITYSCAPES_LABEL_WEIGHTS, CITYSCAPES_MEAN, CITYSCAPES_STD
from utils.visualization import UnNormalize
import time
from datetime import datetime

from eval import eval_net, eval_train
from models import UNet
from models import SegCaps
from models import FCDenseNet57
from models import FCDenseNet67
from models import FCDenseNet103
from models import WCaps
from models import HRNet_Mscale
from models import DeeperX71
from models import DeepV3PlusX71
from models.EfficientUnet.efficient_unet import *

from torch.utils.tensorboard import SummaryWriter
from prettytable import PrettyTable

os.environ['CUDA_VISIBLE_DEVICES'] = options.gpu

dir_checkpoint = options.save_dir

if options.data_name == 'GlomSeg':
    file_naming = f'_{options.data_name}_{options.arch}_{options.optimizer}_LR{options.lr}_BS{options.batch_size}_SCALE{options.scale}_VAL{options.val_int}'
elif options.data_name == 'Cityscapes':
    file_naming = f'_{options.data_name}_{options.arch}_{options.optimizer}_LR{options.lr}_BS{options.batch_size}_COARSE{options.cs_coarse}'
else:
    file_naming = f'_{options.data_name}_{options.arch}_{options.optimizer}_LR{options.lr}_BS{options.batch_size}'

if options.arch == 'WCaps':
    file_naming = file_naming + f'_MODS_{options.modules}_ENCS_{options.enc_ops}_SCALES_{options.wscales}_P_{options.P}'

# For later image logging
if options.data_name == 'CamVid':
    unorm = UnNormalize(mean=cam_mean, std=cam_std)
elif options.data_name == 'Cityscapes':
    unorm = UnNormalize(mean=CITYSCAPES_MEAN, std=CITYSCAPES_STD)


def log_string(out_str):
    LOG_FOUT.write(out_str + '\n')
    LOG_FOUT.flush()
    print(out_str)


def train_net():
    global_step = 0
    best_loss = 10000
    best_acc = 0

    write_dir = options.runs_dir + '/' + datetime.now().strftime('%Y_%m_%d') + '/' + file_naming[1:]
    writer = SummaryWriter(log_dir=write_dir, comment=file_naming)

    log_string('Training Initiated. Listing Training Configurations:')
    table = PrettyTable(['Name', 'Value'])
    table.add_row(['Parameters', sum(param.numel() for param in net.parameters())])
    table.add_row(['Log/Checkpoint Files', dt_save + file_naming])
    table.add_row(['TB Files', write_dir])
    for i in range(len(list(options.__dict__.keys()))):
        if list(options.__dict__.keys())[i] == 'load_model_path':
            continue
        else:
            table.add_row([list(options.__dict__.keys())[i], str(list(options.__dict__.values())[i])])
    log_string(str(table))

    log_string('**' * 30)

    for epoch in range(options.epochs):
        start_time = time.time()
        log_string('**' * 30)
        log_string('Training Epoch %03d, Learning Rate %g' % (epoch + 1, optimizer.param_groups[0]['lr']))
        net.train()
        epoch_loss = 0

        for i, batch in enumerate(train_loader):
            # print('Batch', i)
            imgs = batch['image']
            true_masks = batch['mask']
            # assert imgs.shape[1] == options.n_channels, \
            #     f'Network has been defined with {options.n_channels} input channels, ' \
            #     f'but loaded images have {imgs.shape[1]} channels. Please check that ' \
            #     'the images are loaded correctly.'

            imgs = imgs.to(device=device, dtype=torch.float32)
            mask_type = torch.float32 if options.n_classes == 1 else torch.long
            true_masks = true_masks.to(device=device, dtype=mask_type)

            if (options.arch == 'HRNetM') or (options.arch == 'DeepLabv3+x71') or (options.arch == 'PanDeepLabx71'):
                inputs = {'images': imgs, 'gts': true_masks}
                out = net(inputs)
                loss = out['loss']
                masks_pred = out['pred']
            else:
                masks_pred = net(imgs)
                loss = criterion(masks_pred, true_masks)

            # loss = criterion(masks_pred, true_masks)
            epoch_loss += loss.item()
            writer.add_scalar('Loss/train', loss.item(), global_step)

            optimizer.zero_grad()
            loss.backward()
            # nn.utils.clip_grad_value_(net.parameters(), 0.1) # for older nets, UNet
            optimizer.step()

            global_step += 1
            if (global_step + 1) % options.disp_freq == 0:
                train_acc = eval_train(options, masks_pred, true_masks)
                writer.add_scalar('IOU/train', train_acc, global_step)
                log_string("epoch: {0}, step: {1}, time: {2:.2f}, train_loss: {3:.4f} train_IOU: {4:.3f}"
                           .format(epoch + 1, global_step + 1, time.time() - start_time, loss.item(), train_acc))
                # info = {'loss': epoch_loss,
                #         'accuracy': train_acc}
                # for tag, value in info.items():
                #     train_logger.scalar_summary(tag, value, global_step)
                start_time = time.time()

            # Modified eval function to validate on the global mask
            if global_step % options.val_freq == 0:
                log_string('--' * 30)
                log_string('Evaluating at step #{}'.format(global_step))

                for tag, value in net.named_parameters():
                    tag = tag.replace('.', '/')
                    if value.data is None:
                        writer.add_histogram('weights/' + tag, 0, global_step)
                    else:
                        writer.add_histogram('weights/' + tag, value.data.cpu().numpy(), global_step)
                    if value.grad is None:
                        writer.add_histogram('grads/' + tag, 0, global_step)
                    else:
                        writer.add_histogram('grads/' + tag, value.grad.data.cpu().numpy(), global_step)

                eval_start = time.time()
                val_score, val_IOU, global_dice_score = eval_net(net, val_loader, val_shape, device)
                eval_end = time.time()

                # check for improvement
                loss_str, acc_str = '', ''
                improved = False
                if val_score <= best_loss:
                    loss_str, best_loss = '(improved)', val_score
                if global_dice_score >= best_acc:  # Using dice coefficient on global mask as accuracy
                    acc_str, best_acc = '(improved)', global_dice_score
                    improved = True

                log_string('Validation time: {0:.4f}'.format(eval_end - eval_start))
                log_string("Val Loss: {0:.2f} {1}, Global Dice Score: {2:.3f} {3}, "
                           "Val Avg. IOU: {4:.3f}, lr: {5}"
                           .format(val_score, loss_str, global_dice_score, acc_str,
                                   val_IOU, optimizer.param_groups[0]['lr']))

                scheduler.step(val_score)
                writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], global_step)

                if options.n_classes > 1: # Changed from net.n_classes
                    writer.add_scalar('Loss/test', val_score, global_step)
                    writer.add_scalar('IOU/test', val_IOU, global_step)
                else:
                    writer.add_scalar('Dice/test', val_score, global_step)

                if (options.data_name == 'CamVid') or (options.data_name == 'Cityscapes'):
                    im = torch.stack([unorm(x_) for x_ in imgs])
                else:
                    im = imgs
                writer.add_images('images', im, global_step)

                if options.n_classes == 1:  # Changed from net.n_classes
                    writer.add_images('masks/true', true_masks, global_step)
                    writer.add_images('masks/pred', torch.sigmoid(masks_pred) > 0.5, global_step)
                elif options.data_name == 'CamVid':
                    target_plot = Cam_LabelTensorToPILImage()
                    writer.add_images('masks/true', target_plot(true_masks), global_step)
                    writer.add_images('masks/pred', target_plot(torch.argmax(masks_pred, dim=1)), global_step)
                elif options.data_name == 'Cityscapes':
                    target_plot = City_LabelTensorToPILImage()
                    writer.add_images('masks/true', target_plot(true_masks), global_step)
                    writer.add_images('masks/pred', target_plot(torch.argmax(masks_pred, dim=1)), global_step)

                # saving the checkpoint model if model improved
                if improved:
                    try:
                        os.mkdir(dir_checkpoint)
                        log_string('Created checkpoint directory')
                    except OSError:
                        pass
                    try:
                        os.mkdir(dir_checkpoint + '/' + dt_save + file_naming + '/')
                        log_string('Created architecture model checkpoint directory')
                    except OSError:
                        pass

                    # info = {'loss': val_score,
                    #         'accuracy': val_IOU}
                    # for tag, value in info.items():
                    #     test_logger.scalar_summary(tag, value, global_step)

                    state_dict = net.state_dict()
                    for key in state_dict.keys():
                        state_dict[key] = state_dict[key].cpu()
                    save_path = dir_checkpoint + '/' + dt_save + file_naming + '/'
                    save_path = os.path.join(save_path, '{}.ckpt'.format(global_step))
                    torch.save({
                        'global_step': global_step,
                        'loss': val_score,
                        'acc': val_IOU,
                        'save_dir': dir_checkpoint,
                        'state_dict': state_dict},
                        save_path)
                    log_string('Model saved at: {}'.format(save_path))
                log_string('--' * 30)
                start_time = time.time()

    writer.close()


if __name__ == '__main__':
    cudnn.deterministic = True
    cudnn.benchmark = False

    # ensure reproducibility
    torch.manual_seed(options.random_seed)
    kwargs = {}
    if torch.cuda.is_available():
        torch.cuda.manual_seed(options.random_seed)
        kwargs = {'num_workers': options.workers, 'pin_memory': True}

    ##################################
    # Initialize logs directory
    ##################################
    dt_save = datetime.now().strftime('%Y_%m_%d_%H%M%S')
    logs_dir = os.path.join(options.logs_dir, dt_save)
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    save_log = logs_dir + file_naming + '/'
    if not os.path.exists(save_log):
        os.makedirs(save_log)
    save_log = logs_dir + file_naming + '/'
    LOG_FOUT = open(os.path.join(save_log, 'log_train.txt'), 'w')

    ##################################
    # Initialize the model
    ##################################
    if options.n_classes > 1:
        if options.data_name == 'CamVid':
            class_weights = torch.FloatTensor(cam_class_weight).cuda()
            criterion = nn.CrossEntropyLoss(weight=class_weights)
        elif options.data_name == 'Cityscapes':
            class_weights = torch.FloatTensor(list(CITYSCAPES_LABEL_WEIGHTS.values())).cuda()
            criterion = nn.CrossEntropyLoss(weight=class_weights)
        else:
            criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.BCEWithLogitsLoss()

    if options.arch =='UNet':
        net = UNet(n_channels=options.n_channels, n_classes=options.n_classes, bilinear=False)
    elif options.arch == 'SegCaps':
        net = SegCaps(n_channels=options.n_channels, n_classes=options.n_classes, bilinear=False)
    elif options.arch == 'Tiramisu57':
        net = FCDenseNet57(n_channels=options.n_channels, n_classes=options.n_classes, bilinear=False)
    elif options.arch == 'Tiramisu67':
        net = FCDenseNet67(n_channels=options.n_channels, n_classes=options.n_classes, bilinear=False)
    elif options.arch == 'Tiramisu103':
        net = FCDenseNet103(n_channels=options.n_channels, n_classes=options.n_classes, bilinear=False)
    elif options.arch == 'WCaps':
        net = WCaps(n_channels=options.n_channels, n_classes=options.n_classes, bilinear=False, args=options)
    elif options.arch == 'PanDeepLabx71':
        net = DeeperX71(n_classes=options.n_classes, criterion=criterion, s2s4=True)
    elif options.arch == 'DeepLabv3+x71':
        net = DeepV3PlusX71(n_classes=options.n_classes, criterion=criterion)
    elif options.arch == 'HRNetM':
        net = HRNet_Mscale(n_classes=options.n_classes, criterion=criterion, args=options)
    elif options.arch == 'efficientnet-b2':
        net = get_efficientunet_b2(out_channels=options.n_classes, pretrained=False)

    log_string('Model Generated.')
    log_string("Number of parameters: {}".format(sum(param.numel() for param in net.parameters())))

    ##################################
    # Use cuda
    ##################################
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.to(device)
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        net = nn.DataParallel(net)

    ##################################
    # Optimizer and Loss
    ##################################
    if options.optimizer == 'Adam':
        optimizer = optim.Adam(net.parameters(), lr=options.lr, betas=(options.beta1, options.beta2), eps=1e-8,
                               weight_decay=options.weight_decay, amsgrad=False)
    else:
        optimizer = optim.SGD(net.parameters(), lr=options.lr, momentum=options.beta1, weight_decay=options.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min' if options.n_classes > 1 else 'max', patience=100)

    ##################################
    # Load dataset
    ##################################
    train_loader, val_loader, val_shape = get_train_valid_loader(options.data_dir, options.data_name,
                                                                 options.batch_size, shuffle=True,
                                                                 num_workers=options.workers, pin_memory=True)

    ##################################
    # TRAINING/TESTING
    ##################################
    log_string('')
    log_string('MODEL NAME: ' + dt_save + file_naming)
    log_string('Start training: Total epochs: {}, Batch size: {}, Training size: {}, Validation size: {}'.
               format(options.epochs, options.batch_size, len(train_loader.dataset), len(val_loader.dataset)))
    # train_logger = Logger(os.path.join(save_log, 'train'))
    # test_logger = Logger(os.path.join(save_log, 'test'))
    val_freq = int(np.ceil(len(train_loader.dataset))/options.batch_size)

    if options.load:
        net.load_state_dict(
            torch.load(options.load_model_path, map_location=device)
        )
        log_string(f'Model loaded from {options.load_model_path}')

    try:
        train_net()

    except KeyboardInterrupt:
        # torch.save(net.state_dict(), dir_checkpoint + dt_save + file_naming + '/' + 'INTERRUPTED.pth')
        log_string('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
