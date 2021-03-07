from optparse import OptionParser

parser = OptionParser()

# General settings
parser.add_option('--ar', '--arch', dest='arch', type=str, default='efficientnet-b3',
                    help='The desired segmentation architecture. Input either '
                         'UNet, Tiramisu57/67/103, SegCaps, WCaps, DeepLabv3+x71,'
                         'PanDeepLabx71, or HRNetM')
parser.add_option('--gpu', '--GPU', dest='gpu', type=str, default='3',
                    help='Select cuda device.')
parser.add_option('--df', '--disp_freq', dest='disp_freq', default=45, type='int',
                  help='frequency of train logging (default: 50)')
parser.add_option('--vf', '--val_freq', dest='val_freq', default=90, type='int',
                  help='frequency of val evaluation (default: 2200)')
parser.add_option('--lb', '--load_bool', dest='load', default=False,
                  help='Initialize pretrained model (default False)')
parser.add_option('--lp', '--load_model_path', dest='load_model_path',
                  default='./checkpoints/2021_02_24_212820_KaggleCrops_efficientnet-b2_Adam_LR0.0003_BS16/18990.ckpt',
                  help='path to load a .ckpt model')
parser.add_option('-e', '--epochs', dest='epochs', default=200, type='int',
                  help='number of epochs (default: 500)')
parser.add_option('-b', '--batch-size', dest='batch_size', default=4, type='int',
                  help='batch size (default: 2)')

# GlomSeg Settings
parser.add_option('-s', '--scale', dest='scale', type=float, default=0.1,
                  help='Downscaling factor of the images in GlomSeg')
parser.add_option('-v', '--validation', dest='val_int', type=int, default=0,
                  help='Select the validation fold for GlomSeg.')
parser.add_option('--fo', '--folds', dest='folds', type=int, default=5,
                  help='Number of folds in GlomSeg.')

# WCaps settings
parser.add_option('--mod', '--mod', dest='modules', default=1, type='int',
                  help='Number of encoder/decoder modules (default 3)')
parser.add_option('--eo', '--eo', dest='enc_ops', default=5, type='int',
                  help='Number of encoder ops per branch (default 4)')
parser.add_option('--sc', '--sc', dest='wscales', default=False,
                  help='Include scale rep capsule in final operation (default 4)')
parser.add_option('--P', '--P', dest='P', default=8, type='int',
                  help='Number of features per capsule (P^2)')

# HRNetM settings
parser.add_option('--ocm', '--ocm', dest='ocr_mid_channels', default=512, type='int',
                  help='Channel count, OCR mid')
parser.add_option('--ock', '--ock', dest='ocr_key_channels', default=256, type='int',
                  help='Channel count, OCR key')
parser.add_option('--id', '--id', dest='init_decoder', default=False,
                  help='Initialize decoder weights')
parser.add_option('--oar', '--oar', dest='ocr_aux_loss_rmi', default=False,
                  help='Enable rmi on auxillary loss')
parser.add_option('--oa', '--oa', dest='ocr_alpha', default=0.4, type='float',
                  help='Auxillary loss multiplier')
parser.add_option('--mls', '--mls', dest='mscale_lo_scale', default=0.5, type='float',
                  help='low res training scale')
parser.add_option('--smsw', '--smsw', dest='supervised_mscale_weight', default=0, type='float',
                  help='weight for supervised scales')
parser.add_option('--ns', '--ns', dest='n_scales', default=[0.5, 1.0, 2.0],
                  help='what scales to evaluate with')
parser.add_option('--ac', '--ac', dest='align_corners', default=True,
                  help='Whether or not to align corners')
parser.add_option('--sbc', '--sbc', dest='seg_bot_ch', default=256, type='int',
                  help='seg attention bot channels')
parser.add_option('--moa', '--moa',  dest='mscale_old_arch',
                  default=False, help='use old attention head')
parser.add_option('--msi3', '--msi3',  dest='mscale_inner_3x3',
                  default=False, help='use old attention head')
parser.add_option('--dob', '--dob',  dest='mscale_dropout',
                  default=False, help='use dropout')
parser.add_option('--oaspp', '--oaspp',  dest='ocr_aspp',
                  default=False, help='configure atrous pyramid')
parser.add_option('--oasppbc', '--oasppbc',  dest='ocr_aspp_bot_ch',
                  default=256, type='int', help='set channel count for aspp')

# Pytorch version
parser.add_option('--ptv', '--ptv', dest='pytorch_version', default=1.7, type='float',
                  help='pytorch version for interpolation')

# For data
parser.add_option('--nc', '--num_classes', dest='n_classes', default=1, type='int',
                  help='number of classes (default: 12 camvid, 20 cityscapes)')
parser.add_option('--nch', '--num_channels', dest='n_channels', default=3, type='int',
                  help='number of channels in input data (default: 3)')
parser.add_option('--ddir', '--data_dir', dest='data_dir', default='./data',
                  help='Directory in which data is stored (default: ./data)')
parser.add_option('--dn', '--data_name', dest='data_name', default='KaggleCrops',
                  help='Cityscapes, GlomSeg, CamVid (default: Cityscapes)')
parser.add_option('--rs', '--random_seed', dest='random_seed', default=2018, type='int',
                  help='Seed to ensure reproducibility (default: 2018)')
parser.add_option('-j', '--workers', dest='workers', default=16, type='int',
                  help='number of subprocesses to use for data loading (default: 16)')

# For Cityscapes
parser.add_option('--csc', '--csc', dest='cs_coarse', default=True,
                  help='Include coarse data in training (default False)')
parser.add_option('--is', '--is', dest='cs_imsize', default=(256, 512),
                  help='Set default image size')

# For optimizer/scheduler
parser.add_option('--lr', '--lr', dest='lr', default=0.0003, type='float',
                  help='learning rate (default: 0.001)')
parser.add_option('--wd', '--weight_decay', dest='weight_decay', default=5e-4, type='float',
                  help='weight decay (default: 5e-4)')
parser.add_option('--beta1', '--beta1', dest='beta1', default=0.9, type='float',
                  help='beta 1 for Adam/SGD optimizer (default: 0.9)')
parser.add_option('--beta2', '--beta2', dest='beta2', default=0.999, type='float',
                  help='beta 2 for Adam optimizer (default: 0.999)')
parser.add_option('--pat', '--pat', dest='patience', default=100, type='int',
                  help='patience for LR reduction')
parser.add_option('--opt', '--opt', dest='optimizer', default='Adam', type=str,
                  help='Select your optimizer (Adam, SGD)')

# For CapsNet
parser.add_option('--nh', '--num_heads', dest='num_heads', default=1, type='int',
                  help='number of attention heads for TR (default: 1)')

# For SE caps
parser.add_option('--SE', '--SE', dest='SE', default=False,
                  help='Whether to test Capsule Squeeze Excitation (default = False)')

# For Kaggle
parser.add_option('--kaggle-data-path', dest='kaggle_data_path',
                  default='/home/cougarnet.uh.edu/srizvi7/Desktop/data/public/Kaggle_HuBMAP',
                  help='Path to base directory of Kaggle datasets')
parser.add_option('--loo', '--leave-one-out', dest='test_tiff_value', default=1,  # smallest tiff
                  help='Tiff file to remove from train and add to validation set for Kaggle dataset')
parser.add_option('--ke', '--kaggle-eval', dest='kaggle_eval', default=True,
                  help='Use Kaggle evaluation on global dice image')
parser.add_option('--tw', '--train-window', dest='train_window', default=512,
                  help='Train on images on this size')
parser.add_option('--vw', '--val-window', dest='val_window', default=512,
                  help='Validate on images of this size')
parser.add_option('--tt', '--test-window', dest='test_window', default=512,
                  help='Test on images on this size')

parser.add_option('--th', '--threshold', dest='threshold', default=0.35,
                  help='Threshold of mask to convert values to 0/1 in the mask')  # Notebook used 0.39
parser.add_option('--y', '--y-shift', dest='y_shift', default=-42, type='int',
                  help='Amount to shift global mask in y-direction')
parser.add_option('--x', '--x-shift', dest='x_shift', default=-20, type='int',
                  help='Amount to shift global mask in x-direction')

# For save and loading
parser.add_option('--sd', '--save-dir', dest='save_dir', default='./checkpoints',
                  help='saving directory of .ckpt models (default: ./checkpoints)')
parser.add_option('--ld', '--logs-dir', dest='logs_dir', default='./logs',
                  help='saving directory of logs files (default: ./logs)')
parser.add_option('--rd', '--runs-dir', dest='runs_dir', default='./runs',
                  help='saving directory of tb files (default: ./runs)')

options, _ = parser.parse_args()
