from optparse import OptionParser


parser = OptionParser()

parser.add_option('-g', '--gpu', dest='gpu', default=3, help='GPU to run on')

parser.add_option('-e', '--epochs', dest='epochs', default=150, type='int',
                  help='number of epochs (default: 100)')
parser.add_option('-b', '--batch-size', dest='batch_size', default=16, type='int',
                  help='batch size (default: 16)')
parser.add_option('--df', '--disp_freq', dest='disp_freq', default=150, type='int',
                  help='frequency of displaying the training results (default: 100)')
parser.add_option('--vf', '--val_freq', dest='val_freq', default=300, type='int',
                  help='run validation for each <val_freq> iterations (default: 2000)')
parser.add_option('-j', '--workers', dest='workers', default=0, type='int',
                  help='number of data loading workers (default: 16)')

# For data
parser.add_option('--dn', '--data_path', dest='data_path', default='./',
                  help='path to dataset')
parser.add_option('--ih', '--img_h', dest='img_h', default=256, type='int',
                  help='input image height (default: 256)')
parser.add_option('--iw', '--img_w', dest='img_w', default=256, type='int',
                  help='input image width (default: 256)')
parser.add_option('--ic', '--img_c', dest='img_c', default=3, type='int',
                  help='number of input channels (default: 3)')

# Classes
parser.add_option('--nc', '--num_classes', dest='num_classes', default=1, type='int',
                  help='number of classes (default: 5)')

# For model
parser.add_option('--m', '--model', dest='model', default='efficientnet-b2',
                  help='vgg, inception, resnet, densenet (default: resnet)')
parser.add_option('--lr', '--lr', dest='lr', default=0.001, type='float',
                  help='learning rate(default: 0.001)')
parser.add_option('--lm', '--load_model', dest='load_model',
                  default='/home/cougarnet.uh.edu/sdpatiba/Desktop/Kaggle_2021_HuBMAP/save/20210220_171241/models/15544.ckpt',
                  help='Path to load the model')


# For directories
parser.add_option('--sd', '--save-dir', dest='save_dir', default='./save',
                  help='saving directory of .ckpt models (default: ./save)')

parser.add_option('--loo', '--leave-one-out', dest='test_tiff_value', default=0,
                  help='Tiff file to remove from train and add to validation set')

# Mask options
parser.add_option('--th', '--threshold', dest='threshold', default=0.39,
                  help='Threshold of mask to convert values to 0/1 in the mask')

parser.add_option('--y', '--y-shift', dest='y_shift', default=-40, type='int',
                  help='Amount to shift global mask in y-direction')

parser.add_option('--x', '--x-shift', dest='x_shift', default=-24, type='int',
                  help='Amount to shift global mask in x-direction')

# Window sizes for train, val, and test.
parser.add_option('--tw', '--train-window', dest='train_window', default=256,
                  help='Train on images on this size')

parser.add_option('--vw', '--val-window', dest='val_window', default=256,
                  help='Validate on images of this size')

parser.add_option('--tt', '--test-window', dest='test_window', default=512,
                  help='Test on images on this size')

# Augmentaion
parser.add_option('--augh', '--aughard', dest='augment_hard', default=True,
                  help='Whether to augment hard or not')


options, _ = parser.parse_args()