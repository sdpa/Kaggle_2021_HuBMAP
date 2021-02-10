from optparse import OptionParser


parser = OptionParser()

parser.add_option('-e', '--epochs', dest='epochs', default=100, type='int',
                  help='number of epochs (default: 100)')
parser.add_option('-b', '--batch-size', dest='batch_size', default=16, type='int',
                  help='batch size (default: 16)')
parser.add_option('--df', '--disp_freq', dest='disp_freq', default=100, type='int',
                  help='frequency of displaying the training results (default: 100)')
parser.add_option('--vf', '--val_freq', dest='val_freq', default=200, type='int',
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
parser.add_option('--nc', '--num_classes', dest='num_classes', default=2, type='int',
                  help='number of classes (default: 5)')

# For model
parser.add_option('--m', '--model', dest='model', default='efficientnet-b2',
                  help='vgg, inception, resnet, densenet (default: resnet)')
parser.add_option('--lr', '--lr', dest='lr', default=0.001, type='float',
                  help='learning rate(default: 0.001)')

# For directories
parser.add_option('--sd', '--save-dir', dest='save_dir', default='./save',
                  help='saving directory of .ckpt models (default: ./save)')

parser.add_option('--loo', '--leave-one-out', dest='test_tiff_value', default=0,
                  help='Tiff file to remove from train and add to validation set')


options, _ = parser.parse_args()