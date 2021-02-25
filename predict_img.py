import argparse
import logging
import os
from os.path import splitext
from os import listdir
import imagesize

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from models import UNet
from utils.data_vis import plot_img_and_mask
from utils.dataset import BasicDataset

img_dir = 'data/GlomSeg/imgs/'
mask_dir = 'data/GlomSeg/masks/'
dir_checkpoint = 'checkpoints/'


def predict_img(net,
                full_img,
                device,
                scale_factor=1,
                out_threshold=0.5):
    net.eval()

    ids = [splitext(file)[0] for file in listdir(img_dir)
                if not file.startswith('.')]
    new_width, new_height = 0, 0
    for i, id in enumerate(ids):
        width, height = imagesize.get(img_dir + id + '.png')
        if (width > new_width):
            new_width = width
        if (height > new_height):
            new_height = height

    img = torch.from_numpy(BasicDataset.preprocess(full_img, scale_factor, new_width, new_height))

    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img)

        if net.n_classes > 1:
            probs = F.softmax(output, dim=1)
        else:
            probs = torch.sigmoid(output)

        probs = probs.squeeze(0)

        tf = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(full_img.size[1]),
                transforms.ToTensor()
            ]
        )

        probs = tf(probs.cpu())
        full_mask = probs.squeeze().cpu().numpy()

    return full_mask > out_threshold


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', '-m', default='MODEL.pth',
                        metavar='FILE',
                        help="Specify the file in which the model is stored")
    parser.add_argument('--input', '-i', metavar='INPUT', nargs='+',
                        help='filenames of input images', required=False)
    parser.add_argument('--output', '-o', metavar='INPUT', nargs='+',
                        help='Filenames of ouput images')
    parser.add_argument('--viz', '-v', action='store_true',
                        help="Visualize the images as they are processed",
                        default=False)
    parser.add_argument('--no-save', '-n', action='store_true',
                        help="Do not save the output masks",
                        default=False)
    parser.add_argument('--mask-threshold', '-t', type=float,
                        help="Minimum probability value to consider a mask pixel white",
                        default=0.5)
    parser.add_argument('--scale', '-s', type=float,
                        help="Scale factor for the input images",
                        default=0.5)
    parser.add_argument('--plot_val', '-pv', type=bool, default=True,
                        help="Plot all validation set images")
    parser.add_argument('-val', '--validation', dest='val_int', type=int, default=0,
                        help='Select the validation fold.')

    return parser.parse_args()


def get_output_filenames(args):
    in_files = args.input
    out_files = []

    if not args.output:
        for f in in_files:
            pathsplit = os.path.splitext(f)
            out_files.append("{}_OUT{}".format(pathsplit[0], pathsplit[1]))
    elif len(in_files) != len(args.output):
        logging.error("Input files and output files are not of the same length")
        raise SystemExit()
    else:
        out_files = args.output

    return out_files


def mask_to_image(mask):
    return Image.fromarray((mask * 255).astype(np.uint8))


if __name__ == "__main__":
    args = get_args()

    if args.plot_val:
        val_set = np.load('seg_fold_{}.npz'.format(args.val_int))['FILES']
        in_files = [file.split('/')[-1] for file in val_set
                         if not file.startswith('.')]
    else:
        in_files = args.input
        out_files = get_output_filenames(args)

    net = UNet(n_channels=3, n_classes=1)

    logging.info("Loading model {}".format(args.model))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    net.to(device=device)
    net.load_state_dict(torch.load(dir_checkpoint + args.model, map_location=device))

    logging.info("Model loaded !")

    for i, fn in enumerate(in_files):
        logging.info("\nPredicting image {} ...".format(fn))

        img = Image.open(img_dir + fn)
        orig_mask = Image.open(mask_dir + fn)

        mask = predict_img(net=net,
                           full_img=img,
                           scale_factor=args.scale,
                           out_threshold=args.mask_threshold,
                           device=device)

        # If you want to save the mask:
        # if not args.no_save:
        #     out_fn = out_files[i]
        #     result = mask_to_image(mask)
        #     result.save(out_files[i])
        #
        #     logging.info("Mask saved to {}".format(out_files[i]))

        if args.plot_val:
            logging.info("Visualizing results for image {}, close to continue ...".format(fn))
            plot_img_and_mask(img, mask, orig_mask, 'all_plots/' + str(args.model).split('.pth')[0] + '_val' + str(args.val_int) + '/', fn, single=False)
            print('Plot {} generated.'.format(i))
        else:
            logging.info("Visualizing results for image {}, close to continue ...".format(fn))
            plot_img_and_mask(img, mask, orig_mask, args.output, args.input, single=True)
