from os.path import splitext
from os import listdir
import numpy as np
from glob import glob
import torch
from torch.utils.data import Dataset
import logging
from PIL import Image
import imagesize
import random
import torchvision.transforms.functional as tf

class BasicDataset(Dataset):

    def __init__(self, imgs_dir, masks_dir, scale=1, fold=np.array([]), train=True):
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.scale = scale
        self.fold = fold
        self.train = train
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'

        self.ids_raw = [splitext(file)[0] for file in listdir(imgs_dir)
                    if not file.startswith('.')]

        new_width, new_height = 0, 0
        for i, ids in enumerate(self.ids_raw):
            width, height = imagesize.get(self.imgs_dir + ids + '.png')
            if (width > new_width):
                new_width = width
            if (height > new_height):
                new_height = height
        self.width = new_width
        self.height = new_height

        self.fold_raw = [splitext(file)[0].split('/')[-1] for file in fold
                         if not file.startswith('.')]
        self.ids = []
        for i, ids in enumerate(self.ids_raw):
            if ids in self.fold_raw:
                self.ids.append(self.ids_raw[i])

        logging.info(f'Creating dataset with {len(self.ids)} examples')


    def __len__(self):
        return len(self.ids)

    @classmethod
    def preprocess(cls, pil_img, scale, width, height):
        # Original implementation
        # w, h = pil_img.size
        # newW, newH = int(scale * w), int(scale * h)
        # Not respecting 16/8 div for output mask scale
        # newW, newH = int(scale * width), int(scale * height)
        newW = (int(scale * width) - (int(scale * width) % 16))
        newH = (int(scale * height) - (int(scale * height) % 16))
        assert newW > 0 and newH > 0, 'Scale is too small'
        pil_img = pil_img.resize((newW, newH))

        img_nd = np.array(pil_img)

        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2)

        # HWC to CHW
        img_trans = img_nd.transpose((2, 0, 1))
        if img_trans.max() > 1:
            img_trans = img_trans / 255

        return img_trans

    def __getitem__(self, i):
        idx = self.ids[i]
        mask_file = glob(self.masks_dir + idx + '*')
        img_file = glob(self.imgs_dir + idx + '*')

        assert len(mask_file) == 1, \
            f'Either no mask or multiple masks found for the ID {idx}: {mask_file}'
        assert len(img_file) == 1, \
            f'Either no image or multiple images found for the ID {idx}: {img_file}'

        mask = Image.open(mask_file[0])
        img = Image.open(img_file[0])

        # Image normalization
        img = np.asarray(img)
        img = (img - img.min()) / (img.max() - img.min())
        img = Image.fromarray((img * 255).astype('uint8'))

        if self.train:
            # Random horizontal flipping
            if random.uniform(0, 1) > 0.5:
                img = tf.hflip(img)
                mask = tf.hflip(mask)

            # Random vertical flipping
            if random.uniform(0, 1) > 0.5:
                img = tf.vflip(img)
                mask = tf.vflip(mask)

            # # Random rotations
            # if random.uniform(0, 1) > 0.5:
            #     rot_angle = random.randint(-90, 90)
            #     img = tf.rotate(img, rot_angle)
            #     mask = tf.rotate(mask, rot_angle)

            # Random horizontal/vertical shifts
            if random.uniform(0, 1) > 0.5:
                # For shifting
                a = 1
                b = 0
                c = random.randint(-200, 200)  # left/right (i.e. 5/-5)
                d = 0
                e = 1
                f = random.randint(-200, 200)  # up/down (i.e. 5/-5)
                img = img.transform(img.size, Image.AFFINE, (a, b, c, d, e, f))
                mask = mask.transform(mask.size, Image.AFFINE, (a, b, c, d, e, f))

        assert img.size == mask.size, \
            f'Image and mask {idx} should be the same size, but are {img.size} and {mask.size}'

        img = self.preprocess(img, self.scale, self.width, self.height)
        mask = self.preprocess(mask, self.scale, self.width, self.height)

        return {'image': torch.from_numpy(img), 'mask': torch.from_numpy(mask)}
