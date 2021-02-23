import numpy as np

from .config import options
import torch
from . import camvid
from . import joint_transforms
from .dataset import BasicDataset
from torchvision import datasets
from torchvision import transforms
from .cityscapes import get_cityscapes_loaders
from torch.utils.data import Subset, DataLoader
from .HuBMAPCropDataset import HuBMAPCropDataset


def get_train_valid_loader(data_dir, dataset, batch_size, shuffle=True, num_workers=4, pin_memory=True):
    data_dir = data_dir + '/' + dataset
    train_loader, valid_loader = None, None

    if dataset == "CamVid":
        normalize = transforms.Normalize(mean=camvid.cam_mean, std=camvid.cam_std)
        train_joint_transformer = transforms.Compose([
            joint_transforms.JointRandomSizedCrop(288),
            joint_transforms.JointRandomHorizontalFlip()
        ])
        val_joint_transformer = transforms.Compose([
            joint_transforms.JointScale(288)
        ])
        train_dataset = camvid.CamVid(data_dir, 'train',
                                      joint_transform=train_joint_transformer,
                                      transform=transforms.Compose([
                                          transforms.ToTensor(),
                                          normalize]))
        valid_dataset = camvid.CamVid(data_dir, 'val',
                                      joint_transform=val_joint_transformer,
                                      transform=transforms.Compose([
                                          transforms.ToTensor(),
                                          normalize]))

        if options.arch == 'DeepLab3':
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=options.batch_size, shuffle=True, drop_last=True)
            valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=options.batch_size, shuffle=True, drop_last=True)
        else:
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=options.batch_size, shuffle=True)
            valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=options.batch_size, shuffle=True)

    elif dataset == "GlomSeg":
        train_set = np.array([])
        val_set = np.load(data_dir + '/' + 'seg_fold_{}.npz'.format(options.val_int))['FILES']
        for i in range(5):
            if i == options.val_int:
                continue
            else:
                train_set = np.append(train_set, np.load(data_dir + '/' + 'seg_fold_{}.npz'.format(i))['FILES'])

        train_dataset = BasicDataset(data_dir + '/' + 'imgs/', data_dir + '/' + 'masks/', options.scale, train_set, train=True)
        val_dataset = BasicDataset(data_dir + '/' + 'imgs/', data_dir + '/' + 'masks/', options.scale, val_set, train=False)
        train_loader = DataLoader(train_dataset, batch_size=options.batch_size, shuffle=True, num_workers=options.workers, pin_memory=True)
        valid_loader = DataLoader(val_dataset, batch_size=options.batch_size, shuffle=True, num_workers=options.workers, pin_memory=True, drop_last=True)

    elif dataset == "Cityscapes":
        train_loader, valid_loader, _ = get_cityscapes_loaders(data_dir, image_shape=options.cs_imsize, labels_as_onehot=False,
                           include_coarse_dataset=options.cs_coarse, read_from_zip_archive=False,
                           train_batch_size=options.batch_size, validate_batch_size=options.batch_size,
                           test_batch_size=options.batch_size, num_workers=options.workers)

    elif dataset == "KaggleCrops":
        train_dataset = HuBMAPCropDataset(data_dir, mode="train")
        train_loader = DataLoader(train_dataset, batch_size=options.batch_size,
                                  shuffle=True, num_workers=options.workers, drop_last=False)

        val_dataset = HuBMAPCropDataset(data_dir, mode="val")
        valid_loader = DataLoader(val_dataset, batch_size=options.batch_size,
                                shuffle=False, num_workers=options.workers, drop_last=False)

    return train_loader, valid_loader


def get_test_loader(data_dir,
                    dataset,
                    batch_size,
                    num_workers=4,
                    pin_memory=False):

    data_dir = data_dir + '/' + dataset

    if dataset == "CamVid":
        normalize = transforms.Normalize(mean=camvid.cam_mean, std=camvid.cam_std)
        test_dataset = camvid.CamVid(data_dir, 'test',
                                      joint_transform=None,
                                      transform=transforms.Compose([
                                          transforms.ToTensor(),
                                          normalize]))

        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=options.batch_size, shuffle=True)

    elif dataset == "GlomSeg":
        # TODO: generate a glomseg test set
        test_set = np.load(data_dir + '/' + 'seg_fold_{}.npz'.format(options.val_int))['FILES']
        test_dataset = BasicDataset(data_dir + '/' + 'imgs', data_dir + '/' + 'masks', options.img_scale, test_set, train=False)
        test_loader = DataLoader(test_dataset, batch_size=options.batch_size, shuffle=True, num_workers=options.workers, pin_memory=True, drop_last=True)

    elif dataset == "Cityscapes":
        _, _, test_loader = get_cityscapes_loaders(data_dir, image_shape=(256, 512), labels_as_onehot=False,
                           include_coarse_dataset=False, read_from_zip_archive=False,
                           train_batch_size=options.batch_size, validate_batch_size=options.batch_size,
                           test_batch_size=options.batch_size, num_workers=options.workers)

    return test_loader

