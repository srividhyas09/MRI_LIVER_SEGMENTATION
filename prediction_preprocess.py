import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import pydicom
import monai.transforms
import cv2


class Preprocessing(Dataset):

    """ Dataset for XR wrist images"""
    def __init__(self, data_path, batch_size=1, transform=None):

        self.data_path = data_path
        self.transform = transform
        self.batch_size = batch_size
        self.label_paths = pd.read_csv(data_path)

    def __len__(self):
        return (np.ceil(len(self.label_paths) / float(self.batch_size))).astype(np.int)

    def __getitem__(self, idx):
        # Load labels
        image_path = self.label_paths['image'][idx]
        image_path_2 = self.label_paths['gt'][idx]

        image = pydicom.read_file(image_path)
        ps1 = 1 / np.round(image.PixelSpacing, 2)
        obj1 = monai.transforms.Spacing(ps1)

        gt = image.pixel_array
        image = image.pixel_array

        gt = cv2.imread(image_path_2, cv2.IMREAD_GRAYSCALE)

        image = image.reshape(1, image.shape[0], image.shape[1])
        gt = gt.reshape(1, gt.shape[0], gt.shape[1])

        image = obj1(image)[0]
        gt = obj1(gt)[0]
        # Return sample
        sample = {'image': image, 'mask': gt}

        if self.transform:
            sample = self.transform(sample)

        return sample


class ResizePadCrop(object):

    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']

        pad = monai.transforms.ResizeWithPadOrCrop([300, 300])

        img = pad(image).squeeze(0)
        mask = pad(mask).squeeze(0)

        return {'image': img, 'mask': mask}


class ToTensor(object):

    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']

        if len(image.shape) == 2:
            image = image.reshape(image.shape[0], image.shape[1], 1)

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))

        if (len(mask.shape) == 2):
            mask = mask.reshape(mask.shape[0], mask.shape[1], 1)
        mask = mask.transpose((2, 0, 1))

        return {'image': torch.from_numpy(image),
                'mask': torch.from_numpy(mask.astype(int))}


class Normalization(object):

    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']

        image[image > 900] = 0
        image[image < 300] = 0
        image = cv2.medianBlur(image, 5)

        max = image.max()
        image = (image / max) * 255

        return {'image': image, 'mask': mask}


class min_max_norm(object):

    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']

        image_2d_scaled = (np.maximum(image, 0) / image.max()) * 255.0
        img = np.uint8(image_2d_scaled)

        hist, _ = np.histogram(img, bins=np.arange(0, 256))
        p_low = shade_at_percentile(hist, .1)
        p_high = shade_at_percentile(hist, .9)
        a = 255.0 / (p_high - p_low)
        b = -1.0 * a * p_low
        result = (img.astype(float) * a) + b
        image = result.clip(0, 255.0)

        return {'image': image, 'mask': mask}

def shade_at_percentile(hist, percentile):
    n = np.sum(hist)
    cumulative_sum = np.cumsum(hist)
    return np.argmax(cumulative_sum / n >= percentile)


