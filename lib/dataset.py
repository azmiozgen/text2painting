import random
import sys
from io import BytesIO

import lmdb
import numpy as np
import torch
from PIL import Image, ImageFilter
from torch.utils.data import Dataset

from .utils import ImageUtilities


class ClsDataset(Dataset):
    """Dataset Reader"""

    def __init__(self, _lmdb_path):

        self._lmdb_path = _lmdb_path

        self.env = lmdb.open(self._lmdb_path, max_readers=1, readonly=True,
                             lock=False, readahead=False, meminit=False)

        if not self.env:
            print('Cannot read lmdb from {}'.format(self._lmdb_path))
            sys.exit(0)

        with self.env.begin(write=False) as txn:
            self.n_samples = int(txn.get('num-samples'.encode()))

    def __load_data(self, index):

        with self.env.begin(write=False) as txn:
            image_key = 'image-%09d' %(index + 1)
            label_key = 'label-%09d' %(index + 1)

            label = txn.get(label_key.encode())
            label = label.decode()

            img = txn.get(image_key.encode())
            img = Image.open(BytesIO(img)).convert('RGB')

        return img, label

    def __getitem__(self, index):

        assert index <= len(self), 'index range error'

        image, label = self.__load_data(index)

        return image, label

    def __len__(self):
        return self.n_samples

def pad_image(img):

    w, h = img.size

    pad_h = int(np.random.rand() * h / 2.0) + 1
    pad_w = int(np.random.rand() * w / 2.0) + 1

    new_size = (w + pad_w, h + pad_h)
    offset = (np.random.randint(pad_w), np.random.randint(pad_h))

    bg_colors =tuple( np.random.randint(256, size=3).astype(np.uint8))

    bg = Image.new("RGB", new_size, bg_colors)
    bg.paste(img, offset)

    return bg

def crop_edges_lr(img):

    w, h = img.size

    crop_l = int(np.random.rand() * w / 4.0) + 1
    crop_r = int(np.random.rand() * w / 4.0) + 1

    img = np.array(img)
    img = img[:, crop_l:-crop_r]
    img = Image.fromarray(img)
    return img

class AlignCollate(object):
    """Should be a callable (https://docs.python.org/2/library/functions.html#callable), that gets a minibatch
    and returns minibatch."""

    def __init__(self, mode, labels, mean, std,
                 image_size_height, image_size_width,
                 horizontal_flipping=True, random_rotation=True,
                 color_jittering=True, random_grayscale=True,
                 random_channel_swapping=True, random_gamma=True,
                 random_resolution=True):

        self._mode = mode

        assert self._mode in ['training', 'test']

        self.label_mapping = labels
        self.n_classes = len(self.label_mapping)
        self.mean = mean
        self.std = std
        self.image_size_height = image_size_height
        self.image_size_width = image_size_width

        self.horizontal_flipping = horizontal_flipping
        self.random_rotation = random_rotation
        self.color_jittering = color_jittering
        self.random_grayscale = random_grayscale
        self.random_channel_swapping = random_channel_swapping
        self.random_gamma = random_gamma
        self.random_resolution = random_resolution

        if self._mode == 'training':
            if self.random_resolution:
                self.random_res = ImageUtilities.image_random_resolution([0.7, 1.3])

            if self.horizontal_flipping:
                self.horizontal_flipper = ImageUtilities.image_random_horizontal_flipper()

            if self.random_rotation:
                self.random_rotator = ImageUtilities.image_random_rotator(interpolation=Image.BILINEAR, random_bg=True)

            if self.color_jittering:
                self.color_jitter = ImageUtilities.image_random_color_jitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.15)

            if self.random_gamma:
                self.random_gamma_adjuster = ImageUtilities.image_random_gamma(gamma_range=[0.7, 1.3], gain=1)

            if self.random_channel_swapping:
                self.channel_swapper = ImageUtilities.image_random_channel_swapper(p=0.5)

            if self.random_grayscale:
                self.grayscaler = ImageUtilities.image_random_grayscaler(p=0.5)

        self.resizer = ImageUtilities.image_resizer(self.image_size_height, self.image_size_width)
        self.normalizer = ImageUtilities.image_normalizer(self.mean, self.std)

    def __preprocess(self, image, label):

        if self._mode == 'training':

            image = image.filter(ImageFilter.GaussianBlur(radius=np.random.rand() * 2.0))
            if np.random.rand() < 0.5:
                image = pad_image(image)
            else:
                image = crop_edges_lr(image)

            if self.random_resolution:
                image = self.random_res(image)

            if self.horizontal_flipping:
                is_flip = random.random() < 0.5
                image = self.horizontal_flipper(image, is_flip)

            if self.random_rotation:
                rot_angle = int(np.random.rand() * 10)
                if np.random.rand() >= 0.5:
                    rot_angle = -1 * rot_angle
                rot_expand = np.random.rand() < 0.5
                image = self.random_rotator(image, rot_angle, rot_expand)

            if self.color_jittering:
                image = self.color_jitter(image)

            if self.random_gamma:
                image = self.random_gamma_adjuster(image)

            if self.random_channel_swapping:
                image = self.channel_swapper(image)

            if self.random_grayscale:
                image = self.grayscaler(image)

        image = self.resizer(image)
        image = self.normalizer(image)

        label = np.where(self.label_mapping == label)[0][0]

        return image, label

    def __call__(self, batch):
        images, labels = list(zip(*batch))
        images = list(images)
        labels = list(labels)

        bs = len(images)
        for i in range(bs):
            image, label = self.__preprocess(images[i], labels[i])
            images[i] = image
            labels[i] = label

        images = torch.stack(images)
        labels = torch.LongTensor(np.array(labels))

        return images, labels
