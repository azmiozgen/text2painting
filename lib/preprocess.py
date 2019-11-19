import collections
import math
import random

import numpy as np
from PIL import Image

try:
    import accimage
except ImportError:
    accimage = None

def _is_pil_image(img):

    if accimage is not None:
        return isinstance(img, (Image.Image, accimage.Image))
    else:
        return isinstance(img, Image.Image)

def pad_image(img):

    w, h = img.size

    pad_h = int(np.random.rand() * h / 2.0) + 1
    pad_w = int(np.random.rand() * w / 2.0) + 1

    new_size = (w + pad_w, h + pad_h)
    offset = (np.random.randint(pad_w), np.random.randint(pad_h))

    bg_colors = tuple( np.random.randint(256, size=3).astype(np.uint8))

    bg = Image.new("RGB", new_size, bg_colors)
    bg.paste(img, offset)

    return bg

def crop_edges_lr(img):

    w, _ = img.size

    crop_l = int(np.random.rand() * w / 4.0) + 1
    crop_r = int(np.random.rand() * w / 4.0) + 1

    img = np.array(img)
    img = img[:, crop_l:-crop_r]
    img = Image.fromarray(img)
    return img

def crop(img, i, j, h, w):

    if not _is_pil_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

    return img.crop((j, i, j + w, i + h))

def resize(img, size, interpolation=Image.BILINEAR):

    if not _is_pil_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))
    if not (isinstance(size, int) or (isinstance(size, collections.Iterable) and len(size) == 2)):
        raise TypeError('Got inappropriate size arg: {}'.format(size))

    if isinstance(size, int):
        w, h = img.size
        if (w <= h and w == size) or (h <= w and h == size):
            return img
        if w < h:
            ow = size
            oh = int(size * h / w)
            return img.resize((ow, oh), interpolation)
        else:
            oh = size
            ow = int(size * w / h)
            return img.resize((ow, oh), interpolation)
    else:
        return img.resize(size[::-1], interpolation)

def resized_crop(img, i, j, h, w, size, interpolation=Image.BILINEAR):

    assert _is_pil_image(img), 'img should be PIL Image'
    img = crop(img, i, j, h, w)
    img = resize(img, size, interpolation)
    return img

class RandomResizedCrop(object):

    def __init__(self, size_height, size_width, interpolation=Image.BILINEAR):
        self.size = (size_height, size_width)
        self.interpolation = interpolation

    @staticmethod
    def get_params(img, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.)):

        for _ in range(10):
            area = img.size[0] * img.size[1]
            target_area = random.uniform(*scale) * area
            aspect_ratio = random.uniform(*ratio)

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if random.random() < 0.5:
                w, h = h, w

            if w <= img.size[0] and h <= img.size[1]:
                i = random.randint(0, img.size[1] - h)
                j = random.randint(0, img.size[0] - w)
                return i, j, h, w

        # Fallback
        w = min(img.size[0], img.size[1])
        i = (img.size[1] - w) // 2
        j = (img.size[0] - w) // 2
        return i, j, w, w

    def __call__(self, img, params):

        i, j, h, w = params
        return resized_crop(img, i, j, h, w, self.size, self.interpolation)


### HORIZONTAL FLIPPING ###

def hflip(img):

    if not _is_pil_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

    return img.transpose(Image.FLIP_LEFT_RIGHT)

class RandomHorizontalFlip(object):

    def __call__(self, img, flip):
        if flip:
            return hflip(img)
        return img

### RANDOM ROTATION ###

def rotate(img, angle, resample=Image.BILINEAR, expand=True):

    return img.rotate(angle, resample=resample, expand=expand)

def rotate_with_random_bg(img, angle, resample=Image.BILINEAR, expand=True):

    if not _is_pil_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

    img_np = np.array(img)

    img = img.convert('RGBA')
    img = rotate(img, angle, resample=resample, expand=expand)

    key = np.random.choice([0, 1, 2, 3])
    if key == 0:
        bg = Image.new('RGBA', img.size, (255, ) * 4)    # White image
    elif key == 1:
        bg = Image.new('RGBA', img.size, (0, 0, 0, 255)) # Black image
    elif key == 2:
        mean_color = list(map(int, img_np.mean((0, 1))))
        bg = Image.new('RGBA', img.size, (mean_color[0], mean_color[1], mean_color[2], 255)) # Mean
    elif key == 3:
        median_color = list(map(int, np.median(img_np, (0, 1))))
        bg = Image.new('RGBA', img.size, (median_color[0], median_color[1], median_color[2], 255)) # Median

    img = Image.composite(img, bg, img)
    img = img.convert('RGB')

    return img

class RandomRotate(object):

    def __init__(self, interpolation=Image.BILINEAR, random_bg=True):
        self.interpolation = interpolation
        self.random_bg = random_bg

    def __call__(self, img, angle, expand):
        if self.random_bg:
            return rotate_with_random_bg(img, angle, resample=self.interpolation, expand=expand)
        else:
            return rotate(img, angle, resample=self.interpolation, expand=expand)

### RANDOM CHANNEL SWAPING ###

def swap_channels(img):

    if not _is_pil_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

    img_np = np.array(img)

    channel_idxes = np.random.choice([0, 1, 2], 3, True)

    return Image.fromarray(img_np[:, :, channel_idxes])

class RandomChannelSwap(object):

    def __init__(self, prob):
        self.prob = prob

    def __call__(self, img):
        if np.random.rand() >= self.prob:
            return img

        return swap_channels(img)

### GAMMA CORRECTION ###

def adjust_gamma(img, gamma, gain=1):

    if not _is_pil_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

    if gamma < 0:
        raise ValueError('Gamma should be a non-negative real number')

    gamma_map = [255 * gain * pow(ele / 255., gamma) for ele in range(256)] * 3
    img = img.point(gamma_map)  # use PIL's point-function to accelerate this part

    return img

class RandomGamma(object):

    def __init__(self, gamma_range, gain=1):
        self.min_gamma = gamma_range[0]
        self.max_gamma = gamma_range[1]

        self.gain = gain

    def __call__(self, img):
        gamma = np.random.rand() * (self.max_gamma - self.min_gamma) + self.min_gamma
        return adjust_gamma(img, gamma=gamma, gain=self.gain)

### RESOLUTION ###

def random_resolution(img, ratio):

    if not _is_pil_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

    img_size = np.array(img.size)
    new_size = (img_size * ratio).astype('int')

    img = img.resize(new_size, Image.ANTIALIAS)
    img = img.resize(img_size, Image.ANTIALIAS)

    return img

class RandomResolution(object):

    def __init__(self, ratio_range):
        self.ratio_range = np.arange(ratio_range[0], ratio_range[1], 0.05)

    def __call__(self, img):
        _range = np.random.choice(self.ratio_range)
        return random_resolution(img, _range)

class InvNormalization(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor