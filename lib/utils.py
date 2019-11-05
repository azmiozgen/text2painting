from io import BytesIO, StringIO

import torchvision.transforms as transforms
from PIL import Image

from .preprocess import (RandomChannelSwap, RandomGamma, RandomHorizontalFlip,
                         RandomResizedCrop, RandomResolution, RandomRotate)


class ImageUtilities(object):

    @staticmethod
    def read_image(image_path, is_raw=False):
        if is_raw:
            try:
                img = StringIO(image_path)
            except TypeError:
                img = BytesIO(image_path)
            img = Image.open(img).convert('RGB')
        else:
            img = Image.open(image_path).convert('RGB')
        img_copy = img.copy()
        img.close()
        return img_copy

    @staticmethod
    def image_resizer(height, width, interpolation=Image.BILINEAR):
        return transforms.Resize((height, width), interpolation=interpolation)

    @staticmethod
    def image_random_cropper_and_resizer(height, width, interpolation=Image.BILINEAR):
        return RandomResizedCrop(height, width, interpolation=interpolation)

    @staticmethod
    def image_random_horizontal_flipper():
        return RandomHorizontalFlip()

    @staticmethod
    def image_normalizer(mean, std):
        return transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])

    @staticmethod
    def image_random_rotator(interpolation=Image.BILINEAR, random_bg=True):
        return RandomRotate(interpolation=interpolation, random_bg=random_bg)

    @staticmethod
    def image_random_color_jitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2):
        return transforms.ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue)

    @staticmethod
    def image_random_grayscaler(p=0.5):
        return transforms.RandomGrayscale(p=p)

    @staticmethod
    def image_random_channel_swapper(p=0.5):
        return RandomChannelSwap(prob=p)

    @staticmethod
    def image_random_gamma(gamma_range, gain=1):
        return RandomGamma(gamma_range, gain=gain)

    @staticmethod
    def image_random_resolution(ratio_range):
        return RandomResolution(ratio_range)
