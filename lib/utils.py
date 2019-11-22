from io import BytesIO, StringIO
import uuid

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image, ImageDraw, ImageFont

from .preprocess import (RandomChannelSwap, RandomGamma, RandomHorizontalFlip,
                         RandomResizedCrop, RandomResolution, RandomRotate, InvNormalization)
from lib.config import Config

class GANLoss(nn.Module):

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.
        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):

        if target_is_real:
            target_tensor = self.real_label.expand_as(prediction)
            target_tensor_smooth = target_tensor.detach().cpu() - torch.rand(target_tensor.size()) * 0.1    ## Smooth labels
        else:
            target_tensor = self.fake_label.expand_as(prediction)
            target_tensor_smooth = target_tensor.detach().cpu() + torch.rand(target_tensor.size()) * 0.1    ## Smooth labels
        return target_tensor, target_tensor_smooth

    def __call__(self, prediction, target_is_real):

        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor, target_tensor_smooth = self.get_target_tensor(prediction, target_is_real)
            if torch.cuda.is_available():
                target_tensor_smooth = target_tensor_smooth.to('cuda')
            loss = self.loss(prediction, target_tensor_smooth)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()

        _prediction = prediction.detach().cpu().numpy()
        _target_tensor = target_tensor.detach().cpu().numpy()
        if target_is_real:
            accuracy = np.mean(np.argmax(_prediction, axis=1) == _target_tensor)
        else:
            accuracy = np.mean(np.argmax(_prediction, axis=1) == _target_tensor)
        return loss, accuracy

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
    def image_inverse_normalizer(mean, std):
        return InvNormalization(mean=mean, std=std)

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

def get_uuid():
    return str(uuid.uuid4()).split('-')[-1]

def words2image(text_list, config):

    w = config.IMAGE_SIZE_WIDTH
    h = config.IMAGE_SIZE_HEIGHT
    n_column = config.WORDS2IMAGE_N_COLUMN

    img = Image.fromarray(np.ones((h, w)))
    draw = ImageDraw.Draw(img)

    ## Look for fonts
    for font in config.FONTS:
        try:
            font = ImageFont.truetype(font, 9)
        except OSError:
            continue
    
    if n_column == 1:
        
        x0 = int(w * 0.001)
        y0 = int(h * 0.001)
        word_height = h // len(text_list)
        for i, text in enumerate(text_list):
            y = i * word_height + y0
            x = x0
            draw.text((x, y), text, 0, font=font)

    elif n_column == 2:
        
        x1 = int(w * 0.01)
        x2 = int(w * 0.51)
        y0 = int(h * 0.01)
        word_height = h // len(text_list) * 2
        for i, text in enumerate(text_list):
            y = (i // 2) * word_height + y0 if i % 2 == 0 else (i - 1) // 2 * word_height + y0
            x = x1 if i % 2 == 0 else x2
            draw.text((x, y), text, 0, font=font)
            
    else:
        print("'words2image': Column {} not implemented".format(n_column))
        raise NotImplementedError

    return np.array(img.convert('RGB'))