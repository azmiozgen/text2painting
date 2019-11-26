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

    def __init__(self, gan_mode, device, target_real_label=1.0, target_fake_label=0.0, accuracy=False):
        """ Initialize the GANLoss class.
        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        self.device = device
        self.accuracy = accuracy
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('GAN mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):

        if target_is_real:
            target_tensor = self.real_label.expand_as(prediction)
            target_tensor_smooth = target_tensor.detach().cpu() - torch.rand(target_tensor.size()) * 0.1    ## Smooth labels
        else:
            target_tensor = self.fake_label.expand_as(prediction)
            target_tensor_smooth = target_tensor.detach().cpu() + torch.rand(target_tensor.size()) * 0.1    ## Smooth labels
        return target_tensor, target_tensor_smooth

    def __call__(self, prediction, target_is_real):

        ## Compute loss
        target_tensor, target_tensor_smooth = self.get_target_tensor(prediction, target_is_real)
        target_tensor_smooth = target_tensor_smooth.to(self.device)
        if self.gan_mode in ['lsgan', 'vanilla']:
            loss = self.loss(prediction, target_tensor_smooth)
        elif self.gan_mode == 'wgangp':
            loss = -prediction.mean() if target_is_real else prediction.mean()

        ## Compute accuracy
        if self.accuracy:
            _prediction = prediction.detach().cpu()
            _target_tensor = target_tensor.detach().cpu()
            if target_is_real:
                accuracy = torch.mean(((torch.sigmoid(_prediction) >= 0.5).float() == _target_tensor).float()).item()
            else:
                accuracy = torch.mean(((torch.sigmoid(_prediction) < 0.5).float() == _target_tensor).float()).item()
            return loss, accuracy

        return loss, -1.0

def get_gradient_penalty(netD, real_data, fake_data, device, type='mixed', constant=1.0, lambda_gp=10.0):
    """Calculate the gradient penalty loss, used in WGAN-GP paper https://arxiv.org/abs/1704.00028
    Arguments:
        netD (network)              -- discriminator network
        real_data (tensor array)    -- real images
        fake_data (tensor array)    -- generated images from the generator
        device (str)                -- GPU / CPU: from torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        type (str)                  -- if we mix real and fake data or not [real | fake | mixed].
        constant (float)            -- the constant used in formula ( | |gradient||_2 - constant)^2
        lambda_gp (float)           -- weight for this loss
    Returns the gradient penalty loss
    """
    if lambda_gp > 0.0:
        if type == 'real':   # either use real images, fake images, or a linear interpolation of two.
            interpolatesv = real_data
        elif type == 'fake':
            interpolatesv = fake_data
        elif type == 'mixed':
            alpha = torch.rand(real_data.shape[0], 1, device=device)
            alpha = alpha.expand(real_data.shape[0], real_data.nelement() // real_data.shape[0]).contiguous().view(*real_data.shape)
            interpolatesv = alpha * real_data + ((1 - alpha) * fake_data)
        else:
            raise NotImplementedError('{} not implemented'.format(type))
        interpolatesv.requires_grad_(True)
        disc_interpolates = netD(interpolatesv)
        gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolatesv,
                                        grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                                        create_graph=True, retain_graph=True, only_inputs=True)
        gradients = gradients[0].view(real_data.size(0), -1)  # flat the data
        gradient_penalty = (((gradients + 1e-16).norm(2, dim=1) - constant) ** 2).mean() * lambda_gp        # added eps
        return gradient_penalty, gradients
    else:
        return 0.0, None

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