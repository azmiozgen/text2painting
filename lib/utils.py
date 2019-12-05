from io import BytesIO, StringIO
import uuid

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image, ImageDraw, ImageFont
from graphviz import Digraph
from torch.autograd import Variable, Function

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

    def get_target_tensor(self, prediction, target_is_real, prob_flip_labels=0.0):

        is_flip = np.random.rand() < prob_flip_labels   ## No flipping real-fake labels
        if target_is_real:
            if is_flip:
                target_tensor = self.fake_label.expand_as(prediction)
                target_tensor_smooth = target_tensor.detach().cpu() + torch.rand(target_tensor.size()) * 0.1    ## Smooth labels
            else:
                target_tensor = self.real_label.expand_as(prediction)
                target_tensor_smooth = target_tensor.detach().cpu() - torch.rand(target_tensor.size()) * 0.1    ## Smooth labels
            target_true_tensor = self.real_label.expand_as(prediction)
        else:
            if is_flip:
                target_tensor = self.real_label.expand_as(prediction)
                target_tensor_smooth = target_tensor.detach().cpu() - torch.rand(target_tensor.size()) * 0.1    ## Smooth labels
            else:
                target_tensor = self.fake_label.expand_as(prediction)
                target_tensor_smooth = target_tensor.detach().cpu() + torch.rand(target_tensor.size()) * 0.1    ## Smooth labels
            target_true_tensor = self.fake_label.expand_as(prediction)
        return target_tensor, target_tensor_smooth, target_true_tensor

    def __call__(self, prediction, target_is_real, prob_flip_labels=0.0):

        ## Compute loss
        _, target_tensor_smooth, target_true_tensor = self.get_target_tensor(prediction, target_is_real, prob_flip_labels=prob_flip_labels)
        target_tensor_smooth = target_tensor_smooth.to(self.device)
        if self.gan_mode in ['lsgan', 'vanilla']:
            loss = self.loss(prediction, target_tensor_smooth)
        elif self.gan_mode == 'wgangp':
            loss = -prediction.mean() if target_is_real else prediction.mean()

        ## Compute accuracy
        if self.accuracy:
            _prediction = prediction.detach().cpu()
            _target_tensor = target_true_tensor.detach().cpu()
            if target_is_real:
                accuracy = torch.mean(((torch.sigmoid(_prediction) >= 0.5).float() == _target_tensor).float()).item()
            else:
                accuracy = torch.mean(((torch.sigmoid(_prediction) < 0.5).float() == _target_tensor).float()).item()
            return loss, accuracy

        return loss, -1.0

def get_single_gradient_penalty(netD, real_image, fake_image, device, type='mixed', constant=1.0, lambda_gp=10.0):
    fake_image = fake_image.detach()
    if lambda_gp > 0.0:
        if type == 'real':   # either use real images, fake images, or a linear interpolation of two.
            interpolatesv = real_image
        elif type == 'fake':
            interpolatesv = fake_image
        elif type == 'mixed':
            alpha = torch.rand(real_image.shape[0], 1, device=device)
            alpha = alpha.expand(real_image.shape[0], real_image.nelement() // real_image.shape[0]).contiguous().view(*real_image.shape)
            interpolatesv = alpha * real_image + ((1 - alpha) * fake_image)
        else:
            raise NotImplementedError('{} not implemented'.format(type))
        interpolatesv.requires_grad_(True)
        disc_interpolates = netD(interpolatesv)
        gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=[interpolatesv, ],
                                        grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                                        create_graph=True, retain_graph=True, only_inputs=True)
        gradients = gradients[0].view(real_image.size(0), -1)
        gradient_penalty = (((gradients + 1e-16).norm(2, dim=1) - constant) ** 2).mean() * lambda_gp        # added eps
        return gradient_penalty, gradients
    else:
        return 0.0, None

def get_paired_gradient_penalty(netD, real_data, fake_data, device, type='mixed', constant=1.0, lambda_gp=10.0):
    real_image, real_wv = real_data[0], real_data[1]
    image, wv = fake_data[0].detach(), fake_data[1].detach()
    if lambda_gp > 0.0:
        if type == 'real':   # either use real images, fake images, or a linear interpolation of two.
            interpolatesv, wv = real_image, real_wv
        elif type == 'fake':
            interpolatesv, wv = image, wv
        elif type == 'mixed':
            alpha = torch.rand(real_image.shape[0], 1, device=device)
            alpha = alpha.expand(real_image.shape[0], real_image.nelement() // real_image.shape[0]).contiguous().view(*real_image.shape)
            interpolatesv = alpha * real_image + ((1 - alpha) * image)

            alpha = torch.rand(real_wv.shape[0], 1, device=device)
            alpha = alpha.expand(real_wv.shape[0], real_wv.nelement() // real_wv.shape[0]).contiguous().view(*real_wv.shape)
            wv = alpha * real_wv + ((1 - alpha) * wv)
        else:
            raise NotImplementedError('{} not implemented'.format(type))
        interpolatesv.requires_grad_(True)
        wv.requires_grad_(True)
        disc_interpolates = netD(interpolatesv, wv)
        gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=[interpolatesv, wv],
                                        grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                                        create_graph=True, retain_graph=True, only_inputs=True)
        gradients0 = gradients[0].view(real_image.size(0), -1)
        gradients1 = gradients[1].view(real_wv.size(0), -1)
        gradient_penalty0 = (((gradients0 + 1e-16).norm(2, dim=1) - constant) ** 2).mean() * lambda_gp        # added eps
        gradient_penalty1 = (((gradients1 + 1e-16).norm(2, dim=1) - constant) ** 2).mean() * lambda_gp        # added eps
        gradient_penalty = gradient_penalty0 + gradient_penalty1
        return gradient_penalty, gradients0, gradients1
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
    

    w = config.IMAGE_WIDTH
    h = config.IMAGE_HEIGHT
    n_column = config.WORDS2IMAGE_N_COLUMN

    img = Image.fromarray(np.ones((h, w)))
    draw = ImageDraw.Draw(img)

    ## Look for fonts
    for font in config.FONTS:
        try:
            font = ImageFont.truetype(font, config.FONT_SIZE)
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

def iter_graph(root, callback):
    queue = [root]
    seen = set()
    while queue:
        fn = queue.pop()
        if fn in seen:
            continue
        seen.add(fn)
        for next_fn, _ in fn.next_functions:
            if next_fn is not None:
                queue.append(next_fn)
        callback(fn)

def register_hooks(var):
    fn_dict = {}
    def hook_cb(fn):
        def register_grad(grad_input, grad_output):
            fn_dict[fn] = grad_input
        fn.register_hook(register_grad)
    iter_graph(var.grad_fn, hook_cb)

    def is_bad_grad(grad_output):
        grad_output = grad_output.data
        return grad_output.ne(grad_output).any() or grad_output.gt(1e6).any()

    def make_dot():
        node_attr = dict(style='filled',
                        shape='box',
                        align='left',
                        fontsize='12',
                        ranksep='0.1',
                        height='0.2')
        dot = Digraph(node_attr=node_attr, graph_attr=dict(size="12,12"))

        def size_to_str(size):
            return '('+(', ').join(map(str, size))+')'

        def build_graph(fn):
            if hasattr(fn, 'variable'):  # if GradAccumulator
                u = fn.variable
                node_name = 'Variable\n ' + size_to_str(u.size())
                dot.node(str(id(u)), node_name, fillcolor='lightblue')
            else:
                assert fn in fn_dict, fn
                fillcolor = 'white'
                if any(is_bad_grad(gi) for gi in fn_dict[fn]):
                    fillcolor = 'red'
                dot.node(str(id(fn)), str(type(fn).__name__), fillcolor=fillcolor)
            for next_fn, _ in fn.next_functions:
                if next_fn is not None:
                    next_id = id(getattr(next_fn, 'variable', next_fn))
                    dot.edge(str(next_id), str(id(fn)))
        iter_graph(var.grad_fn, build_graph)

        return dot

    return make_dot