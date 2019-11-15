import torch

from .arch import Generator, Discriminator
from .utils import GANLoss

class GANModel(object):

    def __init__(self, config, device, mode='train'):

        self.config = config
        self.mode = mode
        self.device = device

        ## Init G and D
        self.G = Generator(config).to(device)

        if mode == 'train':
            self.D = Discriminator(config).to(device)

            self.criterionGAN = GANLoss(config.GAN_LOSS).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            self.G_optimizer = torch.optim.Adam(self.G.parameters(), lr=config.LR, betas=(config.BETA, 0.999))
            self.D_optimizer = torch.optim.Adam(self.D.parameters(), lr=config.LR, betas=(config.BETA, 0.999))

    def set_requires_grad(self, net, requires_grad=False):
        for param in net.parameters():
            param.requires_grad = requires_grad
        return net

    def reshape_images(self, data):
        pass

    def backward_D(self, real_images, real_wv_images, fake_images, fake_wv_images):
        # Real-real
        real_real = torch.cat((real_images, real_wv_images), 1)
        pred_rr = self.D(real_real)
        self.loss_D_rr = self.criterionGAN(pred_rr, target_is_real=True)

        ## Real-fake
        real_fake = torch.cat((real_images, fake_wv_images), 1)
        pred_rf = self.D(real_fake.detach())
        self.loss_D_rf = self.criterionGAN(pred_rf, target_is_real=False)

        ## Fake-real
        fake_real = torch.cat((fake_images, real_wv_images), 1)
        pred_fr = self.D(fake_real.detach())
        self.loss_D_fr = self.criterionGAN(pred_fr, target_is_real=False)

        self.loss_D = self.loss_D_rr + self.loss_D_rf + 0.5 * self.loss_D_fr
        self.loss_D.backward()

    def backward_G(self, real_images, real_wv_images, fake_images):
        ## Fake-real
        fake_real = torch.cat((fake_images, real_wv_images), 1)
        pred_fr = self.D(fake_real.detach())
        
        self.loss_G_GAN = self.criterionGAN(pred_fr, target_is_real=True)
        self.loss_G_L1 = self.criterionL1(fake_images, real_images) * self.config.LAMBDA_L1

        self.loss_G = self.loss_G_GAN + self.loss_G_L1
        self.loss_G.backward()

    def fit(self, data):
        real_images, real_wv_images, fake_wv_images = data
        fake_images = self.G(real_images)

        # Update D
        self.D = self.set_requires_grad(self.D, True)     # Enable backprop for D
        self.D_optimizer.zero_grad()                      # Set D's gradients to zero
        self.backward_D(real_images, real_wv_images, fake_images, fake_wv_images)
        self.D_optimizer.step()

        # Update G
        self.D = self.set_requires_grad(self.D, False)    # Disable backprop for D
        self.G_optimizer.zero_grad()                      # Set G's gradients to zero
        self.backward_G(real_images, real_wv_images, fake_images)                   
        self.G_optimizer.step()
