import os
import shutil

import torch

from .arch import Generator, Discriminator
from .utils import GANLoss

class GANModel(object):

    def __init__(self, config, mode='train'):

        self.config = config
        self.mode = mode
        self.device = config.DEVICE

        self.batch_size = config.BATCH_SIZE
        gan_loss = config.GAN_LOSS
        lr = config.LR
        beta = config.BETA
        weight_decay = config.WEIGHT_DECAY
        self.lambda_l1 = config.LAMBDA_L1

        ## Init G and D
        self.G = Generator(config).to(self.device)
        self.networks = [self.G]

        ## Init networks and optimizers
        if mode == 'train':
            self.D = Discriminator(config).to(self.device)

            self.criterionGAN = GANLoss(gan_loss).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            self.G_optimizer = torch.optim.Adam(self.G.parameters(),
                                                lr=lr,
                                                betas=(beta, 0.999),
                                                weight_decay=weight_decay)
            self.D_optimizer = torch.optim.Adam(self.D.parameters(),
                                                lr=lr,
                                                betas=(beta, 0.999),
                                                weight_decay=weight_decay)

            self.networks.append(self.D)

        ## Init losses
        self.loss_G = 0.0
        self.loss_D = 0.0

        print(self.G)
        print(self.D)
        print("Device:", self.device)
        print("Parameters:")
        print("\tBatch size:", self.batch_size)
        print("\tGAN loss:", gan_loss)
        print("\tLearning rate:", lr)
        print("\tAdam optimizer beta:", beta)
        print("\tWeight decay:", weight_decay)
        print("\tGenerator lambda weight:", self.lambda_l1)

    def set_requires_grad(self, net, requires_grad=False):
        for param in net.parameters():
            param.requires_grad = requires_grad
        return net

    def stitch_data(self, data, fake_images_tensor):
        real_images_tensor, real_wv_tensor, fake_wv_tensor = data
        real_wv_tensor_reshaped = real_wv_tensor.reshape(real_images_tensor.shape)
        fake_wv_tensor_reshaped = fake_wv_tensor.reshape(real_images_tensor.shape)

        ## Stitch images and word vectors on channel axis
        real_real_pair = torch.cat((real_images_tensor, real_wv_tensor_reshaped), 1)
        real_fake_pair = torch.cat((real_images_tensor, fake_wv_tensor_reshaped), 1)
        fake_real_pair = torch.cat((fake_images_tensor, real_wv_tensor_reshaped), 1)

        return real_real_pair, real_fake_pair, fake_real_pair

    def backward_D(self, rr_pair, rf_pair, fr_pair):
        # Real-real
        pred_rr = self.D(rr_pair)
        self.loss_D_rr = self.criterionGAN(pred_rr, target_is_real=True)

        ## Real-fake
        pred_rf = self.D(rf_pair)
        self.loss_D_rf = self.criterionGAN(pred_rf, target_is_real=False)

        ## Fake-real
        pred_fr = self.D(fr_pair.detach())
        self.loss_D_fr = self.criterionGAN(pred_fr, target_is_real=False)

        self.loss_D = self.loss_D_rr + self.loss_D_rf + 0.5 * self.loss_D_fr
        self.loss_D.backward()

    def backward_G(self, fr_pair, real_images_tensor, fake_images_tensor):
        ## Fake-real
        pred_fr = self.D(fr_pair.detach())
        
        loss_G_GAN = self.criterionGAN(pred_fr, target_is_real=True)
        loss_G_L1 = self.criterionL1(fake_images_tensor, real_images_tensor) * self.lambda_l1

        self.loss_G = loss_G_GAN + loss_G_L1
        self.loss_G.backward()

    def get_loss(self):
        return self.loss_G.item(), self.loss_D.item()

    def save_model_dict(self, name, epoch, loss_g, loss_d):
        model_dirname = "{}_{:04}_{:.4f}_{:.4f}".format(name, epoch, loss_g, loss_d)
        model_dir = os.path.join(self.config.MODEL_DIR, model_dirname)
        os.makedirs(model_dir, exist_ok=True)
        model_file = os.path.join(model_dir, self.config.MODEL_NAME + '.pth')

        old_lib_dir = os.path.join(self.config.BASE_DIR, 'lib')
        new_lib_dir = os.path.join(model_dir, 'lib')
        shutil.copytree(old_lib_dir, new_lib_dir)

        loss_g, loss_d = self.get_loss()
        save_dict = {
                    'g' : self.G.state_dict(), 
                    'g_optim' : self.G_optimizer.state_dict(),
                    'd' : self.D.state_dict(),
                    'd_optim' : self.D_optimizer.state_dict()
                    }
        torch.save(save_dict, model_file)

    def fit(self, data, phase='train'):
        ## Data to device
        real_images_tensor, real_wv_tensor, fake_wv_tensor = data
        real_images_tensor = real_images_tensor.to(self.device)
        real_wv_tensor = real_wv_tensor.to(self.device)
        fake_wv_tensor = fake_wv_tensor.to(self.device)
        data = real_images_tensor, real_wv_tensor, fake_wv_tensor

        ## Forward G
        real_wv_flat_tensor = real_wv_tensor.view(self.batch_size, -1)
        fake_images_tensor = self.G(real_wv_flat_tensor)
        # print("G output:", fake_images_tensor.shape)

        ## Stich images and word vectors on channel axis
        rr_pair, rf_pair, fr_pair = self.stitch_data(data, fake_images_tensor)
        # print("Real-real pair:", rr_pair.shape)
        # print("Real-fake pair:", rf_pair.shape)
        # print("Fake-real pair:", fr_pair.shape)

        if phase == 'train':
            # Update D
            self.D = self.set_requires_grad(self.D, True)     # Enable backprop for D
            self.D_optimizer.zero_grad()                      # Set D's gradients to zero
            self.backward_D(rr_pair, rf_pair, fr_pair)
            self.D_optimizer.step()

            # Update G
            self.D = self.set_requires_grad(self.D, False)    # Disable backprop for D
            self.G_optimizer.zero_grad()                      # Set G's gradients to zero
            self.backward_G(fr_pair, real_images_tensor, fake_images_tensor)                   
            self.G_optimizer.step()