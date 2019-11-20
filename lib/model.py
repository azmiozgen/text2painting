import os
import shutil

import numpy as np
from PIL import Image
import torch
from torchvision.utils import make_grid

from .arch import Generator, Discriminator
from .utils import GANLoss, get_uuid, words2image, ImageUtilities

class GANModel(object):

    def __init__(self, config, mode='train'):

        self.config = config
        self.mode = mode
        self.device = config.DEVICE
        self.model_name = config.MODEL_NAME
        self.log_header = config.LOG_HEADER

        self.batch_size = config.BATCH_SIZE
        gan_loss = config.GAN_LOSS
        lr = config.LR
        beta = config.BETA
        weight_decay = config.WEIGHT_DECAY
        self.lambda_l1 = config.LAMBDA_L1

        ## Init G and D
        self.G = Generator(config).to(self.device)

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

            self.G_lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.G_optimizer, 
                                                                             mode='min',
                                                                             factor=0.5,
                                                                             threshold=0.01,
                                                                             patience=5)
            
            self.D_lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.D_optimizer, 
                                                                             mode='min',
                                                                             factor=0.5,
                                                                             threshold=0.01,
                                                                             patience=5)


        ## Init things (these will get values later) 
        self.loss_G = None
        self.loss_D = None
        self.model_dir = None
        self.train_log_file = None
        self.val_log_file = None

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

    def update_lr(self):        
        self.G_lr_scheduler.step(0)
        self.D_lr_scheduler.step(0)
        D_lr = self.G_optimizer.param_groups[0]['lr']
        G_lr = self.D_optimizer.param_groups[0]['lr']

        print('\t\t(G learning rate is {:.4E})'.format(G_lr))
        print('\t\t(D learning rate is {:.4E})'.format(D_lr))

    def init_model_dir(self):
        model_dirname = "{}_{}".format(self.model_name, get_uuid())
        model_dir = os.path.join(self.config.MODEL_DIR, model_dirname)
        os.makedirs(model_dir, exist_ok=True)
        self.model_dir = model_dir

        ## Copy lib/ tree
        old_lib_dir = os.path.join(self.config.BASE_DIR, 'lib')
        new_lib_dir = os.path.join(model_dir, 'lib')
        shutil.copytree(old_lib_dir, new_lib_dir)

        ## Init log files
        train_log_filename = self.model_name + '_train_log.csv'
        val_log_filename = self.model_name + '_val_log.csv'
        train_log_file = os.path.join(self.model_dir, train_log_filename)
        val_log_file = os.path.join(self.model_dir, val_log_filename)
        self.train_log_file = train_log_file
        self.val_log_file = val_log_file
        with open(train_log_file, 'w') as f:
            f.write(self.log_header + '\n')
        with open(val_log_file, 'w') as f:
            f.write(self.log_header + '\n')

    def save_model_dict(self, epoch, iteration, loss_g, loss_d):
        model_filename = "{}_{:04}_{:08}_{:.4f}_{:.4f}.pth".format(self.model_name, epoch, iteration, loss_g, loss_d)
        model_file = os.path.join(self.model_dir, model_filename)
        save_dict = {
                    'g' : self.G.state_dict(), 
                    'g_optim' : self.G_optimizer.state_dict(),
                    'g_lr_scheduler' : self.G_lr_scheduler.state_dict(),
                    'd' : self.D.state_dict(),
                    'd_optim' : self.D_optimizer.state_dict(),
                    'd_lr_scheduler' : self.D_lr_scheduler.state_dict()
                    }
        torch.save(save_dict, model_file)

    def save_logs(self, phase, epoch, iteration, loss_g, loss_d):
        log_file = self.train_log_file if phase == 'train' else self.val_log_file
        log_row_str = '{},{},{:.4f},{:.4f}\n'.format(epoch, iteration, loss_g, loss_d)
        with open(log_file, 'a') as f:
            f.write(log_row_str)

    def generate_grid(self, real_wv_tensor, real_images_tensor, word2vec_model):
        ## Generate fake image
        fake_images_tensor = self.forward(real_wv_tensor)

        images_bag = []
        for fake_image, real_image, real_wvs in zip(fake_images_tensor, real_images_tensor, real_wv_tensor):
            words = []

            ## Get words from word vectors
            for real_wv in real_wvs:
                real_wv = np.array(real_wv)
                word, _ = word2vec_model.wv.similar_by_vector(real_wv)[0]
                words.append(word)

            ## Words are visualized by converting image
            word_image = words2image(words)

            ## Inverse normalize  ## TODO if input not normalized remove it
            fake_image = ImageUtilities.image_inverse_normalizer(self.config.MEAN, self.config.STD)(fake_image)
            real_image = ImageUtilities.image_inverse_normalizer(self.config.MEAN, self.config.STD)(real_image)

            ## Go to cpu numpy array
            fake_image = fake_image.detach().cpu().numpy().transpose(1, 2, 0)
            real_image = real_image.detach().cpu().numpy().transpose(1, 2, 0)

            images_bag.extend([word_image, fake_image, real_image])

        images_bag = np.array(images_bag)
        grid = make_grid(torch.Tensor(images_bag.transpose(0, 3, 1, 2)), nrow=6).permute(1, 2, 0)
        grid_pil = Image.fromarray(np.array(grid * 255, dtype=np.uint8))
        return grid_pil

    def save_output(self, img_pil, filename):
        output_dir = os.path.join(self.model_dir, 'output')
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, filename)
        img_pil.save(output_file)

    def forward(self, real_wv_tensor):
        ## Data to device
        real_wv_tensor = real_wv_tensor.to(self.device)

        ## Forward G
        real_wv_flat_tensor = real_wv_tensor.view(self.batch_size, -1)
        fake_images_tensor = self.G(real_wv_flat_tensor)

        return fake_images_tensor

    def fit(self, data, phase='train'):
        ## Data to device
        real_images_tensor, real_wv_tensor, fake_wv_tensor = data
        real_images_tensor = real_images_tensor.to(self.device)
        real_wv_tensor = real_wv_tensor.to(self.device)
        fake_wv_tensor = fake_wv_tensor.to(self.device)
        data = real_images_tensor, real_wv_tensor, fake_wv_tensor

        ## Forward G
        fake_images_tensor = self.forward(fake_wv_tensor)
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


