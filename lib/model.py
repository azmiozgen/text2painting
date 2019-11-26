from abc import ABC, abstractmethod
import os
import shutil
import time

import numpy as np
from PIL import Image
import torch
from torchvision.utils import make_grid

from .arch import GeneratorResNet, DiscriminatorPixel
from .utils import GANLoss, get_gradient_penalty, get_uuid, words2image, ImageUtilities

class BaseModel(ABC):

    def __init__(self, config, model_file=None, mode='train'):

        assert mode in ['train', 'test'], 'Mode should be one of "train, test"'
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

        self.state_dict = {
                          'g' : None,
                          'g_optim' : None,
                          'g_lr_scheduler' : None,
                          'd' : None,
                          'd_optim' : None,
                          'd_lr_scheduler' : None,
                          'epoch' : None
                          }
        self.epoch = 0
        self.loss_G = None
        self.loss_D = None
        self.model_dir = None
        self.train_log_file = None
        self.val_log_file = None

    @abstractmethod
    def load_state_dict(self, model_file):
        pass
    @abstractmethod
    def set_state_dict(self):
        pass
    @abstractmethod
    def save_model_dict(self, epoch, iteration, loss_g, loss_d):
        pass
    @abstractmethod
    def set_model_dir(self, model_file=None):
        pass
    @abstractmethod
    def save_logs(self, log_tuple):
        pass
    @abstractmethod
    def set_requires_grad(self, net, requires_grad=False):
        pass
    @abstractmethod
    def get_losses(self):
        pass
    @abstractmethod
    def get_D_accuracy(self):
        pass
    @abstractmethod
    def update_lr(self):
        pass
    @abstractmethod
    def forward(self, real_wv_tensor):
        pass
    @abstractmethod
    def fit(self, data, phase='train'):
        pass

class GANModel(BaseModel):

    def __init__(self, config, model_file=None, mode='train', reset_lr=False):

        assert mode in ['train', 'test'], 'Mode should be one of "train, test"'
        self.config = config
        self.mode = mode
        self.reset_lr = reset_lr
        self.device = config.DEVICE
        self.model_name = config.MODEL_NAME
        self.log_header = config.LOG_HEADER

        self.batch_size = config.BATCH_SIZE
        self.gan_loss = config.GAN_LOSS
        lr = config.LR
        beta = config.BETA
        weight_decay = config.WEIGHT_DECAY
        self.lambda_l1 = config.LAMBDA_L1

        ## Init G and D
        self.G = GeneratorResNet(config).to(self.device)

        ## Init networks and optimizers
        if mode == 'train':
            self.D = DiscriminatorPixel(config).to(self.device)

            self.G_criterionGAN = GANLoss(self.gan_loss, self.device, accuracy=False).to(self.device)
            self.D_criterionGAN = GANLoss(self.gan_loss, self.device, accuracy=True).to(self.device)
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
                                                                             factor=0.75,
                                                                             threshold=0.01,
                                                                             patience=10)
            
            self.D_lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.D_optimizer, 
                                                                             mode='min',
                                                                             factor=0.75,
                                                                             threshold=0.01,
                                                                             patience=10)

        if torch.cuda.device_count() > 1:
            self.G = torch.nn.DataParallel(self.G)
            self.D = torch.nn.DataParallel(self.D)

        ## Init things (these will get values later) 
        self.state_dict = {
                          'g' : None,
                          'g_optim' : None,
                          'g_lr_scheduler' : None,
                          'd' : None,
                          'd_optim' : None,
                          'd_lr_scheduler' : None,
                          }
        self.epoch = 1
        self.loss_G = None
        self.loss_D = None
        self.loss_gp_fr = None
        self.loss_gp_rf = None
        self.accuracy_D_rr = 0.0
        self.accuracy_D_rf = 0.0
        self.accuracy_D_fr = 0.0
        self.model_dir = None
        self.train_log_file = None
        self.val_log_file = None

        if model_file:
            self.load_state_dict(model_file)
            self.set_model_dir(model_file)
            print("{} loaded.".format(self.model_dir))
        else:
            self.set_state_dict()
            self.set_model_dir()
            print("{} created.".format(self.model_dir))
        time.sleep(1.0)

        # print(self.G)
        # print(self.D)
        print("Device:", self.device)
        print("Parameters:")
        print("\tBatch size:", self.batch_size)
        print("\tGAN loss:", self.gan_loss)
        print("\tLearning rates (G, D):", self.G_optimizer.param_groups[0]['lr'], self.D_optimizer.param_groups[0]['lr'])
        print("\tAdam optimizer beta:", beta)
        print("\tWeight decay:", weight_decay)
        print("\tGenerator lambda weight:", self.lambda_l1)

    def load_state_dict(self, model_file):
        ## Get epoch
        self.epoch = int(os.path.basename(model_file).split('_')[1]) + 1

        state = torch.load(model_file)
        self.G.load_state_dict(state['g'])
        if self.mode == 'train':
            self.G_optimizer.load_state_dict(state['g_optim'])
            self.D.load_state_dict(state['d'])
            self.D_optimizer.load_state_dict(state['d_optim'])
            self.G_lr_scheduler.load_state_dict(state['g_lr_scheduler'])
            self.D_lr_scheduler.load_state_dict(state['d_lr_scheduler'])
        if self.reset_lr:
            self.G_optimizer.param_groups[0]['lr'] = self.config.LR
            self.D_optimizer.param_groups[0]['lr'] = self.config.LR
        self.set_state_dict()

    def set_state_dict(self):
        self.state_dict['g'] = self.G.state_dict()
        if self.mode == 'train':
            self.state_dict['g_optim'] = self.G_optimizer.state_dict()
            self.state_dict['g_lr_scheduler'] = self.G_lr_scheduler.state_dict()
            self.state_dict['d'] = self.D.state_dict()
            self.state_dict['d_optim'] = self.D_optimizer.state_dict()
            self.state_dict['d_lr_scheduler'] = self.D_lr_scheduler.state_dict()

    def save_model_dict(self, epoch, iteration, loss_g, loss_d):
        model_filename = "{}_{:04}_{:08}_{:.4f}_{:.4f}.pth".format(self.model_name, epoch, iteration, loss_g, loss_d)
        model_file = os.path.join(self.model_dir, model_filename)
        self.set_state_dict()
        torch.save(self.state_dict, model_file)

    def set_model_dir(self, model_file=None):
        if model_file:
            model_dir = os.path.join(self.config.MODEL_DIR, os.path.basename(os.path.dirname(model_file)))
        else:
            model_dirname = "{}_{}".format(self.model_name, get_uuid())
            model_dir = os.path.join(self.config.MODEL_DIR, model_dirname)
            os.makedirs(model_dir, exist_ok=True)
        self.model_dir = model_dir

        ## Copy current lib/ tree
        current_lib_dir = os.path.join(self.config.BASE_DIR, 'lib')
        copy_lib_dir = os.path.join(model_dir, 'lib')
        if os.path.isdir(copy_lib_dir):
            shutil.rmtree(copy_lib_dir)
        shutil.copytree(current_lib_dir, copy_lib_dir)

        ## Init log files
        train_log_filename = self.model_name + '_train_log.csv'
        val_log_filename = self.model_name + '_val_log.csv'
        train_log_file = os.path.join(self.model_dir, train_log_filename)
        val_log_file = os.path.join(self.model_dir, val_log_filename)
        self.train_log_file = train_log_file
        self.val_log_file = val_log_file

        ## Write headers if new model
        if not model_file:
            with open(train_log_file, 'w') as f:
                f.write(self.log_header + '\n')
            with open(val_log_file, 'w') as f:
                f.write(self.log_header + '\n')

    def save_logs(self, log_tuple):
        phase, epoch, iteration, loss_g, loss_d, acc_rr, acc_rf, acc_fr = log_tuple
        log_file = self.train_log_file if phase == 'train' else self.val_log_file
        log_row_str = '{},{},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f}\n'.format(epoch, iteration, loss_g, loss_d, acc_rr, acc_rf, acc_fr)
        with open(log_file, 'a') as f:
            f.write(log_row_str)

    def set_requires_grad(self, net, requires_grad=False):
        for param in net.parameters():
            param.requires_grad = requires_grad
        return net

    # def stitch_data(self, data, fake_images_tensor):
    #     real_images_tensor, real_wv_tensor, fake_wv_tensor = data
    #     b, _, h, w = real_images_tensor.shape
    #     real_wv_tensor_reshaped = real_wv_tensor.reshape((b, 1, h, w))
    #     fake_wv_tensor_reshaped = fake_wv_tensor.reshape((b, 1, h, w))

    #     ## Stitch images and word vectors on channel axis
    #     real_real_pair = torch.cat((real_images_tensor, real_wv_tensor_reshaped), 1)
    #     real_fake_pair = torch.cat((real_images_tensor, fake_wv_tensor_reshaped), 1)
    #     fake_real_pair = torch.cat((fake_images_tensor, real_wv_tensor_reshaped), 1)

    #     return real_real_pair, real_fake_pair, fake_real_pair

    def set_inputs(self, data, fake_images_tensor):
        real_images_tensor, real_wv_tensor, fake_wv_tensor = data
        b = real_wv_tensor.size(0)
        real_wv_tensor = real_wv_tensor.view(b, -1)
        fake_wv_tensor = fake_wv_tensor.view(b, -1)

        ## Make pairs
        real_real_pair = (real_images_tensor, real_wv_tensor)
        real_fake_pair = (real_images_tensor, fake_wv_tensor)
        fake_real_pair = (fake_images_tensor, real_wv_tensor)

        return real_real_pair, real_fake_pair, fake_real_pair

    def backward_D(self, rr_pair, rf_pair, fr_pair, update=True):

        ## Open pairs
        real_images, real_wvs = rr_pair
        real_images, fake_wvs = rf_pair
        fake_images, real_wvs = fr_pair

        # Real-real
        pred_rr = self.D(real_images, real_wvs)
        self.loss_D_rr, self.accuracy_D_rr = self.D_criterionGAN(pred_rr, target_is_real=True)

        ## Real-fake
        pred_rf = self.D(real_images, fake_wvs)
        self.loss_D_rf, self.accuracy_D_rf = self.D_criterionGAN(pred_rf, target_is_real=False)

        ## Fake-real
        pred_fr = self.D(fake_images.detach(), real_wvs.detach())
        self.loss_D_fr, self.accuracy_D_fr = self.D_criterionGAN(pred_fr, target_is_real=False)

        if self.gan_loss == 'wgangp':
            self.loss_gp_fr, _ = get_gradient_penalty(self.D, rr_pair, fr_pair.detach(), self.device,
                                                   type='mixed', constant=1.0, lambda_gp=10.0)
            self.loss_gp_rf, _ = get_gradient_penalty(self.D, rr_pair, rf_pair.detach(), self.device,
                                                   type='mixed', constant=1.0, lambda_gp=10.0)
            self.loss_gp_fr.backward(retain_graph=True)
            self.loss_gp_rf.backward(retain_graph=True)

        self.loss_D = self.loss_D_rr + self.loss_D_rf + 0.5 * self.loss_D_fr
        if update:
            self.loss_D.backward()

    def backward_G(self, fr_pair, real_images_tensor, fake_images_tensor, update=True):

        ## Open pair
        fake_images, real_wvs = fr_pair

        ## Fake-real
        pred_fr = self.D(fake_images.detach(), real_wvs.detach())

        loss_G_GAN, _ = self.G_criterionGAN(pred_fr, target_is_real=True)
        loss_G_L1 = self.criterionL1(fake_images_tensor, real_images_tensor) * self.lambda_l1

        self.loss_G = loss_G_GAN + loss_G_L1
        if update:
            self.loss_G.backward()

    def get_losses(self):
        loss_g = self.loss_G.item() if self.loss_G else -1.0
        loss_d = self.loss_D.item() if self.loss_D else -1.0
        loss_gp_fr = self.loss_gp_fr.item() if self.loss_gp_fr else -1.0
        loss_gp_rf = self.loss_gp_rf.item() if self.loss_gp_rf else -1.0
        return loss_g, loss_d, loss_gp_fr, loss_gp_rf

    def get_D_accuracy(self):
        return (self.accuracy_D_rr, self.accuracy_D_rf, self.accuracy_D_fr)

    def update_lr(self):
        self.G_lr_scheduler.step(0)
        self.D_lr_scheduler.step(0)
        D_lr = self.G_optimizer.param_groups[0]['lr']
        G_lr = self.D_optimizer.param_groups[0]['lr']

        print('\t\t(G learning rate is {:.4E})'.format(G_lr))
        print('\t\t(D learning rate is {:.4E})'.format(D_lr))

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

            ## Unique words are visualized by converting into image
            words = np.unique(words)
            word_image = words2image(words, self.config)

            ## Inverse normalize  ## TODO if input not normalized remove it
            fake_image = ImageUtilities.image_inverse_normalizer(self.config.MEAN, self.config.STD)(fake_image)
            real_image = ImageUtilities.image_inverse_normalizer(self.config.MEAN, self.config.STD)(real_image)

            ## Go to cpu numpy array
            fake_image = fake_image.detach().cpu().numpy().transpose(1, 2, 0)
            real_image = real_image.detach().cpu().numpy().transpose(1, 2, 0)

            images_bag.extend([word_image, fake_image, real_image])

        images_bag = np.array(images_bag)
        grid = make_grid(torch.Tensor(images_bag.transpose(0, 3, 1, 2)), nrow=self.config.N_GRID_ROW).permute(1, 2, 0)
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

    def fit(self, data, phase='train', train_D=True, train_G=True):
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
        rr_pair, rf_pair, fr_pair = self.set_inputs(data, fake_images_tensor)
        # print("Real-real pair:", rr_pair.shape)
        # print("Real-fake pair:", rf_pair.shape)
        # print("Fake-real pair:", fr_pair.shape)

        if phase == 'train':

            ## Update D
            self.D = self.set_requires_grad(self.D, train_D)
            # all_true = np.all([param.requires_grad for param in self.D.parameters()])
            # print("All D parameters have grad:", str(all_true))
            self.D_optimizer.zero_grad()
            self.backward_D(rr_pair, rf_pair, fr_pair, update=train_D)
            if train_D:
                self.D_optimizer.step()

            ## Update G
            self.D = self.set_requires_grad(self.D, False)      # Disable backprop for D
            self.G = self.set_requires_grad(self.G, train_G)
            self.G_optimizer.zero_grad()
            self.backward_G(fr_pair, real_images_tensor, fake_images_tensor, update=train_G)
            if train_G:
                self.G_optimizer.step()

        else:
            self.backward_D(rr_pair, rf_pair, fr_pair, update=False)
            self.backward_G(fr_pair, real_images_tensor, fake_images_tensor, update=False)
