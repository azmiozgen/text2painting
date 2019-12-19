from abc import ABC, abstractmethod
import os
import shutil
import time

import numpy as np
from PIL import Image
import torch
import torchvision
from torchvision.utils import make_grid

from .arch import GeneratorResNet, DiscriminatorStack, GeneratorRefinerUNet, DiscriminatorDecider, GeneratorRefinerUNet2, DiscriminatorDecider2
from .utils import (GANLoss, get_single_gradient_penalty, get_paired_gradient_penalty,
                    get_uuid, words2image, ImageUtilities)

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
        self.gan_loss1 = config.GAN_LOSS1
        self.gan_loss2 = config.GAN_LOSS2

        g_lr = config.G_LR
        d_lr = config.D_LR
        g_refiner_lr = config.G_REFINER_LR
        d_decider_lr = config.D_DECIDER_LR
        g_refiner2_lr = config.G_REFINER2_LR
        d_decider2_lr = config.D_DECIDER2_LR

        lr_drop_factor = config.LR_DROP_FACTOR
        lr_drop_patience = config.LR_DROP_PATIENCE
        lr_min_val = config.LR_MIN_VAL

        beta = config.BETA
        weight_decay = config.WEIGHT_DECAY
        weight_init = config.WEIGHT_INIT
        init_gain = config.INIT_GAIN

        self.lambda_l1 = config.LAMBDA_L1
        self.inv_normalize = config.NORMALIZE
        self.prob_flip_labels = config.PROB_FLIP_LABELS
        self.min_similarity_prob = config.MIN_WV_SIMILARITY_PROB

        self.inverse_normalizer = ImageUtilities.image_inverse_normalizer(self.config.MEAN, self.config.STD)
        self.image_resizer = ImageUtilities.image_resizer(self.config.IMAGE_HEIGHT, self.config.IMAGE_WIDTH, interpolation=Image.NEAREST)

        ## Init G
        self.G = GeneratorResNet(config).to(self.device)
        self.G_refiner = GeneratorRefinerUNet(config).to(self.device)
        self.G_refiner2 = GeneratorRefinerUNet2(config).to(self.device)

        ## Init D, optimizers, schedulers
        if mode == 'train':
            self.D = DiscriminatorStack(config).to(self.device)
            self.D_decider = DiscriminatorDecider(config).to(self.device)
            self.D_decider2 = DiscriminatorDecider2(config).to(self.device)

            self.G_criterionGAN = GANLoss(self.gan_loss1, self.device, accuracy=False).to(self.device)
            self.D_criterionGAN = GANLoss(self.gan_loss1, self.device, accuracy=True).to(self.device)
            self.G_refiner_criterionGAN = GANLoss(self.gan_loss2, self.device, accuracy=True).to(self.device)
            self.D_decider_criterionGAN = GANLoss(self.gan_loss2, self.device, accuracy=True).to(self.device)
            self.G_refiner2_criterionGAN = GANLoss(self.gan_loss2, self.device, accuracy=True).to(self.device)
            self.D_decider2_criterionGAN = GANLoss(self.gan_loss2, self.device, accuracy=True).to(self.device)
            self.G_criterion_dist = torch.nn.MSELoss()
            self.G_refiner_criterion_dist = torch.nn.MSELoss()
            self.G_refiner2_criterion_dist = torch.nn.MSELoss()

            self.G_optimizer = torch.optim.Adam(self.G.parameters(),
                                                lr=g_lr,
                                                betas=(beta, 0.999),
                                                weight_decay=weight_decay)
            self.D_optimizer = torch.optim.Adam(self.D.parameters(),
                                                lr=d_lr,
                                                betas=(beta, 0.999),
                                                weight_decay=weight_decay)
            self.G_refiner_optimizer = torch.optim.Adam(self.G_refiner.parameters(),
                                                lr=g_refiner_lr,
                                                betas=(beta, 0.999),
                                                weight_decay=weight_decay)
            self.D_decider_optimizer = torch.optim.Adam(self.D_decider.parameters(),
                                                lr=d_decider_lr,
                                                betas=(beta, 0.999),
                                                weight_decay=weight_decay)
            self.G_refiner2_optimizer = torch.optim.Adam(self.G_refiner2.parameters(),
                                                lr=g_refiner2_lr,
                                                betas=(beta, 0.999),
                                                weight_decay=weight_decay)
            self.D_decider2_optimizer = torch.optim.Adam(self.D_decider2.parameters(),
                                                lr=d_decider2_lr,
                                                betas=(beta, 0.999),
                                                weight_decay=weight_decay)

            self.G_lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.G_optimizer, 
                                                                             mode='min',
                                                                             factor=lr_drop_factor,
                                                                             threshold=0.01,
                                                                             patience=lr_drop_patience,
                                                                             min_lr=lr_min_val)

            self.D_lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.D_optimizer, 
                                                                             mode='min',
                                                                             factor=lr_drop_factor,
                                                                             threshold=0.01,
                                                                             patience=lr_drop_patience,
                                                                             min_lr=lr_min_val)

            self.G_refiner_lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.G_refiner_optimizer, 
                                                                                     mode='min',
                                                                                     factor=lr_drop_factor,
                                                                                     threshold=0.01,
                                                                                     patience=lr_drop_patience,
                                                                                     min_lr=lr_min_val)

            self.D_decider_lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.D_decider_optimizer, 
                                                                                     mode='min',
                                                                                     factor=lr_drop_factor,
                                                                                     threshold=0.01,
                                                                                     patience=lr_drop_patience,
                                                                                     min_lr=lr_min_val)

            self.G_refiner2_lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.G_refiner2_optimizer, 
                                                                                     mode='min',
                                                                                     factor=lr_drop_factor,
                                                                                     threshold=0.01,
                                                                                     patience=lr_drop_patience,
                                                                                     min_lr=lr_min_val)

            self.D_decider2_lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.D_decider2_optimizer, 
                                                                                     mode='min',
                                                                                     factor=lr_drop_factor,
                                                                                     threshold=0.01,
                                                                                     patience=lr_drop_patience,
                                                                                     min_lr=lr_min_val)

        ## Parallelize over gpus
        if torch.cuda.device_count() > 1:
            self.G = torch.nn.DataParallel(self.G)
            self.D = torch.nn.DataParallel(self.D)
            self.G_refiner = torch.nn.DataParallel(self.G_refiner)
            self.D_decider = torch.nn.DataParallel(self.D_decider)
            self.G_refiner2 = torch.nn.DataParallel(self.G_refiner2)
            self.D_decider2 = torch.nn.DataParallel(self.D_decider2)

        ## Init weights
        self.init_weights(self.G, weight_init, init_gain=init_gain)
        self.init_weights(self.D, weight_init, init_gain=init_gain)
        self.init_weights(self.G_refiner, weight_init, init_gain=init_gain)
        self.init_weights(self.D_decider, weight_init, init_gain=init_gain)
        self.init_weights(self.G_refiner2, weight_init, init_gain=init_gain)
        self.init_weights(self.D_decider2, weight_init, init_gain=init_gain)

        ## Init things (these will get values later) 
        self.state_dict = {
                          'g' : None,
                          'g_optim' : None,
                          'g_lr_scheduler' : None,
                          'd' : None,
                          'd_optim' : None,
                          'd_lr_scheduler' : None,
                          'g_refiner' : None,
                          'g_refiner_optim' : None,
                          'g_refiner_lr_scheduler' : None,
                          'd_decider' : None,
                          'd_decider_optim' : None,
                          'd_decider_lr_scheduler' : None,
                          'g_refiner2' : None,
                          'g_refiner2_optim' : None,
                          'g_refiner2_lr_scheduler' : None,
                          'd_decider2' : None,
                          'd_decider2_optim' : None,
                          'd_decider2_lr_scheduler' : None,
                          }
        self.epoch = 1
        self.loss_G = None
        self.loss_D = None
        self.loss_G_refiner = None
        self.loss_D_decider = None
        self.loss_G_refiner2 = None
        self.loss_D_decider2 = None
        self.loss_gp_fr = None
        self.loss_gp_rf = None
        self.loss_gp_decider_fr = None
        self.loss_gp_decider2_fr = None
        self.accuracy_D_rr = 0.0
        self.accuracy_D_rf = 0.0
        self.accuracy_D_fr = 0.0
        self.accuracy_D_decider_rr = 0.0
        self.accuracy_D_decider_fr = 0.0
        self.accuracy_D_decider2_rr = 0.0
        self.accuracy_D_decider2_fr = 0.0
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

        G_lr = self.G_optimizer.param_groups[0]['lr']
        D_lr = self.D_optimizer.param_groups[0]['lr']
        G_refiner_lr = self.G_refiner_optimizer.param_groups[0]['lr']
        D_decider_lr = self.D_decider_optimizer.param_groups[0]['lr']
        G_refiner2_lr = self.G_refiner2_optimizer.param_groups[0]['lr']
        D_decider2_lr = self.D_decider2_optimizer.param_groups[0]['lr']

        # print(self.G)
        # print(self.D)
        print("# parameters of G: {:2E}".format(sum(p.numel() for p in self.G.parameters())))
        print("# parameters of D: {:2E}".format(sum(p.numel() for p in self.D.parameters())))
        print("# parameters of G refiner: {:2E}".format(sum(p.numel() for p in self.G_refiner.parameters())))
        print("# parameters of D decider: {:2E}".format(sum(p.numel() for p in self.D_decider.parameters())))
        print("# parameters of G refiner2: {:2E}".format(sum(p.numel() for p in self.G_refiner2.parameters())))
        print("# parameters of D decider2: {:2E}".format(sum(p.numel() for p in self.D_decider2.parameters())))
        print("Device:", self.device)
        print("Parameters:")
        print("\tBatch size:", self.batch_size)
        print("\tGAN loss1:", self.gan_loss1)
        print("\tGAN loss2:", self.gan_loss2)
        # print("\tLearning rates (G, D): {:.4f}, {:.4f}".format(G_lr, D_lr))
        print("\tLearning rates (G, D, G_refiner, D_decider, G_refiner2, D_decider2): \
{:.2E}, {:.2E}, {:.2E}, {:.2E}, {:.2E}, {:.2E}".format(G_lr, D_lr, G_refiner_lr, D_decider_lr, G_refiner2_lr, D_decider2_lr))
        print("\tDropout rates (G, D): {:.2f}, {:.2f}".format(config.G_DROPOUT, config.D_DROPOUT))
        print("\tAdam optimizer beta:", beta)
        print("\tWeight decay:", weight_decay)
        print("\tWeight initialization:", weight_init)
        print("\tGenerator lambda weight:", self.lambda_l1)

    def init_weights(self, net, init_type, init_gain=0.02):
        def init_func(m):
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    torch.nn.init.normal_(m.weight.data, 0.0, init_gain)
                elif init_type == 'xavier':
                    torch.nn.init.xavier_normal_(m.weight.data, gain=init_gain)
                elif init_type == 'kaiming':
                    torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    torch.nn.init.orthogonal_(m.weight.data, gain=init_gain)
                else:
                    raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
                if hasattr(m, 'bias') and m.bias is not None:
                    torch.nn.init.constant_(m.bias.data, 0.0)
            elif classname.find('BatchNorm2d') != -1:
                torch.nn.init.normal_(m.weight.data, 1.0, init_gain)
                torch.nn.init.constant_(m.bias.data, 0.0)

        net.apply(init_func)

    def load_state_dict(self, model_file):
        ## Get epoch
        self.epoch = int(os.path.basename(model_file).split('_')[1]) + 1

        state = torch.load(model_file)
        self.G.load_state_dict(state['g'])
        self.G_refiner.load_state_dict(state['g_refiner'])
        self.G_refiner2.load_state_dict(state['g_refiner2'])
        if self.mode == 'train':
            self.D.load_state_dict(state['d'])
            self.D_decider.load_state_dict(state['d_decider'])
            self.D_decider2.load_state_dict(state['d_decider2'])
            self.G_optimizer.load_state_dict(state['g_optim'])
            self.D_optimizer.load_state_dict(state['d_optim'])
            self.G_refiner_optimizer.load_state_dict(state['g_refiner_optim'])
            self.G_refiner2_optimizer.load_state_dict(state['g_refiner2_optim'])
            self.D_decider_optimizer.load_state_dict(state['d_decider_optim'])
            self.D_decider2_optimizer.load_state_dict(state['d_decider2_optim'])
            self.G_lr_scheduler.load_state_dict(state['g_lr_scheduler'])
            self.D_lr_scheduler.load_state_dict(state['d_lr_scheduler'])
            self.G_refiner_lr_scheduler.load_state_dict(state['g_refiner_lr_scheduler'])
            self.D_decider_lr_scheduler.load_state_dict(state['d_decider_lr_scheduler'])
            self.G_refiner2_lr_scheduler.load_state_dict(state['g_refiner2_lr_scheduler'])
            self.D_decider2_lr_scheduler.load_state_dict(state['d_decider2_lr_scheduler'])
        if self.reset_lr:
            self.G_optimizer.param_groups[0]['lr'] = self.config.G_LR
            self.D_optimizer.param_groups[0]['lr'] = self.config.D_LR
            self.G_refiner_optimizer.param_groups[0]['lr'] = self.config.G_REFINER_LR
            self.D_decider_optimizer.param_groups[0]['lr'] = self.config.D_DECIDER_LR
            self.G_refiner2_optimizer.param_groups[0]['lr'] = self.config.G_REFINER2_LR
            self.D_decider2_optimizer.param_groups[0]['lr'] = self.config.D_DECIDER2_LR
        self.set_state_dict()

    def set_state_dict(self):
        self.state_dict['g'] = self.G.state_dict()
        self.state_dict['g_refiner'] = self.G_refiner.state_dict()
        self.state_dict['g_refiner2'] = self.G_refiner2.state_dict()
        if self.mode == 'train':
            self.state_dict['d'] = self.D.state_dict()
            self.state_dict['d_decider'] = self.D_decider.state_dict()
            self.state_dict['d_decider2'] = self.D_decider2.state_dict()
            self.state_dict['g_optim'] = self.G_optimizer.state_dict()
            self.state_dict['d_optim'] = self.D_optimizer.state_dict()
            self.state_dict['g_refiner_optim'] = self.G_refiner_optimizer.state_dict()
            self.state_dict['d_decider_optim'] = self.D_decider_optimizer.state_dict()
            self.state_dict['g_refiner2_optim'] = self.G_refiner2_optimizer.state_dict()
            self.state_dict['d_decider2_optim'] = self.D_decider2_optimizer.state_dict()
            self.state_dict['g_lr_scheduler'] = self.G_lr_scheduler.state_dict()
            self.state_dict['d_lr_scheduler'] = self.D_lr_scheduler.state_dict()
            self.state_dict['g_refiner_lr_scheduler'] = self.G_refiner_lr_scheduler.state_dict()
            self.state_dict['d_decider_lr_scheduler'] = self.D_decider_lr_scheduler.state_dict()
            self.state_dict['g_refiner2_lr_scheduler'] = self.G_refiner2_lr_scheduler.state_dict()
            self.state_dict['d_decider2_lr_scheduler'] = self.D_decider2_lr_scheduler.state_dict()

    def save_model_dict(self, epoch, iteration, loss_g, loss_d, loss_g_refiner, loss_d_decider, loss_g_refiner2, loss_d_decider2):
        model_filename = "{}_{:04}_{:08}_{:.4f}_{:.4f}_{:.4f}_{:.4f}_{:.4f}_{:.4f}.pth".format(\
                         self.model_name, epoch, iteration, loss_g, loss_d, loss_g_refiner, loss_d_decider, loss_g_refiner2, loss_d_decider2)
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
        phase, epoch, iteration, loss_g, loss_d, loss_g_refiner, loss_d_decider, loss_g_refiner2, loss_d_decider2,\
            acc_rr, acc_rf, acc_fr, acc_decider_rr, acc_decider_fr, acc_decider2_rr, acc_decider2_fr = log_tuple
        log_file = self.train_log_file if phase == 'train' else self.val_log_file
        log_row_str = '{},{},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f}\n'.format(
                                                                                                                                 epoch,
                                                                                                                                 iteration,
                                                                                                                                 loss_g,
                                                                                                                                 loss_d,
                                                                                                                                 loss_g_refiner,
                                                                                                                                 loss_d_decider,
                                                                                                                                 loss_g_refiner2,
                                                                                                                                 loss_d_decider2,
                                                                                                                                 acc_rr,
                                                                                                                                 acc_rf,
                                                                                                                                 acc_fr,
                                                                                                                                 acc_decider_rr,
                                                                                                                                 acc_decider_fr,
                                                                                                                                 acc_decider2_rr,
                                                                                                                                 acc_decider2_fr)
        with open(log_file, 'a') as f:
            f.write(log_row_str)

    def set_requires_grad(self, net, requires_grad=False):
        for param in net.parameters():
            param.requires_grad = requires_grad
        return net

    def set_inputs(self, data, fake_images, refined1, refined2):
        real_first_images, real_second_images, real_images, real_wvs, fake_wvs = data

        real_wvs = real_wvs.view(self.batch_size, -1)
        fake_wvs = fake_wvs.view(self.batch_size, -1)

        # ## Make pairs
        # real_real_first_pair = (real_first_images, real_wv)
        # real_real_second_pair = (real_second_images, real_wv)
        # real_real_pair = (real_images, real_wv)
        # real_fake_pair = (real_first_images, fake_wv)
        # fake_real_pair = (fake_images, real_wv)
        # refined1_real_pair = (refined1, real_wv)
        # refined2_real_pair = (refined2, real_wv)

        return real_first_images, real_second_images, real_images,\
               fake_images, refined1, refined2,\
               real_wvs, fake_wvs
        # return real_real_first_pair, real_real_second_pair, real_real_pair, real_fake_pair, \
        #        fake_real_pair, refined1_real_pair, refined2_real_pair

    def backward_D(self, real_first_images, fake_images, real_wvs, fake_wvs, update=True, prob_flip_labels=0.0):

        # Real-real
        pred_rr = self.D(real_first_images, real_wvs)
        loss_D_rr, self.accuracy_D_rr = self.D_criterionGAN(pred_rr, target_is_real=True, prob_flip_labels=prob_flip_labels)
        if update:
            loss_D_rr.backward()

        ## Fake-real
        pred_fr = self.D(fake_images.detach(), real_wvs)
        loss_D_fr, self.accuracy_D_fr = self.D_criterionGAN(pred_fr, target_is_real=False, prob_flip_labels=prob_flip_labels)
        if update:
            loss_D_fr.backward()

        ## Real-fake
        # pred_rf = self.D(real_first_images, fake_wvs)
        # loss_D_rf, self.accuracy_D_rf = self.D_criterionGAN(pred_rf, target_is_real=False, prob_flip_labels=prob_flip_labels)
        # if update:
        #     loss_D_rf.backward()

        if self.gan_loss1 == 'wgangp':
            self.loss_gp_fr, _, _ = get_paired_gradient_penalty(self.D, (real_first_images, real_wvs), (fake_images, real_wvs), self.device,
                                                   type='mixed', constant=1.0, lambda_gp=10.0)
            # self.loss_gp_fr, _ = get_single_gradient_penalty(self.D, real_first_images, fake_images, self.device,
            #                                        type='mixed', constant=1.0, lambda_gp=10.0)
            # self.loss_gp_rf, _, _ = get_paired_gradient_penalty(self.D, (real_first_images, real_wvs), (real_first_images, fake_wvs), self.device,
            #                                        type='mixed', constant=1.0, lambda_gp=10.0)
            if update:
                self.loss_gp_fr.backward(retain_graph=True)
                # self.loss_gp_rf.backward(retain_graph=True)

            self.loss_D = loss_D_rr + loss_D_fr + self.loss_gp_fr
        else:
            self.loss_D = loss_D_rr + loss_D_fr

    def backward_D_decider(self, real_second_images, refined1, update=True, prob_flip_labels=0.0):

        # Real-real
        pred_rr = self.D_decider(real_second_images)
        loss_D_decider_rr, self.accuracy_D_decider_rr = self.D_decider_criterionGAN(pred_rr, target_is_real=True, prob_flip_labels=prob_flip_labels)
        if update:
            loss_D_decider_rr.backward()

        ## Fake refined-real
        pred_refined_fr = self.D_decider(refined1.detach())
        loss_D_decider_fr, self.accuracy_D_decider_fr = self.D_decider_criterionGAN(pred_refined_fr, target_is_real=False, prob_flip_labels=prob_flip_labels)
        if update:
            loss_D_decider_fr.backward()

        if self.gan_loss2 == 'wgangp':
            self.loss_gp_decider_fr, _ = get_single_gradient_penalty(self.D_decider, real_second_images, refined1, self.device,
                                                                     type='mixed', constant=1.0, lambda_gp=10.0)
            if update:
                self.loss_gp_decider_fr.backward(retain_graph=True)

            self.loss_D_decider = loss_D_decider_rr + loss_D_decider_fr + self.loss_gp_decider_fr
        else:
            self.loss_D_decider = loss_D_decider_rr + loss_D_decider_fr

    def backward_D_decider2(self, real_images, refined2, update=True, prob_flip_labels=0.0):

        # Real-real
        pred_rr = self.D_decider2(real_images)
        loss_D_decider2_rr, self.accuracy_D_decider2_rr = self.D_decider2_criterionGAN(pred_rr, target_is_real=True, prob_flip_labels=prob_flip_labels)
        if update:
            loss_D_decider2_rr.backward()

        ## Fake refined-real
        pred_refined_fr = self.D_decider2(refined2.detach())
        loss_D_decider2_fr, self.accuracy_D_decider2_fr = self.D_decider2_criterionGAN(pred_refined_fr, target_is_real=False, prob_flip_labels=prob_flip_labels)
        if update:
            loss_D_decider2_fr.backward()

        if self.gan_loss2 == 'wgangp':
            self.loss_gp_decider2_fr, _ = get_single_gradient_penalty(self.D_decider2, real_images, refined2, self.device,
                                                                      type='mixed', constant=1.0, lambda_gp=10.0)
            if update:
                self.loss_gp_decider2_fr.backward(retain_graph=True)

            self.loss_D_decider2 = loss_D_decider2_rr + loss_D_decider2_fr + self.loss_gp_decider2_fr
        else:
            self.loss_D_decider2 = loss_D_decider2_rr + loss_D_decider2_fr

    def backward_G(self, real_first_images, fake_images, real_wvs, update=True, prob_flip_labels=0.0):

        ## Fake-real
        pred_fr = self.D(fake_images, real_wvs)

        loss_G_dist = self.G_criterion_dist(fake_images, real_first_images) * self.lambda_l1
        loss_G_GAN, _ = self.G_criterionGAN(pred_fr, target_is_real=True, prob_flip_labels=prob_flip_labels)

        if update:
            loss_G_dist.backward(retain_graph=True)
            loss_G_GAN.backward(retain_graph=True)

        self.loss_G = loss_G_dist + loss_G_GAN

    def backward_G_refiner(self, real_second_images, refined1, update=True, prob_flip_labels=0.0):

        ## Fake-real
        pred_refined_fr = self.D_decider(refined1)

        loss_G_refiner_dist = self.G_refiner_criterion_dist(refined1, real_second_images) * self.lambda_l1
        loss_G_refiner_GAN, _ = self.G_refiner_criterionGAN(pred_refined_fr, target_is_real=True, prob_flip_labels=prob_flip_labels)

        if update:
            loss_G_refiner_dist.backward(retain_graph=True)
            loss_G_refiner_GAN.backward(retain_graph=True)

        self.loss_G_refiner = loss_G_refiner_dist + loss_G_refiner_GAN

    def backward_G_refiner2(self, real_images, refined2, update=True, prob_flip_labels=0.0):

        ## Fake-real
        pred_refined_fr = self.D_decider2(refined2)

        loss_G_refiner2_dist = self.G_refiner2_criterion_dist(refined2, real_images) * self.lambda_l1
        loss_G_refiner2_GAN, _ = self.G_refiner2_criterionGAN(pred_refined_fr, target_is_real=True, prob_flip_labels=prob_flip_labels)

        if update:
            loss_G_refiner2_dist.backward(retain_graph=True)
            loss_G_refiner2_GAN.backward(retain_graph=True)

        self.loss_G_refiner2 = loss_G_refiner2_dist + loss_G_refiner2_GAN

    def get_losses(self):
        loss_g = self.loss_G.item() if self.loss_G else -1.0
        loss_d = self.loss_D.item() if self.loss_D else -1.0
        loss_g_refiner = self.loss_G_refiner.item() if self.loss_G_refiner else -1.0
        loss_d_decider = self.loss_D_decider.item() if self.loss_D_decider else -1.0
        loss_g_refiner2 = self.loss_G_refiner2.item() if self.loss_G_refiner2 else -1.0
        loss_d_decider2 = self.loss_D_decider2.item() if self.loss_D_decider2 else -1.0
        loss_gp_fr = self.loss_gp_fr.item() if self.loss_gp_fr else -1.0
        loss_gp_rf = self.loss_gp_rf.item() if self.loss_gp_rf else -1.0
        loss_gp_decider_fr = self.loss_gp_decider_fr.item() if self.loss_gp_decider_fr else -1.0
        loss_gp_decider2_fr = self.loss_gp_decider2_fr.item() if self.loss_gp_decider2_fr else -1.0
        return loss_g, loss_d, loss_g_refiner, loss_d_decider, loss_g_refiner2, loss_d_decider2,\
               loss_gp_fr, loss_gp_rf, loss_gp_decider_fr, loss_gp_decider2_fr

    def get_D_accuracy(self):
        return self.accuracy_D_rr, self.accuracy_D_rf, self.accuracy_D_fr,\
               self.accuracy_D_decider_rr, self.accuracy_D_decider_fr, self.accuracy_D_decider2_rr, self.accuracy_D_decider2_fr

    def update_lr(self, loss_g, loss_d, loss_g_refiner, loss_d_decider, loss_g_refiner2, loss_d_decider2):
        self.G_lr_scheduler.step(loss_g)
        self.D_lr_scheduler.step(loss_d)
        self.G_refiner_lr_scheduler.step(loss_g_refiner)
        self.D_decider_lr_scheduler.step(loss_d_decider)
        self.G_refiner2_lr_scheduler.step(loss_g_refiner2)
        self.D_decider2_lr_scheduler.step(loss_d_decider2)
        G_lr = self.G_optimizer.param_groups[0]['lr']
        D_lr = self.D_optimizer.param_groups[0]['lr']
        G_refiner_lr = self.G_refiner_optimizer.param_groups[0]['lr']
        D_decider_lr = self.D_decider_optimizer.param_groups[0]['lr']
        G_refiner2_lr = self.G_refiner2_optimizer.param_groups[0]['lr']
        D_decider2_lr = self.D_decider2_optimizer.param_groups[0]['lr']

        print('\t\t(G learning rate is {:.4E})'.format(G_lr))
        print('\t\t(D learning rate is {:.4E})'.format(D_lr))
        print('\t\t(G_refiner learning rate is {:.4E})'.format(G_refiner_lr))
        print('\t\t(D_decider learning rate is {:.4E})'.format(D_decider_lr))
        print('\t\t(G_refiner2 learning rate is {:.4E})'.format(G_refiner2_lr))
        print('\t\t(D_decider2 learning rate is {:.4E})'.format(D_decider2_lr))

    def generate_grid(self, real_wvs, fake_images, refined1, refined2, real_images, word2vec_model):

        images_bag = []
        for real_wv, fake_image, _refined1, _refined2, real_image in zip(real_wvs, fake_images, refined1, refined2, real_images):

            ## Get words from word vectors
            words = []
            for _real_wv in real_wv:
                _real_wv = np.array(_real_wv)
                word, prob = word2vec_model.wv.similar_by_vector(_real_wv)[0]
                if prob > self.min_similarity_prob:  ## Eliminate noise words
                    words.append(word)

            ## Unique words are visualized by converting into image
            words = np.unique(words)
            word_image = words2image(words, self.config)

            ## Inverse normalize
            if self.inv_normalize:
                fake_image = self.inverse_normalizer(fake_image)
                _refined1 = self.inverse_normalizer(_refined1)
                _refined2 = self.inverse_normalizer(_refined2)
                real_image = self.inverse_normalizer(real_image)

            ## Go to cpu numpy array
            fake_image = fake_image.detach().cpu().numpy().transpose(1, 2, 0)
            fake_image = np.array(self.image_resizer(Image.fromarray((fake_image * 255.).astype(np.uint8)))) / 255.
            _refined1 = _refined1.detach().cpu().numpy().transpose(1, 2, 0)
            _refined1 = np.array(self.image_resizer(Image.fromarray((_refined1 * 255.).astype(np.uint8)))) / 255.
            _refined2 = _refined2.detach().cpu().numpy().transpose(1, 2, 0)
            real_image = real_image.detach().cpu().numpy().transpose(1, 2, 0)

            images_bag.extend([word_image, fake_image, _refined1, _refined2, real_image])

        images_bag = np.array(images_bag)
        grid = make_grid(torch.Tensor(images_bag.transpose(0, 3, 1, 2)), nrow=self.config.N_GRID_ROW).permute(1, 2, 0)
        grid_pil = Image.fromarray(np.array(grid * 255, dtype=np.uint8))
        return grid_pil

    def save_img_output(self, img_pil, filename):
        output_dir = os.path.join(self.model_dir, 'output')
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, 'G_img_grid_' + filename)
        img_pil.save(output_file)

    def save_grad_output(self, filename):
        output_dir = os.path.join(self.model_dir, 'output')
        os.makedirs(output_dir, exist_ok=True)
        G_output_file = os.path.join(output_dir, 'G_grads_' + filename)
        D_output_file = os.path.join(output_dir, 'D_grads' + filename)

        if np.all([param.requires_grad for param in self.G.parameters()]):
            get_G_dot = register_hooks(self.loss_G)
            try:
                G_dot = get_G_dot()
            except AssertionError:
                return
            G_dot.save(G_output_file)
        if np.all([param.requires_grad for param in self.D.parameters()]):
            get_D_dot = register_hooks(self.loss_D)
            try:
                D_dot = get_D_dot()
            except AssertionError:
                return
            D_dot.save(D_output_file)

    def forward(self, net, x):
        return net(x)

    def fit(self, data, phase='train', train_D=True, train_G=True, classic=True):
        ## Data to device
        real_first_images, real_second_images, real_images, real_wvs, fake_wvs = data
        real_first_images = real_first_images.to(self.device)
        real_second_images = real_second_images.to(self.device)
        real_images = real_images.to(self.device)
        real_wvs = real_wvs.to(self.device)
        fake_wvs = fake_wvs.to(self.device)
        data = real_first_images, real_second_images, real_images, real_wvs, fake_wvs

        ## Forward G
        real_wvs_flat = real_wvs.view(self.batch_size, -1)
        fake_images = self.forward(self.G, real_wvs_flat)

        ## Forward G_refiner
        refined1 = self.forward(self.G_refiner, fake_images)
        
        ## Forward G_refiner2
        refined2 = self.forward(self.G_refiner2, refined1)

        ## Set inputs
        real_first_images, real_second_images, real_images, fake_images, refined1, refined2, real_wvs, fake_wvs =\
            self.set_inputs(data, fake_images, refined1, refined2)

        if phase == 'train':

            ## Update D
            self.D = self.set_requires_grad(self.D, train_D)
            # all_true = np.all([param.requires_grad for param in self.D.parameters()])
            # print("All D parameters have grad:", str(all_true))
            self.D_optimizer.zero_grad()
            self.backward_D(real_first_images, fake_images, real_wvs, fake_wvs, update=train_D, prob_flip_labels=self.prob_flip_labels)
            if train_D:
                self.D_optimizer.step()

            ## Update G
            self.D = self.set_requires_grad(self.D, False)      # Disable backprop for D
            self.G = self.set_requires_grad(self.G, train_G)
            self.G_optimizer.zero_grad()
            self.backward_G(real_first_images, fake_images, real_wvs, update=train_G, prob_flip_labels=self.prob_flip_labels)
            if train_G:
                self.G_optimizer.step()
            ## -------------------------------------

            ## Update D_decider
            self.D_decider = self.set_requires_grad(self.D_decider, train_D)
            self.D_decider_optimizer.zero_grad()
            self.backward_D_decider(real_second_images, refined1, update=train_D, prob_flip_labels=self.prob_flip_labels)
            if train_D:
                self.D_decider_optimizer.step()

            ## Update G_refiner
            self.D_decider = self.set_requires_grad(self.D_decider, False)      # Disable backprop for D
            self.G_refiner = self.set_requires_grad(self.G_refiner, train_G)
            self.G_refiner_optimizer.zero_grad()
            self.backward_G_refiner(real_second_images, refined1, update=train_G, prob_flip_labels=self.prob_flip_labels)
            if train_G:
                self.G_refiner_optimizer.step()
            ## -------------------------------------

            ## Update D_decider2
            self.D_decider2 = self.set_requires_grad(self.D_decider2, train_D)
            self.D_decider2_optimizer.zero_grad()
            self.backward_D_decider2(real_images, refined2, update=train_D, prob_flip_labels=self.prob_flip_labels)
            if train_D:
                self.D_decider2_optimizer.step()

            ## Update G_refiner
            self.D_decider2 = self.set_requires_grad(self.D_decider2, False)      # Disable backprop for D
            self.G_refiner2 = self.set_requires_grad(self.G_refiner2, train_G)
            self.G_refiner2_optimizer.zero_grad()
            self.backward_G_refiner2(real_images, refined2, update=train_G, prob_flip_labels=self.prob_flip_labels)
            if train_G:
                self.G_refiner2_optimizer.step()

        else:
            self.backward_D(real_first_images, fake_images, real_wvs, fake_wvs, update=False, prob_flip_labels=0.0)
            self.backward_G(real_first_images, fake_images, real_wvs, update=False, prob_flip_labels=0.0)
            self.backward_D_decider(real_second_images, refined1, update=False, prob_flip_labels=0.0)
            self.backward_G_refiner(real_second_images, refined1, update=False, prob_flip_labels=0.0)
            self.backward_D_decider2(real_images, refined2, update=False, prob_flip_labels=0.0)
            self.backward_G_refiner2(real_images, refined2, update=False, prob_flip_labels=0.0)

        return fake_images, refined1, refined2