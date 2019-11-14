# import torch


# class Pix2PixModel(object):
#     def __init__(self, config, mode='train'):

#         self.mode = mode

#         # Define networks
#         self.G = networks.define_G(opt.input_nc,
#                                    opt.output_nc,
#                                    opt.ngf,
#                                    opt.netG,
#                                    opt.norm,
#                                    not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

#         # Define D; channels for D is input_nc + output_nc
#         if self.mode == 'train':
#             self.D = networks.define_D(opt.input_nc + opt.output_nc,
#                                        opt.ndf,
#                                        opt.netD,
#                                        opt.n_layers_D,
#                                        opt.norm,
#                                        opt.init_type,
#                                        opt.init_gain,
#                                        self.gpu_ids)

#         if self.mode == 'train':
#             # define loss functions
#             self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
#             self.criterionL1 = torch.nn.L1Loss()
#             # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
#             self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
#             self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
#             self.optimizers.append(self.optimizer_G)
#             self.optimizers.append(self.optimizer_D)

#     def set_input(self, input):
#         """Unpack input data from the dataloader and perform necessary pre-processing steps.
#         Parameters:
#             input (dict): include the data itself and its metadata information.
#         The option 'direction' can be used to swap images in domain A and domain B.
#         """
#         AtoB = self.opt.direction == 'AtoB'
#         self.real_A = input['A' if AtoB else 'B'].to(self.device)
#         self.real_B = input['B' if AtoB else 'A'].to(self.device)
#         self.image_paths = input['A_paths' if AtoB else 'B_paths']

#     def forward(self):
#         """Run forward pass; called by both functions <optimize_parameters> and <test>."""
#         self.fake_B = self.netG(self.real_A)  # G(A)

#     def backward_D(self):
#         """Calculate GAN loss for the discriminator"""
#         # Fake; stop backprop to the generator by detaching fake_B
#         fake_AB = torch.cat((self.real_A, self.fake_B), 1)  # we use conditional GANs; we need to feed both input and output to the discriminator
#         pred_fake = self.netD(fake_AB.detach())
#         self.loss_D_fake = self.criterionGAN(pred_fake, False)
#         # Real
#         real_AB = torch.cat((self.real_A, self.real_B), 1)
#         pred_real = self.netD(real_AB)
#         self.loss_D_real = self.criterionGAN(pred_real, True)
#         # combine loss and calculate gradients
#         self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
#         self.loss_D.backward()

#     def backward_G(self):
#         """Calculate GAN and L1 loss for the generator"""
#         # First, G(A) should fake the discriminator
#         fake_AB = torch.cat((self.real_A, self.fake_B), 1)
#         pred_fake = self.netD(fake_AB)
#         self.loss_G_GAN = self.criterionGAN(pred_fake, True)
#         # Second, G(A) = B
#         self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_L1
#         # combine loss and calculate gradients
#         self.loss_G = self.loss_G_GAN + self.loss_G_L1
#         self.loss_G.backward()

#     def optimize_parameters(self):
#         self.forward()                   # compute fake images: G(A)
#         # update D
#         self.set_requires_grad(self.netD, True)  # enable backprop for D
#         self.optimizer_D.zero_grad()     # set D's gradients to zero
#         self.backward_D()                # calculate gradients for D
#         self.optimizer_D.step()          # update D's weights
#         # update G
#         self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
#         self.optimizer_G.zero_grad()        # set G's gradients to zero
#         self.backward_G()                   # calculate graidents for G
#         self.optimizer_G.step()             # udpate G's weights