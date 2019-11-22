
import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler

def conv3x3(in_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False):
    return nn.Conv2d(
                     in_planes,
                     out_planes,
                     kernel_size=kernel_size,
                     stride=stride,
                     padding=padding,
                     bias=bias
                     )


def upsample_block(in_planes, out_planes):
    # Upsample the spatial size by a factor of 2
    block = nn.Sequential(
                         nn.Upsample(scale_factor=2, mode='nearest'),
                         conv3x3(in_planes, out_planes),
                         nn.BatchNorm2d(out_planes),
                        #  nn.InstanceNorm2d(out_planes),
                         nn.ReLU(True)
                         )

    return block

class ResBlock(nn.Module):
    def __init__(self, n_channels):
        super(ResBlock, self).__init__()

        self.block = nn.Sequential(
                                  conv3x3(n_channels, n_channels),
                                  nn.BatchNorm2d(n_channels),
                                  nn.ReLU(True),
                                  conv3x3(n_channels, n_channels),
                                  nn.BatchNorm2d(n_channels)
                                  )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.block(x)
        out += residual
        out = self.relu(out)
        return out


class Generator(nn.Module):
    def __init__(self, config):
        super(Generator, self).__init__()

        n_input = config.N_INPUT
        ngf = config.NGF * 8
        n_channel = config.N_CHANNELS
        self.n_input = n_input
        self.ngf = ngf

        self.fc = nn.Sequential(
                               nn.Linear(n_input, ngf * 4 * 4, bias=False),
                               nn.BatchNorm1d(ngf * 4 * 4),
                            #    nn.InstanceNorm1d(ngf * 4 * 4),
                               nn.ReLU(True)
                               )                                 # -> ngf x H x W

        self.upsample1 = upsample_block(ngf, ngf // 2)           # ngf x H x W -> ngf/2 x 2H x 2W
        self.upsample2 = upsample_block(ngf // 2, ngf // 4)      # -> ngf/4 x 4H x 4W
        self.upsample3 = upsample_block(ngf // 4, ngf // 8)      # -> ngf/8 x 8H x 8W
        self.upsample4 = upsample_block(ngf // 8, ngf // 16)     # -> ngf/16 x 16H x 16W

        self.dropout = nn.Dropout2d(0.5)

        self.image = nn.Sequential(
                                  conv3x3(ngf // 16, n_channel),
                                  nn.Tanh()
                                  )                              # -> 3 x 16H x 16W

    def forward(self, word_vectors):
        # out = self.dropout(word_vectors)
        out = self.fc(word_vectors)
        out = out.view(-1, self.ngf, 4, 4)
        out = self.upsample1(out)
        out = self.upsample2(out)
        out = self.upsample3(out)
        out = self.upsample4(out)
        out = self.dropout(out)
        out = self.image(out)

        return out

### DO NOT USE BatchNorm in D if WGANGP is used as loss function ###
class Discriminator(nn.Module):
    def __init__(self, config):
        super(Discriminator, self).__init__()
        ndf = config.NDF
        ngf = config.NGF
        n_channel = config.N_CHANNELS * 2   ## Stitching images and word vectors
        self.ndf = ndf
        self.ngf = ngf

        self.conv = nn.Sequential(                                                        # (6) x H x W
                                 nn.Conv2d(n_channel, ndf, 4, 2, 1, bias=False),    # (ndf) x H/2 x W/2
                                 nn.LeakyReLU(0.2, inplace=True),
                                 nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),      # (ndf * 2) x H/4 x W/4
                                 nn.BatchNorm2d(ndf * 2),
                                #  nn.InstanceNorm2d(ndf * 2),
                                 nn.LeakyReLU(0.2, inplace=True),
                                 nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),    # (ndf * 4) x H/4 x W/4
                                 nn.BatchNorm2d(ndf * 4),
                                #  nn.InstanceNorm2d(ndf * 4),
                                 nn.LeakyReLU(0.2, inplace=True),
                                 nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),    # (ndf * 8) x H/16 x W/16
                                 nn.BatchNorm2d(ndf * 8),
                                #  nn.InstanceNorm2d(ndf * 8),
                                 nn.LeakyReLU(0.2, inplace=True),
                                #  nn.Dropout2d(0.5, True),
                                 )

        self.get_cond_logits = DiscriminatorLogits(ndf, ngf, bcondition=False)
        self.get_uncond_logits = None

    def forward(self, image):
        out = self.conv(image)

        return out


class DiscriminatorLogits(nn.Module):
    def __init__(self, ndf, ngf, bcondition=True):
        super(DiscriminatorLogits, self).__init__()
        self.ndf = ndf
        self.ngf = ngf
        self.bcondition = bcondition
        if bcondition:
            self.outlogits = nn.Sequential(
                                          conv3x3(ndf * 8 + ngf, ndf * 8),
                                          nn.BatchNorm2d(ndf * 8),
                                        #   nn.InstanceNorm2d(ndf * 8),
                                          nn.LeakyReLU(0.2, inplace=True),
                                        #   nn.Dropout2d(0.5, True),
                                          nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=4),
                                          )
        else:
            self.outlogits = nn.Sequential(
                                        #   nn.Dropout2d(0.5, True),
                                          nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=4),
                                          )

    def forward(self, h_code, c_code=None):
        # conditioning output
        if self.bcondition and c_code is not None:
            c_code = c_code.view(-1, self.ngf, 1, 1)
            c_code = c_code.repeat(1, 1, 4, 4)
            h_c_code = torch.cat((h_code, c_code), 1)    # (ngf + ndf) x 4 x 4
        else:
            h_c_code = h_code

        output = self.outlogits(h_c_code)
        return output.view(-1)



## ------------------------------------------------



class GeneratorAlt1(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        
        self.main = nn.Sequential(
            ## 1 x 2000
            nn.ConvTranspose2d(1 * 2000, 64, kernel_size=4, stride=1, padding=0, bias=False),  
            ## 64 x 4 x 4
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1, bias=False),
            ## 32 x 8 x 8
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1, bias=False),
            ## 16 x 16 x 16
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1, bias=False),
            ## 8 x 32 x 32
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 3, kernel_size=4, stride=2, padding=1, bias=False),
            ## 3 x 64 x 64
            nn.Tanh()
        )

    def forward(self, x):
        return self.main(x)

class DiscriminatorAlt1(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(                                                ## 3 x H x W
            nn.Conv2d(3, 16, kernel_size=4, stride=2, padding=1, bias=False),     ## 16 x H/2 x W/2
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1, bias=False),    ## 32 x H/4 x W/4
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1, bias=False),    ## 64 x H/8 x W/8
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),   ## 128 x H/16 x W/16
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 1, kernel_size=4, stride=1, padding=0, bias=False),    ## 1 x H/64 x W/64
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)

# class Discriminator(nn.Module):
#     def __init__(self, config):
#         super(Discriminator, self).__init__()

#         ndf = config.NDF
#         ngf = config.NGF
#         self.ndf = ndf
#         self.ngf = ngf

#         self.conv = nn.Sequential(
#                                        nn.Conv2d(3, ndf, 4, 2, 1, bias=False),            # 128 * 128 * ndf
#                                        nn.LeakyReLU(0.2, inplace=True),
#                                        nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
#                                        nn.BatchNorm2d(ndf * 2),
#                                        nn.LeakyReLU(0.2, inplace=True),                   # 64 * 64 * ndf * 2
#                                        nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
#                                        nn.BatchNorm2d(ndf * 4),
#                                        nn.LeakyReLU(0.2, inplace=True),                   # 32 * 32 * ndf * 4
#                                        nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
#                                        nn.BatchNorm2d(ndf * 8),
#                                        nn.LeakyReLU(0.2, inplace=True),                   # 16 * 16 * ndf * 8
#                                        nn.Conv2d(ndf * 8, ndf * 16, 4, 2, 1, bias=False),
#                                        nn.BatchNorm2d(ndf * 16),
#                                        nn.LeakyReLU(0.2, inplace=True),                   # 8 * 8 * ndf * 16
#                                        nn.Conv2d(ndf * 16, ndf * 32, 4, 2, 1, bias=False),
#                                        nn.BatchNorm2d(ndf * 32),
#                                        nn.LeakyReLU(0.2, inplace=True),                   # 4 * 4 * ndf * 32
#                                        conv3x3(ndf * 32, ndf * 16),
#                                        nn.BatchNorm2d(ndf * 16),
#                                        nn.LeakyReLU(0.2, inplace=True),                   # 4 * 4 * ndf * 16
#                                        conv3x3(ndf * 16, ndf * 8),
#                                        nn.BatchNorm2d(ndf * 8),
#                                        nn.LeakyReLU(0.2, inplace=True)                    # 4 * 4 * ndf * 8
#                                        )

#         self.get_cond_logits = DiscriminatorLogits(ndf, ngf, bcondition=True)
#         self.get_uncond_logits = DiscriminatorLogits(ndf, ngf, bcondition=False)

#     def forward(self, x):
#         out = self.conv(x)

#         return out

# class EncoderRNN(nn.Module):
#     def __init__(self, input_size, hidden_size, n_layers=1):
#         super(EncoderRNN, self).__init__()
#         self.input_size = input_size
#         self.hidden_size = hidden_size

#         self.gru = nn.GRU(hidden_size, hidden_size, n_layers)

#     def forward(self, x, hidden):
#         output, hidden = self.gru(x, hidden)
#         return output, hidden

#     def init_hidden(self):
#         return torch.zeros(1, self.input_size, self.hidden_size)