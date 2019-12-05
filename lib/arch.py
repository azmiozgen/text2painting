
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

class ResnetBlock(nn.Module):

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)  # add skip connections
        return out


class GeneratorResNet(nn.Module):

    def __init__(self, config):
        super(GeneratorResNet, self).__init__()

        n_input = config.N_INPUT
        ngf = config.NGF
        self.ngf = ngf
        norm_layer = config.NORM_LAYER
        use_dropout = config.USE_DROPOUT
        n_blocks = config.N_BLOCKS
        padding_type = config.PADDING_TYPE
        assert(n_blocks >= 0)

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.fc = nn.Sequential(
                               nn.Linear(n_input, ngf * 4 * 4, bias=False),
                            #    nn.BatchNorm1d(ngf * 4 * 4),
                               nn.ReLU(True)
                               )                              # -> ngf x H x W

        self.start = nn.Sequential(nn.ReflectionPad2d(3),
                                   nn.Conv2d(ngf, ngf, kernel_size=7, padding=0, bias=use_bias),
                                   norm_layer(ngf),
                                   nn.ReLU(True))             # -> ngf x H x W

        def build_blocks(channel, n_blocks):
            blocks = []
            for i in range(n_blocks):       # add ResNet blocks
                blocks += [ResnetBlock(channel, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
            blocks = nn.Sequential(*blocks)                  # -> ngf x H x W
            return blocks

        self.block1 = build_blocks(ngf, n_blocks)
        self.upsample1 = upsample_block(ngf, ngf // 2)           # ngf x H x W -> ngf/2 x 2H x 2W
        # self.block2 = build_blocks(ngf // 2, n_blocks // 2)
        self.upsample2 = upsample_block(ngf // 2, ngf // 4)      # -> ngf/4 x 4H x 4W
        # self.block3 = build_blocks(ngf // 4, n_blocks // 2)
        self.upsample3 = upsample_block(ngf // 4, ngf // 8)      # -> ngf/8 x 8H x 8W
        # self.block4 = build_blocks(ngf // 8, n_blocks // 2)
        self.upsample4 = upsample_block(ngf // 8, ngf // 16)     # -> ngf/16 x 16H x 16W
        self.block5 = build_blocks(ngf // 16, n_blocks)
        # self.upsample5 = upsample_block(ngf // 16, ngf // 32)     # -> ngf/32 x 32H x 32W

        self.dropout = nn.Dropout2d(0.2)
        
        self.finish = nn.Sequential(nn.ReflectionPad2d(3),
                                    nn.Conv2d(ngf // 16, 3, kernel_size=7, padding=0),    # 3 x 16H x 16W
                                    nn.Tanh())

    def forward(self, word_vectors):
        out = self.fc(word_vectors)
        out = out.view(-1, self.ngf, 4, 4)
        out = self.start(out)
        out = self.block1(out)
        out = self.upsample1(out)
        out = self.upsample2(out)
        out = self.upsample3(out)
        out = self.upsample4(out)
        out = self.block5(out)
        # out = self.upsample5(out)
        out = self.finish(out)

        return out

class GeneratorRefiner(nn.Module):

    def __init__(self, config):
        super(GeneratorRefiner, self).__init__()

        n_channels = config.N_CHANNELS
        n_input = config.N_INPUT
        ngf = config.NG_REF_F
        self.ngf = ngf
        norm_layer = config.NORM_LAYER
        use_dropout = config.USE_DROPOUT
        n_blocks = config.N_BLOCKS
        padding_type = config.PADDING_TYPE
        assert(n_blocks >= 0)

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.downsample = nn.Sequential(                                                         ## 3 x H x W
            nn.Conv2d(n_channels, ngf, kernel_size=4, stride=2, padding=1, bias=False),      ## ngf x H/2 x W/2
            nn.BatchNorm2d(ngf),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ngf, ngf * 2, kernel_size=4, stride=2, padding=1, bias=False),       ## ngf x H/4 x W/4
            nn.BatchNorm2d(ngf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ngf * 2, ngf * 4, kernel_size=4, stride=2, padding=1, bias=False),            ## ngf x H/8 x W/8
            # nn.BatchNorm2d(ngf * 4),
            # nn.LeakyReLU(0.2, inplace=True),
            # nn.Conv2d(ngf * 4, ngf * 8, kernel_size=4, stride=2, padding=1, bias=False),                 ## ngf x H/16 x W/16
            # nn.BatchNorm2d(ngf * 8),
            # nn.LeakyReLU(0.2, inplace=True),
            # nn.Conv2d(ngf * 8, ngf * 8, kernel_size=4, stride=2, padding=1, bias=False),                 ## ngf x H/32 x W/32
        )

        def build_blocks(channel, n_blocks):
            blocks = []
            for i in range(n_blocks):       # add ResNet blocks
                blocks += [ResnetBlock(channel, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
            blocks = nn.Sequential(*blocks)                  # -> ngf x H x W
            return blocks

        self.block1 = build_blocks(ngf * 4, n_blocks)
        self.upsample1 = upsample_block(ngf * 4, ngf * 2)           # ngf x H x W -> ngf/2 x 2H x 2W
        self.upsample2 = upsample_block(ngf * 2, ngf)      # -> ngf/4 x 4H x 4W
        self.upsample3 = upsample_block(ngf, ngf // 2)      # -> ngf/8 x 8H x 8W
        # self.upsample4 = upsample_block(ngf, ngf // 2)     # -> ngf/16 x 16H x 16W
        # self.upsample5 = upsample_block(ngf // 2, 3)     # -> ngf/16 x 32H x 32W
        self.block5 = build_blocks(ngf // 2, n_blocks)

        self.dropout = nn.Dropout2d(0.2)
        
        self.finish = nn.Sequential(
                                    nn.ReflectionPad2d(3),
                                    nn.Conv2d(ngf // 2, 3, kernel_size=7, padding=0),    # 3 x 16H x 16W
                                    nn.Tanh()
                                   )

    def forward(self, image):
        out = self.downsample(image)
        out = self.block1(out)
        out = self.upsample1(out)
        out = self.upsample2(out)
        out = self.upsample3(out)
        # out = self.upsample4(out)
        # out = self.upsample5(out)
        out = self.block5(out)
        out = self.finish(out)

        return out

class DiscriminatorStack(nn.Module):
    def __init__(self, config):
        super(DiscriminatorStack, self).__init__()
        ndf = config.NDF
        ngf = config.NGF
        fc_in = config.N_INPUT
        fc_out = config.IMAGE_WIDTH * config.IMAGE_HEIGHT
        n_channel = config.N_CHANNELS + 1   ## Stitching images and word vectors
        self.ndf = ndf
        self.ngf = ngf

        ## No Batch Normalization
        self.fc = nn.Sequential(
                               nn.Linear(fc_in, fc_out, bias=False),
                               nn.ReLU(True)
                               )

        self.dropout = nn.Dropout(0.1)

        self.conv = nn.Sequential(                                                       ## 4 x H x W
            nn.Conv2d(n_channel, ndf, kernel_size=4, stride=2, padding=1, bias=False),           ## ndf x H/2 x W/2
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1, bias=False),     ## ndf * 2 x H/4 x W/4
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1, bias=False), ## ndf * 4 x H/8 x W/8
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=2, padding=1, bias=False), ## ndf * 8 x H/16 x W/16
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=2, padding=1, bias=False),       ## ndf * 8 x H/32 x W/32
            # nn.BatchNorm2d(ndf * 8),
            # nn.LeakyReLU(0.2, inplace=True),
            # nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=2, padding=1, bias=False),       ## 1 x H/64 x W/64
            # nn.Dropout2d(0.2)
        )


    def forward(self, image, word_vectors):
        b, _, h, w = image.size()
        wv_out = self.fc(word_vectors)
        wv_out = wv_out.view(b, 1, h, w)
        # wv_out = self.dropout(wv_out)

        stacked = torch.cat((image, wv_out), dim=1)

        return self.conv(stacked)

class DiscriminatorDecider(nn.Module):
    def __init__(self, config):
        super(DiscriminatorDecider, self).__init__()
        ndf = config.ND_DEC_F
        n_channel = config.N_CHANNELS

        self.conv = nn.Sequential(                                                       ## 3 x H x W
            nn.Conv2d(n_channel, ndf, kernel_size=4, stride=2, padding=1, bias=False),           ## ndf x H/2 x W/2
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1, bias=False),     ## ndf * 2 x H/4 x W/4
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1, bias=False), ## ndf * 4 x H/8 x W/8
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=2, padding=1, bias=False), ## ndf * 8 x H/16 x W/16
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=2, padding=1, bias=False),       ## ndf * 8 x H/32 x W/32
            # nn.Dropout2d(0.2)
            # nn.BatchNorm2d(ndf * 8),
            # nn.LeakyReLU(0.2, inplace=True),
            # nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=2, padding=1, bias=False),       ## 1 x H/64 x W/64
        )

    def forward(self, image):
        return self.conv(image)


class GeneratorSimple(nn.Module):
    def __init__(self, config):
        super(GeneratorSimple, self).__init__()
        
        n_input = config.N_INPUT
        ngf = config.NGF

        self.conv = nn.Sequential(
            nn.ConvTranspose2d(n_input, ngf * 8, kernel_size=4, stride=1, padding=0, bias=False),       ## ngf * 8 x 4 x 4 
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 8, ngf * 4, kernel_size=4, stride=2, padding=1, bias=False),      ## ngf * 4 x 8 x 8
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, kernel_size=4, stride=2, padding=1, bias=False),  ## ngf * 2 x 16 x 16
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, ngf, kernel_size=4, stride=2, padding=1, bias=False),        ## 3 x 32 x 32
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf, 3, kernel_size=4, stride=2, padding=1, bias=False),     ## 3 x 64 x 64
            nn.Tanh()
        )

    def forward(self, x):
        return self.conv(x)

class DiscriminatorSimple(nn.Module):
    def __init__(self, config):
        super(DiscriminatorSimple, self).__init__()

        n_input = config.N_INPUT
        ndf = config.NDF

        self.conv = nn.Sequential(                                                       ## 3 x H x W
            nn.Conv2d(3, ndf, kernel_size=4, stride=2, padding=1, bias=False),           ## ndf x H/2 x W/2
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1, bias=False),     ## ndf * 2 x H/4 x W/4
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1, bias=False), ## ndf * 4 x H/8 x W/8
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=2, padding=1, bias=False), ## ndf * 8 x H/16 x W/16
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=2, padding=1, bias=False),       ## 1 x H/32 x W/32
        )

    def forward(self, x):
        return self.conv(x)

class GeneratorStack(nn.Module):

    def __init__(self, config):
        super(GeneratorStack, self).__init__()

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
        self.upsample5 = upsample_block(ngf // 16, ngf // 32)     # -> ngf/32 x 32H x 32W

        self.dropout = nn.Dropout2d(0.5)

        self.image = nn.Sequential(
                                  conv3x3(ngf // 32, n_channel),
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
        out = self.upsample5(out)
        out = self.dropout(out)
        out = self.image(out)

        return out

if __name__ == '__main__':
    from config import Config
    config = Config()
    G = GeneratorResNet(config)
    image = torch.Tensor(2, 4096)
    G(image)