import glob
import os
import sys
import time
from multiprocessing import cpu_count

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageFilter
import torch
import torch.nn as nn

from gensim.models import Word2Vec
from lib.arch import Generator, Discriminator
from lib.config import Config
from lib.dataset import AlignCollate, ImageBatchSampler, TextArtDataLoader
from lib.model import GANModel
from torch.utils.data import DataLoader

DATA_DIR = 'united_small'
BATCH_SIZE = 2
N_EPOCHS = 10
# N_WORKERS = cpu_count() - 1
N_WORKERS = 0
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

CONFIG = Config()

if __name__ == "__main__":

    ## Data loaders
    train_dataset = TextArtDataLoader(DATA_DIR, CONFIG, mode='train')
    val_dataset = TextArtDataLoader(DATA_DIR, CONFIG, mode='val')
    train_align_collate = AlignCollate(CONFIG, 'train')
    val_align_collate = AlignCollate(CONFIG, 'val')
    train_batch_sampler = ImageBatchSampler(DATA_DIR, CONFIG, mode='train')
    val_batch_sampler = ImageBatchSampler(DATA_DIR, CONFIG, mode='val')
    train_loader = DataLoader(train_dataset,
                            batch_size=BATCH_SIZE,
                            shuffle=False,
                            num_workers=N_WORKERS,
                            pin_memory=True,
                            collate_fn=train_align_collate,
                            sampler=train_batch_sampler,
                            )
    val_loader = DataLoader(val_dataset,
                            batch_size=BATCH_SIZE,
                            shuffle=False,
                            num_workers=N_WORKERS,
                            pin_memory=True,
                            collate_fn=val_align_collate,
                            sampler=val_batch_sampler
                            )

    ## Init model with G and D
    model = GANModel(config, DEVICE, mode='train')

    model.G.train()
    model.D.train()

    for epoch in range(N_EPOCHS):
        epoch_start = time.time()
        total_g_loss = 0.0
        total_d_loss = 0.0
        
        for phase in ['train', 'val']:
            if phase == 'train':
                data_loader = train_loader
            else:
                data_loader = val_loader

            for i, data in enumerate(data_loader):

                batch_size = images.size()[0]

                real_images, real_wv_images, fake_wv_images = data

                ## Fşt batch
                model.fit(data)

                # Update total loss
                loss_g, loss_d = model.get_loss()
                total_g_loss += loss_g.item()
                total_d_loss += loss_d.item()

                # Print logs
                if i % 20 == 0:
                    print('[{0:3d}/{1}] {2:3d}/{3} loss_g: {4:.4f} | loss_d: {5:4f}'
                        .format(epoch + 1, N_EPOCHS, i + 1, len(data_loader), loss_g.item(), loss_d.item()))

            print("Epoch time: {}".format(time.time() - epoch_start))

            break

        #     # Save your model weights
        #     if (epoch + 1) % 5 == 0:
        #         save_dict = {
        #             'g':G.state_dict(), 
        #             'g_optim':optimizer_g.state_dict(),
        #             'd': D.state_dict(),
        #             'd_optim': optimizer_d.state_dict()
        #         }
        #         torch.save(save_dict, os.path.join(MODEL_PATH, 'checkpoint_{}.pth'.format(epoch + 1)))
                
        #     # Merge noisy input, ground truth and network output so that you can compare your results side by side
        #     out = torch.cat([img, fake], dim=2).detach().cpu().clamp(0.0, 1.0)
        #     vutils.save_image(out, os.path.join(OUTPUT_PATH, "{}_{}.png".format(epoch, i)), normalize=True)
            
        #     # Calculate avarage loss for the current epoch
        #     avg_g_loss = total_g_loss / len(data_loader)
        #     avg_d_loss = total_d_loss / len(data_loader)
        #     print('Epoch[{}] Training Loss G: {:4f} | D: {:4f}'.format(epoch + 1, avg_g_loss, avg_d_loss))
            
        #     cache_train_g.append(avg_g_loss)
        #     cache_train_d.append(avg_d_loss)
