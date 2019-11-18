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

DATA_DIR = 'united'
CONFIG = Config()

if __name__ == "__main__":

    ## Set GPU
    # os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    # os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    # print("DEVICE:", torch.cuda.current_device())

    ## Data loaders
    print("\nData loaders initializing..")
    train_dataset = TextArtDataLoader(DATA_DIR, CONFIG, mode='train')
    val_dataset = TextArtDataLoader(DATA_DIR, CONFIG, mode='val')
    train_align_collate = AlignCollate(CONFIG, 'train')
    val_align_collate = AlignCollate(CONFIG, 'val')
    train_batch_sampler = ImageBatchSampler(DATA_DIR, CONFIG, mode='train')
    val_batch_sampler = ImageBatchSampler(DATA_DIR, CONFIG, mode='val')
    train_loader = DataLoader(train_dataset,
                              batch_size=CONFIG.BATCH_SIZE,
                              shuffle=False,
                              num_workers=CONFIG.N_WORKERS,
                              pin_memory=True,
                              collate_fn=train_align_collate,
                              sampler=train_batch_sampler,
                             )
    val_loader = DataLoader(val_dataset,
                            batch_size=CONFIG.BATCH_SIZE,
                            shuffle=False,
                            num_workers=CONFIG.N_WORKERS,
                            pin_memory=True,
                            collate_fn=val_align_collate,
                            sampler=val_batch_sampler
                            )
    print("\tTrain size:", len(train_dataset))
    print("\tValidation size:", len(val_dataset))
    n_train_batch = len(train_dataset) // CONFIG.BATCH_SIZE
    n_val_batch = len(val_dataset) // CONFIG.BATCH_SIZE
    time.sleep(0.5)

    ## Init model with G and D
    print("\nModel initializing..")
    model = GANModel(CONFIG, mode='train')
    time.sleep(0.5)

    print("\nTraining starting..")
    for epoch in range(CONFIG.N_EPOCHS):
        print("Epoch {}/{}:".format(epoch + 1, CONFIG.N_EPOCHS))
        epoch_start = time.time()
        total_loss_g = 0.0
        total_loss_d = 0.0

        for phase in ['train']:
            print("\t{} phase".format(phase.title()))
            if phase == 'train':
                data_loader = val_loader  ## TODO
                n_batch = n_val_batch
                model.G.train()
                model.D.train()
            else:
                data_loader = val_loader
                n_batch = n_val_batch
                model.G.eval()
                model.D.eval()

            for i, data in enumerate(data_loader):
                real_images, real_wv_tensor, fake_wv_tensor = data
                batch_size = real_images.size()[0]

                # print("IMAGE:", real_images.shape)
                # print("WV:", real_wv_tensor.shape)
                # print("Fake WV:", fake_wv_tensor.shape)

                ## Fit batch
                # print("\nModel optimizing..")
                model.fit(data, phase=phase)

                # Update total loss
                loss_g, loss_d = model.get_loss()
                total_loss_g += loss_g
                total_loss_d += loss_d

                # Print logs
                if i % CONFIG.N_PRINT_BATCH == 0:
                    print("\t\tBatch {: 4}/{: 4}:".format(i, n_batch), end=' ')
                    print("G loss: {:.4f} | D loss: {:.4f}".format(loss_g, loss_d))

        total_loss_g /= n_batch
        total_loss_d /= n_batch
        print("\tEpoch time: {:.2f} seconds".format(time.time() - epoch_start))
        print("\t\tTotal G loss: {:.4f} | Total D loss: {:.4f}".format(total_loss_g, total_loss_d))

        ## Save model
        if (epoch + 1) % CONFIG.N_SAVE_MODEL_EPOCHS == 0:
            model.save_model_dict(CONFIG.MODEL_NAME, epoch + 1, total_loss_g, total_loss_d)
            
        # # Merge noisy input, ground truth and network output so that you can compare your results side by side
        # out = torch.cat([img, fake], dim=2).detach().cpu().clamp(0.0, 1.0)
        # vutils.save_image(out, os.path.join(OUTPUT_PATH, "{}_{}.png".format(epoch, i)), normalize=True)
        
        # cache_train_g.append(avg_g_loss)
        # cache_train_d.append(avg_d_loss)
