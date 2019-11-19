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

CONFIG = Config()

if __name__ == "__main__":

    ## Set GPU
    # os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    # os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    # print("DEVICE:", torch.cuda.current_device())

    ## Data loaders
    print("\nData loaders initializing..")
    train_dataset = TextArtDataLoader(CONFIG, mode='train')
    val_dataset = TextArtDataLoader(CONFIG, mode='val')
    train_align_collate = AlignCollate(CONFIG, 'train')
    val_align_collate = AlignCollate(CONFIG, 'val')
    train_batch_sampler = ImageBatchSampler(CONFIG, mode='train')
    val_batch_sampler = ImageBatchSampler(CONFIG, mode='val')
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
    model.init_model_dir()
    print("Created", model.model_dir)
    time.sleep(1.0)

    print("\nTraining starting..")
    for epoch in range(CONFIG.N_EPOCHS):
        print("Epoch {}/{}:".format(epoch + 1, CONFIG.N_EPOCHS))
        total_loss_g = 0.0
        total_loss_d = 0.0

        for phase in ['train', 'val']:
            phase_start = time.time()
            print("\t{} phase:".format(phase.title()))
            if phase == 'train':
                # data_loader = val_loader  ## TODO
                # n_batch = n_val_batch
                data_loader = train_loader
                n_batch = n_train_batch
                model.G.train()
                model.D.train()
            else:
                data_loader = val_loader
                n_batch = n_val_batch
                model.G.eval()
                model.D.eval()

            for i, data in enumerate(data_loader):
                iteration = epoch * n_batch + i

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
                # loss_g, loss_d = -1.0, -1.0
                total_loss_g += loss_g
                total_loss_d += loss_d

                ## Save logs
                if i % CONFIG.N_LOG_BATCH == 0:
                    model.save_logs(phase, epoch, iteration, loss_g, loss_d)

                # Print logs
                if i % CONFIG.N_PRINT_BATCH == 0:
                    print("\t\tBatch {: 4}/{: 4}:".format(i, n_batch), end=' ')
                    print("G loss: {:.4f} | D loss: {:.4f}".format(loss_g, loss_d))

                ## Save visual outputs
                try:
                    if i % CONFIG.N_SAVE_VISUALS_BATCH == 0 and phase == 'val':
                        output_filename = "{}_{:04}_{:08}.png".format(model.model_name, epoch, iteration)
                        grid_img_pil = model.generate_grid(real_wv_tensor, real_images, train_dataset.word2vec_model)
                        model.save_output(grid_img_pil, output_filename)
                except Exception as e:
                    print(e, 'Passing.')

            total_loss_g /= n_batch
            total_loss_d /= n_batch
            print("\t\t{p} G loss: {:.4f} | {p} D loss: {:.4f}".format(total_loss_g, total_loss_d, p=phase.title()))
            print("\t{} time: {:.2f} seconds".format(phase.title(), time.time() - phase_start))

        ## Save model
        if (epoch + 1) % CONFIG.N_SAVE_MODEL_EPOCHS == 0:
            model.save_model_dict(epoch + 1, iteration, total_loss_g, total_loss_d)
            
