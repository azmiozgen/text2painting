import argparse
import glob
import os
import sys
import time
from multiprocessing import cpu_count

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageFilter

from gensim.models import Word2Vec
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from lib.config import Config
from lib.dataset import AlignCollate, ImageBatchSampler, TextArtDataLoader
from lib.model import GANModel


CONFIG = Config()



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='Model file to load')
    args = parser.parse_args()
    model_file = args.model

    ## Data loaders
    print("Data loader initializing..")
    val_dataset = TextArtDataLoader(CONFIG, kind='val')
    val_align_collate = AlignCollate(CONFIG, mode='test')
    # val_batch_sampler = ImageBatchSampler(CONFIG, kind='val')
    val_loader = DataLoader(val_dataset,
                            batch_size=CONFIG.BATCH_SIZE,
                            shuffle=False,
                            num_workers=CONFIG.N_WORKERS,
                            pin_memory=True,
                            collate_fn=val_align_collate,
                            # sampler=val_batch_sampler,
                            drop_last=True,
                            )
    print("\tValidation size:", len(val_dataset))
    n_val_batch = len(val_dataset) // CONFIG.BATCH_SIZE
    time.sleep(0.5)

    ## Init model
    print("\nModel initializing..")
    model = GANModel(CONFIG, model_file=model_file, mode='test', reset_lr=False)
    time.sleep(1.0)

    print("\nTesting starting..")
    start = time.time()

    total_loss_g = 0.0
    total_loss_d = 0.0
    total_loss_g_refiner = 0.0
    total_loss_d_decider = 0.0
    total_loss_g_refiner2 = 0.0
    total_loss_d_decider2 = 0.0
    total_loss_gp_fr = 0.0
    total_loss_gp_rf = 0.0
    total_loss_gp_decider_fr = 0.0
    total_loss_gp_decider2_fr = 0.0
    total_acc_rr = 0.0
    total_acc_rf = 0.0
    total_acc_fr = 0.0
    total_acc_decider_rr = 0.0
    total_acc_decider_fr = 0.0
    total_acc_decider2_rr = 0.0
    total_acc_decider2_fr = 0.0

    data_loader = val_loader
    n_batch = n_val_batch
    model.G.eval()
    model.D.eval()
    model.G_refiner.eval()
    model.D_decider.eval()
    model.G_refiner2.eval()
    model.D_decider2.eval()
    train_D = False
    train_G = False

    for i, data in enumerate(data_loader):
        iteration = i

        ## Get data
        real_first_images, real_second_images, real_images, real_wvs, fake_wvs = data
        batch_size = real_images.size()[0]

        ## Forward G
        real_wvs_flat = real_wvs.view(batch_size, -1)
        fake_images = model.forward(model.G, real_wvs_flat)

        ## Forward G_refiner
        refined1 = model.forward(model.G_refiner, fake_images)
        
        ## Forward G_refiner2
        refined2 = model.forward(model.G_refiner2, refined1)

        ## Update total loss
        loss_g, loss_d, loss_g_refiner, loss_d_decider, loss_g_refiner2, loss_d_decider2,\
            loss_gp_fr, loss_gp_rf, loss_gp_decider_fr, loss_gp_decider2_fr = model.get_losses()
        total_loss_g += loss_g
        total_loss_d += loss_d
        total_loss_g_refiner += loss_g_refiner
        total_loss_d_decider += loss_d_decider
        total_loss_g_refiner2 += loss_g_refiner2
        total_loss_d_decider2 += loss_d_decider2
        if loss_gp_fr:
            total_loss_gp_fr += loss_gp_fr
        if loss_gp_rf:
            total_loss_gp_rf += loss_gp_rf
        if loss_gp_decider_fr:
            total_loss_gp_decider_fr += loss_gp_decider_fr
        if loss_gp_decider2_fr:
            total_loss_gp_decider2_fr += loss_gp_decider2_fr

        ## Get D accuracy
        acc_rr, acc_rf, acc_fr, acc_decider_rr, acc_decider_fr, acc_decider2_rr, acc_decider2_fr = model.get_D_accuracy()
        total_acc_rr += acc_rr
        total_acc_rf += acc_rf
        total_acc_fr += acc_fr
        total_acc_decider_rr += acc_decider_rr
        total_acc_decider_fr += acc_decider_fr
        total_acc_decider2_rr += acc_decider2_rr
        total_acc_decider2_fr += acc_decider2_fr

        # Print logs
        if i % CONFIG.N_PRINT_BATCH == 0:
            print("\t\tBatch {: 4}/{: 4}:".format(i, n_batch), end=' ')
            if CONFIG.GAN_LOSS1 == 'wgangp':
                print("G loss: {:.4f} | D loss: {:.4f}".format(loss_g, loss_d), end=' ')
                print("| G refiner loss: {:.4f} | D decider loss {:.4f}".format(loss_g_refiner, loss_d_decider), end=' ')
                print("| G refiner2 loss: {:.4f} | D decider2 loss {:.4f}".format(loss_g_refiner2, loss_d_decider2), end=' ')
                print("| GP loss fake-real: {:.4f}".format(loss_gp_fr), end=' ')
                print("| GP loss real-fake: {:.4f}".format(loss_gp_rf), end=' ')
                print("| GP loss fake refined1-fake: {:.4f}".format(loss_gp_decider_fr), end=' ')
                print("| GP loss fake refined2-fake: {:.4f}".format(loss_gp_decider2_fr))
            else:
                print("G loss: {:.4f} | D loss: {:.4f}".format(loss_g, loss_d), end=' ')
                print("| G refiner loss: {:.4f} | D decider loss {:.4f}".format(loss_g_refiner, loss_d_decider), end=' ')
                print("| G refiner2 loss: {:.4f} | D decider2 loss {:.4f}".format(loss_g_refiner2, loss_d_decider2))
            print("\t\t\tAccuracy D real-real: {:.4f} | real-fake: {:.4f} | fake-real {:.4f}".format(acc_rr, acc_rf, acc_fr))
            print("\t\t\tAccuracy D decider real-real: {:.4f} | fake refined1-real {:.4f}".format(acc_decider_rr, acc_decider_fr))
            print("\t\t\tAccuracy D decider2 real-real: {:.4f} | fake refined2-real {:.4f}".format(acc_decider2_rr, acc_decider2_fr))

        ## Save visual outputs
        try:
            output_filename = "{}_{:08}.png".format(model.output_dir, iteration)
            grid_img_pil = model.generate_grid(real_wvs, fake_images, refined1, refined2, real_images, val_dataset.word2vec_model)
            model.save_img_output(grid_img_pil, output_filename)
        except Exception as e:
            print('Grid image generation failed.', e, 'Passing.')

    total_loss_g /= (i + 1)
    total_loss_d /= (i + 1)
    total_loss_g_refiner /= (i + 1)
    total_loss_d_decider /= (i + 1)
    total_loss_g_refiner2 /= (i + 1)
    total_loss_d_decider2 /= (i + 1)
    total_loss_gp_fr /= (i + 1)
    total_loss_gp_rf /= (i + 1)
    total_loss_gp_decider_fr /= (i + 1)
    total_loss_gp_decider2_fr /= (i + 1)
    total_acc_rr /= (i + 1)
    total_acc_rf /= (i + 1)
    total_acc_fr /= (i + 1)
    total_acc_decider_rr /= (i + 1)
    total_acc_decider_fr /= (i + 1)
    total_acc_decider2_rr /= (i + 1)
    total_acc_decider2_fr /= (i + 1)
    if CONFIG.GAN_LOSS1 == 'wgangp':
        print("\t\tG loss: {:.4f} | D loss: {:.4f}".format(total_loss_g, total_loss_d), end=' ')
        print("| {p} G refiner loss: {:.4f} | {p} D decider loss: {:.4f}".format(total_loss_g_refiner, total_loss_d_decider), end=' ')
        print("| {p} G refiner2 loss: {:.4f} | {p} D decider2 loss: {:.4f}".format(total_loss_g_refiner2, total_loss_d_decider2), end=' ')
        print("| GP loss fake-real: {:.4f}".format(total_loss_gp_fr), end=' ')
        print("| GP loss real-fake: {:.4f}".format(total_loss_gp_rf), end=' ')
        print("| GP loss real refined1-fake: {:.4f}".format(total_loss_gp_decider_fr), end=' ')
        print("| GP loss real refined2-fake: {:.4f}".format(total_loss_gp_decider2_fr))
    else:
        print("\t\tG loss: {:.4f} | {p} D loss: {:.4f}".format(total_loss_g, total_loss_d), end=' ')
        print("\t\t{p} G refiner loss: {:.4f} | {p} D decider loss: {:.4f}".format(total_loss_g_refiner, total_loss_d_decider))
        print("\t\t{p} G refiner2 loss: {:.4f} | {p} D decider2 loss: {:.4f}".format(total_loss_g_refiner2, total_loss_d_decider2))
    print("\t\tAccuracy D real-real: {:.4f} | real-fake: {:.4f} | fake-real {:.4f}".format(total_acc_rr, total_acc_rf, total_acc_fr))
    print("\t\tAccuracy D decider real-real: {:.4f} | fake refined1-real {:.4f}".format(total_acc_decider_rr, total_acc_decider_fr))
    print("\t\tAccuracy D decider2 real-real: {:.4f} | fake refined2-real {:.4f}".format(total_acc_decider2_rr, total_acc_decider2_fr))
    print("\tTesting time: {:.2f} seconds".format(time.time() - start))
