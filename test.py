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

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='Model file to load')
    args = parser.parse_args()
    model_file = args.model

    ## Import model from relevant lib 
    model_dir = os.path.dirname(os.path.abspath(model_file))
    model_lib_dir = os.path.join(model_dir, 'lib')
    sys.path.append(model_lib_dir)
    from model import GANModel
    from lib.config import Config
    from lib.dataset import AlignCollate, ImageBatchSampler, TextArtDataLoader
    from lib.utils import generate_noise

    CONFIG = Config()

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

    print("\nTesting..")
    start = time.time()

    data_loader = val_loader
    n_batch = n_val_batch
    model.G.eval()
    model.G_refiner.eval()
    model.G_refiner2.eval()
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


        for j, (fake, _refined1, _refined2, real_image) in enumerate(zip(fake_images, refined1, refined2, real_images)):
            _id = i * batch_size + j
            filename = "{:08}.png".format(_id)

            ## Noise output
            noise_output = generate_noise(CONFIG)

            ## Save fake
            try:
                model.save_img_test_single(fake.clone(), filename, kind='fake')
            except Exception as e:
                print('Fake image {} saving failed.'.format(filename), e, 'Passing.')

            ## Save refined
            try:
                model.save_img_test_single(_refined1.clone(), filename, kind='refined')
            except Exception as e:
                print('Refined image {} saving failed.'.format(filename), e, 'Passing.')

            ## Save refined2
            try:
                model.save_img_test_single(_refined2.clone(), filename, kind='refined2')
            except Exception as e:
                print('Refined2 image {} saving failed.'.format(filename), e, 'Passing.')

            ## Save real
            try:
                model.save_img_test_single(real_image.clone(), filename, kind='real')
            except Exception as e:
                print('Real image {} saving failed.'.format(filename), e, 'Passing.')

            ## Save noise
            try:
                model.save_img_test_single(noise_output.clone(), filename, kind='noise')
            except Exception as e:
                print('Noise image {} saving failed.'.format(filename), e, 'Passing.')

        ## Save grid
        try:
            grid_filename = "{}_{:08}.png".format(model.model_name, iteration)
            grid_img_pil = model.generate_grid(real_wvs, fake_images, refined1, refined2, real_images, val_dataset.word2vec_model)
            model.save_img_test_grid(grid_img_pil, grid_filename)
        except Exception as e:
            print('Grid image generation failed.', e, 'Passing.')

    print("\tTesting time: {:.2f} seconds".format(time.time() - start))
