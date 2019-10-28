import argparse
import os
import sys

import numpy as np
import torch

from lib import AlignCollate, ClsDataset, Model
from settings import ModelSettings, TrainingSettings

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, help='path of model')
    parser.add_argument('--lmdb', required=True, help='path of lmdb')
    parser.add_argument('--usegpu', action='store_true', help='enables cuda to train on gpu')
    parser.add_argument('--batchsize', type=int, default=1, help='input batch size')
    parser.add_argument('--nworkers', type=int, help='number of data loading workers [0 to do it using main process]', default=0)
    opt = parser.parse_args()

    model_path = opt.model

    ms = ModelSettings()
    ts = TrainingSettings()

    if torch.cuda.is_available() and not opt.usegpu:
        print('WARNING: You have a CUDA device, so you should probably run with --cuda')

    # Define Data Loaders
    pin_memory = False
    if opt.usegpu:
        pin_memory = True

    test_dataset = ClsDataset(opt.lmdb)
    test_align_collate = AlignCollate('test', ms.LABELS, ms.MEAN, ms.STD, ms.IMAGE_SIZE_HEIGHT, ms.IMAGE_SIZE_WIDTH,
                                      horizontal_flipping=ms.HORIZONTAL_FLIPPING,
                                      random_rotation=ms.RANDOM_ROTATION,
                                      color_jittering=ms.COLOR_JITTERING, random_grayscale=ms.RANDOM_GRAYSCALE,
                                      random_channel_swapping=ms.RANDOM_CHANNEL_SWAPPING, random_gamma=ms.RANDOM_GAMMA,
                                      random_resolution=ms.RANDOM_RESOLUTION)

    assert test_dataset
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=opt.batchsize, shuffle=False,
                                              num_workers=opt.nworkers, pin_memory=pin_memory, collate_fn=test_align_collate)

    # Define Model
    model = Model(ts.MODEL, ts.N_CLASSES, load_model_path=model_path, usegpu=opt.usegpu)

    # Test Model
    predictions, labels = model.test(test_loader)

    predictions = np.array(predictions)
    labels = np.array(labels)
    acc = np.mean(predictions.argmax(1) == labels)
    print("ACCURACY : ", acc)

    print("LABELS:")
    for i, l in enumerate(ms.LABELS):
        print("\t{} : {}".format(i, l))
    output_path = "./output_test"
    if not os.path.isdir(output_path):
        os.makedirs(output_path)
    for index in predictions:
        # print("\t{} -----> {}, {} (index:{})".format(labels[index], predictions[index].argmax(), predictions[index], index))
        gt = labels[index]
        pred = predictions[index].argmax()
        prob = predictions[index].max()
        test_dataset[index][0].save(os.path.join(output_path, "{}_{}_{}_{:.4f}.png".format(index, gt, pred, prob)))
    print("All predictions were written to", output_path)

    ## Confusions
    print("LABELS:")
    for i, l in enumerate(ms.LABELS):
        print("\t{} : {}".format(i, l))
    if acc < 1.0:
        output_path = "./output"
        if not os.path.isdir(output_path):
            os.makedirs(output_path)
        confusion_indexes = np.where(predictions.argmax(1) != labels)[0]
        print("Confusions with probs:")
        for index in confusion_indexes:
            print("\t{} -----> {}, {} (index:{})".format(labels[index], predictions[index].argmax(), predictions[index], index))
            test_dataset[index][0].save(os.path.join(output_path, "{}.png".format(index)))
        print("Wrong predictions were written to", output_path)
