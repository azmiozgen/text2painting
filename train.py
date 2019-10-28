import argparse
import datetime
import getpass
import os
import random
import shutil

import numpy as np
import torch

from lib import AlignCollate, ClsDataset, Model
from settings import TrainingSettings

ts = TrainingSettings()

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='', help='path to model (to continue training)')
parser.add_argument('--usegpu', action='store_true', help='enables cuda to train on gpu')
parser.add_argument('--nepochs', type=int, default=200, help='number of epochs to train for')
parser.add_argument('--batchsize', type=int, default=128, help='input batch size')
parser.add_argument('--nworkers', type=int, help='number of data loading workers [0 to do it using main process]', default=1)
opt = parser.parse_args()

def generate_run_id():

    username = getpass.getuser()

    now = datetime.datetime.now()
    date = list(map(str, [now.year, now.month, now.day]))
    coarse_time = list(map(str, [now.hour, now.minute]))
    fine_time = list(map(str, [now.second, now.microsecond]))

    run_id = '_'.join(['-'.join(date), '-'.join(coarse_time), username, '-'.join(fine_time)])
    return run_id

RUN_ID = generate_run_id()
model_save_path = os.path.abspath(os.path.join(os.path.abspath(__file__), os.path.pardir, 'models', RUN_ID))
os.mkdir(model_save_path)

CODE_BASE_DIR = os.path.abspath(os.path.join(os.path.abspath(__file__), os.path.pardir))
shutil.copytree(os.path.join(CODE_BASE_DIR, 'settings'), os.path.join(model_save_path, 'settings'))
shutil.copytree(os.path.join(CODE_BASE_DIR, 'lib'), os.path.join(model_save_path, 'lib'))

init_fp = open(os.path.join(model_save_path, '__init__.py'), 'w')
init_fp.close()

if torch.cuda.is_available() and not opt.usegpu:
    print('WARNING: You have a CUDA device, so you should probably run with --cuda')

# Load Seeds
random.seed(ts.SEED)
np.random.seed(ts.SEED)
torch.manual_seed(ts.SEED)

# Define Data Loaders
pin_memory = False
if opt.usegpu:
    pin_memory = True

train_dataset = ClsDataset(ts.TRAINING_LMDB)
train_align_collate = AlignCollate('training', ts.LABELS, ts.MEAN, ts.STD, ts.IMAGE_SIZE_HEIGHT, ts.IMAGE_SIZE_WIDTH,
                                   horizontal_flipping=ts.HORIZONTAL_FLIPPING,
                                   random_rotation=ts.RANDOM_ROTATION,
                                   color_jittering=ts.COLOR_JITTERING, random_grayscale=ts.RANDOM_GRAYSCALE,
                                   random_channel_swapping=ts.RANDOM_CHANNEL_SWAPPING, random_gamma=ts.RANDOM_GAMMA,
                                   random_resolution=ts.RANDOM_RESOLUTION)

assert train_dataset
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.batchsize, shuffle=True,
                                           num_workers=opt.nworkers, pin_memory=pin_memory, collate_fn=train_align_collate)

test_dataset = ClsDataset(ts.VALIDATION_LMDB)
test_align_collate = AlignCollate('test', ts.LABELS, ts.MEAN, ts.STD, ts.IMAGE_SIZE_HEIGHT, ts.IMAGE_SIZE_WIDTH,
                                  horizontal_flipping=ts.HORIZONTAL_FLIPPING,
                                  random_rotation=ts.RANDOM_ROTATION,
                                  color_jittering=ts.COLOR_JITTERING, random_grayscale=ts.RANDOM_GRAYSCALE,
                                  random_channel_swapping=ts.RANDOM_CHANNEL_SWAPPING, random_gamma=ts.RANDOM_GAMMA,
                                  random_resolution=ts.RANDOM_RESOLUTION)

assert test_dataset
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=opt.batchsize, shuffle=False,
                                          num_workers=opt.nworkers, pin_memory=pin_memory, collate_fn=test_align_collate)

# Define Model
model = Model(ts.MODEL, ts.N_CLASSES, load_model_path=opt.model, usegpu=opt.usegpu)

# Train Model
model.fit(ts.LEARNING_RATE, ts.WEIGHT_DECAY, ts.CLIP_GRAD_NORM, ts.LR_DROP_FACTOR, ts.LR_DROP_PATIENCE, ts.OPTIMIZER,
          opt.nepochs, train_loader, test_loader, model_save_path)
