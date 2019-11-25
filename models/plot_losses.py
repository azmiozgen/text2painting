import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot(df, output_file, mode='train', figsize=(15, 10), G_yticks=2, D_yticks=0.1):
    assert mode in ['train', 'val'], "{} not available. Choose one of ['train', 'val']".format(mode)

    length = len(df.index)

    plt.figure(figsize=figsize)
    plt.suptitle("{} losses".format(mode.title()))

    ## Plot G
    net_name = 'Generator'
    col_name = 'G_loss'
    plt.subplot(1, 2, 1)
    plt.title("{} loss".format(net_name))
    plt.xlabel("Iterations")
    plt.ylabel("Loss")

    plt.grid()
    plt.yticks(np.arange(min(df[col_name]), max(df[col_name]), G_yticks))
    plt.plot(df['Iteration'], df[col_name])

    ## Plot D
    net_name = 'Discriminator'
    col_name = 'D_loss'
    plt.subplot(1, 2, 2)
    plt.title("{} loss".format(net_name))
    plt.xlabel("Iterations")
    plt.ylabel("Loss")

    plt.grid()
    plt.yticks(np.arange(min(df[col_name]), max(df[col_name]), D_yticks))
    plt.plot(df['Iteration'], df[col_name])

    plt.savefig(output_file)

if __name__ == '__main__':

    if len(sys.argv) != 2:
        print("Usage: python {} <model_dir>".format(sys.argv[0]))
        exit()

    model_dir = sys.argv[1]

    model_name = model_dir.split('_')[0]
    train_csv_file = os.path.join(model_dir, model_name + '_train_log.csv')
    val_csv_file = os.path.join(model_dir, model_name + '_val_log.csv')
    train_output_file = os.path.join(model_dir, model_name + '_train_loss_plot.png')
    val_output_file = os.path.join(model_dir, model_name + '_val_loss_plot.png')

    df_train = pd.read_csv(train_csv_file, delimiter=',')
    df_val = pd.read_csv(val_csv_file, delimiter=',')

    plot(df_train, train_output_file)
    plot(df_val, val_output_file)