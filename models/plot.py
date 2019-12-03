import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot(df, output_file, mode='train', figsize=(15, 10), G_yticks=2, D_yticks=0.1):
    assert mode in ['train', 'val'], "{} not available. Choose one of ['train', 'val']".format(mode)

    length = len(df.index)

    plt.figure(figsize=figsize)
    plt.suptitle("{} plots".format(mode.title()))
    plt.tight_layout()

    ## Plot G loss
    net_name = 'Generator'
    col_name = 'G_loss'
    plt.subplot(3, 3, 1)
    plt.title("{} loss".format(net_name))
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.grid()
    plt.yticks(np.arange(min(df[col_name]), max(df[col_name]) + 0.1, G_yticks))
    plt.plot(df['Iteration'], df[col_name])

    ## Plot D loss
    net_name = 'Discriminator'
    col_name = 'D_loss'
    plt.subplot(3, 3, 2)
    plt.title("{} loss".format(net_name))
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.grid()
    plt.yticks(np.arange(min(df[col_name]), max(df[col_name]) + 0.1, G_yticks))
    plt.plot(df['Iteration'], df[col_name])

    ## Plot G refiner loss
    net_name = 'Generator refiner'
    col_name = 'G_refiner_loss'
    plt.subplot(3, 3, 3)
    plt.title("{} loss".format(net_name))
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.grid()
    plt.yticks(np.arange(min(df[col_name]), max(df[col_name]) + 0.1, G_yticks))
    plt.plot(df['Iteration'], df[col_name])

    ## Plot D real-real accuracy
    net_name = 'Discriminator'
    col_name = 'D_rr_acc'
    plt.subplot(3, 3, 4)
    plt.title("{} real-real accuracy".format(net_name))
    plt.xlabel("Iterations")
    plt.ylabel("Accuracy")
    plt.grid()
    plt.yticks(np.arange(0.0, 1.1, 0.1))
    plt.plot(df['Iteration'], df[col_name])

    ## Plot D fake-real accuracy
    net_name = 'Discriminator'
    col_name = 'D_fr_acc'
    plt.subplot(3, 3, 5)
    plt.title("{} fake-real accuracy".format(net_name))
    plt.xlabel("Iterations")
    plt.ylabel("Accuracy")
    plt.grid()
    plt.yticks(np.arange(0.0, 1.1, 0.1))
    plt.plot(df['Iteration'], df[col_name])

    ## Plot D fake refined-real accuracy
    net_name = 'Discriminator'
    col_name = 'D_refined_fr_acc'
    plt.subplot(3, 3, 6)
    plt.title("{} fake refined-real accuracy".format(net_name))
    plt.xlabel("Iterations")
    plt.ylabel("Accuracy")
    plt.grid()
    plt.yticks(np.arange(0.0, 1.1, 0.1))
    plt.plot(df['Iteration'], df[col_name])

    ## Plot D real-fake accuracy
    net_name = 'Discriminator'
    col_name = 'D_rf_acc'
    plt.subplot(3, 3, 7)
    plt.title("{} real-fake accuracy".format(net_name))
    plt.xlabel("Iterations")
    plt.ylabel("Accuracy")
    plt.grid()
    plt.yticks(np.arange(0.0, 1.1, 0.1))
    plt.plot(df['Iteration'], df[col_name])

    plt.subplots_adjust(bottom=0.05, left=0.07, right=0.93, top=0.93, wspace = 0.25, hspace = 0.4)
    plt.savefig(output_file)

if __name__ == '__main__':

    if len(sys.argv) != 2:
        print("Usage: python {} <model_dir>".format(sys.argv[0]))
        exit()

    model_dir = sys.argv[1]

    model_name = model_dir.split('_')[0]

    ## Plot train
    train_csv_file = os.path.join(model_dir, model_name + '_train_log.csv')
    if os.path.isfile(train_csv_file):
        train_output_file = os.path.join(model_dir, model_name + '_train_plot.png')
        df_train = pd.read_csv(train_csv_file, delimiter=',')
        if len(df_train.index) > 0:
            plot(df_train, train_output_file, mode='train')
        else:
            print("Empty table for", train_csv_file)
    else:
        print(train_csv_file, "not found")

    ## Plot val
    val_csv_file = os.path.join(model_dir, model_name + '_val_log.csv')
    if os.path.isfile(val_csv_file):
        val_output_file = os.path.join(model_dir, model_name + '_val_plot.png')
        df_val = pd.read_csv(val_csv_file, delimiter=',')
        if len(df_val.index) > 0:
            plot(df_val, val_output_file, mode='val')
        else:
            print("Empty table for", val_csv_file)
    else:
        print(val_csv_file, "not found")
