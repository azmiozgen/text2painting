## Get image shapes
import glob
import os
import re
import sys

import numpy as np
from PIL import Image, ImageStat

HEADER = "mean_width, mean_height, mean_r, mean_g, mean_b, std_r, std_g, std_b, n_mean_r, n_mean_g, n_mean_b, n_std_r, n_std_g, n_std_b\n"

if __name__ == '__main__':

    if len(sys.argv) != 4:
        print("Usage python {} <subset> <labels_filename> <output_name>".format(sys.argv[0]))
        exit()
    SUBSET = sys.argv[1]
    LABELS_FILENAME = sys.argv[2]
    OUTPUT_NAME = sys.argv[3]
    OUTPUT_FILE = os.path.join(SUBSET, '{}_image_stats.csv'.format(OUTPUT_NAME))
    labels_file = os.path.abspath(os.path.join(__file__, os.path.pardir, SUBSET, LABELS_FILENAME))

    with open(labels_file, 'r') as f:
        lines = f.readlines()

    line_counter = 0
    with open(OUTPUT_FILE, 'w') as f:
        f.write(HEADER)

        widths = []
        heights = []
        means = []
        stds = []
        for i, line in enumerate(lines):
            row = line.strip().split(',')
            image_relative_file = row[0]
            image_file = os.path.abspath(os.path.join(__file__, os.path.pardir, os.path.pardir, image_relative_file))
            
            if i % 1000 == 0:
                print("{}/{}".format(i, len(lines)))

            if not os.path.isfile(image_file):
                print(image_file, 'not found. Passing.')
                continue

            try:
                img = Image.open(image_file).convert('RGB')
            except OSError as e:
                print(e)
                continue

            img_size = img.size
            means.append(ImageStat.Stat(img).mean)
            stds.append(ImageStat.Stat(img).stddev)
            widths.append(img.size[0])
            heights.append(img.size[1])

        means_of_means = np.array(means).mean(axis=1)
        means_of_stds = np.array(stds).std(axis=1)
        f.write("{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f}\n".format(
                                                            np.mean(widths), np.mean(heights),
                                                            means_of_means[0], means_of_means[1], means_of_means[2],
                                                            means_of_stds[0], means_of_stds[1], means_of_stds[2],
                                                            means_of_means[0]/255., means_of_means[1]/255., means_of_means[2]/255.,
                                                            means_of_stds[0]/255., means_of_stds[1]/255., means_of_stds[2]/255.))

    print(OUTPUT_FILE, "was written.")
