## Get image shapes
import glob
import os
import re
import sys

# import numpy as np
import pandas as pd
from PIL import Image

HEADER = 'mean_width,mean_height,std_mean,std_height\n'

if __name__ == '__main__':

    if len(sys.argv) != 2:
       print("Usage python {} <SUBSET>".format(sys.argv[0]))
       exit()
    SUBSET = sys.argv[1]
    IMAGE_SHAPES_FILE = os.path.join(SUBSET, 'image_shapes.csv')
    OUTPUT_FILE = os.path.join(SUBSET, 'image_stats.csv')

    if not os.path.isfile(IMAGE_SHAPES_FILE):
       print(IMAGE_SHAPES_FILE, 'not found. Exiting.')
       exit()

    if os.path.isfile(OUTPUT_FILE):
       print(OUTPUT_FILE, 'exists. Exiting.')
       exit()

    df = pd.read_csv(IMAGE_SHAPES_FILE, delimiter=',')
    width_mean = df['width'].mean().round(2)
    height_mean = df['height'].mean().round(2)
    width_std = df['width'].std().round(2)
    height_std = df['height'].std().round(2)
    
    print("Subset", SUBSET)
    print("\tMean width:", width_mean)
    print("\tMean height:", height_mean)
    print("\tStd dev. width:", width_std)
    print("\tStd dev. height:", height_std)
    
    with open(OUTPUT_FILE, 'w') as f:
        f.write(HEADER)
        row_str = str(width_mean) + ',' + str(height_mean) + ',' + str(width_std) + ',' + str(height_std) + '\n'
        f.write(row_str)

    print(OUTPUT_FILE, "was written.")
