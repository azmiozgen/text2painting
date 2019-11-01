## Get image shapes
import glob
import os
import re
import sys

from PIL import Image

HEADER = 'filename,width,height\n'

if __name__ == '__main__':

    if len(sys.argv) != 2:
       print("Usage python {} <SUBSET>".format(sys.argv[0]))
       exit()
    SUBSET = sys.argv[1]
    OUTPUT_FILE = os.path.join(SUBSET, 'image_shapes.csv')

    image_files = glob.glob(os.path.join(SUBSET, 'images', '*'))
    n_images = len(image_files)
    print(n_images, "image files")

    if os.path.isfile(OUTPUT_FILE):
       print(OUTPUT_FILE, 'exists. Exiting.')
       exit()

    line_counter = 0
    with open(OUTPUT_FILE, 'w') as f:
        f.write(HEADER)
        for image_file in image_files:
            image_filename = os.path.basename(image_file)

            img_pil = Image.open(image_file)
            width, height = img_pil.size

            row_str = image_filename + ',' + str(width) + ',' + str(height) + '\n'
            f.write(row_str)
            line_counter += 1


    print(OUTPUT_FILE, "was written with {} lines.".format(line_counter))
