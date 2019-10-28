import argparse
import json
import os
import uuid

import cv2
import numpy as np

from pred_utils import get_model, identify_script
from settings import  PredSettings

if __name__ == "__main__":

    '''
        Read image files from a file
    '''

    ## Expected list file example
        ## book5/page-101/line_crop-0.png
        ## book5/page-101/line_crop-1.png
        ## ...

    parser = argparse.ArgumentParser()
    parser.add_argument('--lst', '-l', help="File that contains image filepaths", required=True, type=str)
    parser.add_argument('--usegpu', action='store_true', help="Enable cuda")
    args = parser.parse_args()

    input_image_filepaths_file = args.lst

    ps = PredSettings()

    # Get input images
    with open(input_image_filepaths_file, 'r') as f:
        input_image_filepaths = [l.strip() for l in f.readlines()]

    # Get Model
    model = get_model(args.usegpu)

    # Predict
    for image_file in input_image_filepaths:

        ## Get image and read it
        if not os.path.isfile(image_file):
            print("No image file {}".format(image_file))
            continue
        line_img = cv2.imread(image_file)
        line_height, line_width = line_img.shape[:2]

        ## Get corresponding json file
        json_file = ".".join(image_file.split(".")[:-1]) + ".json"
        if not os.path.isfile(json_file):
            print("No json file {} ".format(json_file))
            continue
        with open(json_file, 'r') as f:
            json_output = json.load(f)
        word_outputs = json_output['Line']['words']

        ## Get word coordinates and crop it and predict it
        for word_i, word_output in enumerate(word_outputs):
            word_xmin_ratio, word_xmax_ratio = word_output["coordinates"]

            ## Relative line coordinates
            word_xmin_inline = int(np.ceil(line_width * float(word_xmin_ratio)))
            word_xmax_inline = int(np.floor(line_width * float(word_xmax_ratio)))

            try:
                if word_xmin_inline < word_xmax_inline:
                    crop = line_img[0:line_height, word_xmin_inline:word_xmax_inline]
                    crop_path = "/tmp/{}.png".format(str(uuid.uuid4()))
                    cv2.imwrite(crop_path, crop)
                    script_output = identify_script(crop_path, model)
                    os.remove(crop_path)

                    ## Consider threshold
                    pred = script_output['Script']['prediction']
                    prob = script_output['Script']['probability']
                    if pred == 'arabic':
                        json_output['Line']['words'][word_i]['lang'] = script_output['Script']
                    elif pred == 'latin' and float(prob) > ps.LATIN_CONF_THRESHOLD:
                        json_output['Line']['words'][word_i]['lang'] = script_output['Script']
                    else:
                        json_output['Line']['words'][word_i]['lang'] = {'prediction' : 'arabic', 'probability' : '0.5'}
                else:
                    json_output['Line']['words'][word_i]['lang'] = {'prediction' : 'arabic', 'probability' : '0.5'}
            except Exception as e:
                json_output['Line']['words'][word_i]['lang'] = {'prediction' : 'arabic', 'probability' : '0.5'}

        # Save
        with open(json_file, 'w') as f:
            json.dump(json_output, f)
