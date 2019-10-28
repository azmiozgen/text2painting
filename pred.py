import argparse
import json
import os

from pred_utils import get_model, identify_script
from settings import  PredSettings

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', '-d', help="Process directory path", required=True, type=str)
    parser.add_argument('--input', '-i', help="Input image name", type=str, default="Image.png")
    parser.add_argument('--usegpu', action='store_true', help="Enable cuda")
    args = parser.parse_args()

    process_dir_path = args.dir
    input_image_name = args.input
    input_image_file = os.path.join(process_dir_path, input_image_name)

    ps = PredSettings()

    if not os.path.isfile(input_image_file):
        print('Cannot find input image : {}'.format(input_image_file))
        exit()

    # Get Model
    model = get_model(args.usegpu)

    # Predict
    try:
        script_output = identify_script(input_image_file, model)

        ## Consider threshold
        pred = script_output['Script']['prediction']
        prob = script_output['Script']['probability']
        if pred == 'latin' and float(prob) <= ps.LATIN_CONF_THRESHOLD:
            script_output = {'Script' : {'prediction' : 'arabic', 'probability' : '0.5'}}
    except Exception as e:
        script_output = {'Script' : {'prediction' : 'arabic', 'probability' : '0.5'}}

    # Save
    json.dump(script_output, open(os.path.join(process_dir_path, "output_{}.json".format(script_output["Script"]["prediction"])), 'w'))
