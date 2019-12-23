import json
import argparse
import cv2
import numpy as np
from model import network

if __name__ == '__main__':
    ap = argparse.ArgumentParser()

    ap.add_argument('--config', required = True, help = 'path to config file')
    ap.add_argument('--img', required = True, help = 'path to input image', default = None)
    ap.add_argument('--model', required = True, help = 'path to model', default = None)

    args = vars(ap.parse_args())

    f = open(args['config'], 'r')
    json_data = json.load(f)
    f.close()

    model = network(model_path = args['model'], input_shape = json_data['train']['input_shape'])

    model.predict(args['img'], json_data['data']['scaler_data'], json_data['data']['polygon_data'], json_data['data']['label'])
