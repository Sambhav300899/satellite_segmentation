import json
import argparse
import cv2
import numpy as np
import tifffile as tiff
from model import network

def stretch_8bit(bands, lower_percent = 2, higher_percent = 98):
    out = np.zeros_like(bands)

    for i in range(3):
        a = 0
        b = 255
        c = np.percentile(bands[:,:,i], lower_percent)
        d = np.percentile(bands[:,:,i], higher_percent)
        t = a + (bands[:,:,i] - c) * (b - a) / (d - c)
        t[t<a] = a
        t[t>b] = b
        out[:,:,i] =t

    return out.astype(np.uint8)

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

    img = tiff.imread(args['img']).transpose([1, 2, 0])
    img_rgb = np.zeros((img.shape[0], img.shape[1], 3))

    img_rgb[:,:,2] = img[:,:,4] #red
    img_rgb[:,:,1] = img[:,:,2] #green
    img_rgb[:,:,0] = img[:,:,1] #blue
    img_rgb = stretch_8bit(img_rgb)

    model.predict(img_rgb)
