import os
import cv2
import random
import numpy as np
import pandas as pd
import tifffile as tiff
from labels import labels
from utils import get_mask, get_rgb_img

def generator(lbl, img_path = 'dataset/sixteen_band', scaler_path = 'dataset/grid_sizes.csv', polygon_path = 'dataset/train_wkt_v4.csv', bs = 16, input_shape = (512, 512, 3),):
    df_grid = pd.read_csv(scaler_path)
    df_poly = pd.read_csv(polygon_path)
    im_paths = [img_path + os.path.sep + path.split('.')[0] for path in list(df_poly['ImageId'])]

    while True:
        X = []
        Y = []
        train_paths = random.sample(im_paths, bs)

        for img_path in train_paths:
            img = tiff.imread(img_path + '_M.tif').transpose(1, 2, 0)
            mask = get_mask(df_grid, df_poly, img_path, img.shape, lbl)

            img_rgb = get_rgb_img(img)

            img_rgb = cv2.resize(img_rgb, input_shape[:2])
            mask = cv2.resize(mask, input_shape[:2])

            X.append(img_rgb.astype('float') / 255.0)
            Y.append(mask)

        Y = np.expand_dims(Y, axis = 3)
        yield [np.array(X), np.array(Y)]
