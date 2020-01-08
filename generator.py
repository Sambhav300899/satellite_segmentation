import random
random.seed(42)

import os
import cv2
import numpy as np
import pandas as pd
import tifffile as tiff
from labels import labels
from utils import get_mask, stretch_nbit, create_crops

def generator(lbl, img_path = 'dataset/three_band', scaler_path = 'dataset/grid_sizes.csv', polygon_path = 'dataset/train_wkt_v4.csv', bs = 16, input_shape = (512, 512, 3), train_val_split = 0.25, subset = 'train'):
    df_grid = pd.read_csv(scaler_path)
    df_poly = pd.read_csv(polygon_path)
    im_paths = [img_path + os.path.sep + path.split('.')[0] for path in list(df_poly['ImageId'])]
    random.shuffle(im_paths)
    split = len(im_paths) * train_val_split

    if subset == 'train':
        im_paths = im_paths[:int(split)]
    elif subset == 'val':
        im_paths = im_paths[int(split):]

    while True:
        crops_rgb = []
        crops_mask = []
        random.shuffle(im_paths)

        for im_path in im_paths:
            img = tiff.imread(im_path + '.tif').transpose(1, 2, 0)
            mask = get_mask(df_grid, df_poly, im_path, img.shape, lbl)
            img_rgb = stretch_nbit(img)

            img_rgb = cv2.resize(img_rgb, (3300, 3300))
            mask = cv2.resize(mask, (3300, 3300))

            crops_rgb, _ = create_crops(img_rgb, input_shape)
            crops_mask, _ = create_crops(mask, input_shape)

            while len(crops_rgb) > bs:
                X = []
                Y = []
                for k in range(0, bs):
                    X.append(crops_rgb.pop().astype('float') / 255.0)
                    Y.append(crops_mask.pop())

                Y = np.expand_dims(Y, axis = 3)
                yield [np.array(X), np.array(Y)]
