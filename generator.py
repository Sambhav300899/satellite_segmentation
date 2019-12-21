import os
import cv2
import random
import numpy as np
import pandas as pd
import tifffile as tiff
from labels import labels
from shapely import wkt, affinity

def get_scalers(im_size):
    h, w = im_size
    w_ = w * (w / (w + 1))
    h_ = h * (h / (h + 1))
    return (w_, h_)

def stretch_8bit(bands, lower_percent = 2, higher_percent = 98):
    out = np.zeros_like(bands)

    for i in range(3):
        a = 0
        b = 255
        c = np.percentile(bands[:, :, i], lower_percent)
        d = np.percentile(bands[:, :, i], higher_percent)
        t = a + (bands[: , :, i] - c) * (b - a) / (d - c)
        t[t < a] = a
        t[t > b] = b
        out[:, :, i] =t

    return out.astype(np.uint8)

def create_mask(polys, im_size):
    img_mask = np.zeros(im_size, np.uint8)

    if not polys :
        return img_mask

    int_coords = lambda x : np.array(x).round().astype(np.int32)

    exteriors = [int_coords(poly.exterior.coords) for poly in polys]
    interiors = [int_coords(pi.coords) for poly in polys for pi in poly.interiors]

    cv2.fillPoly(img_mask, exteriors, 1)
    cv2.fillPoly(img_mask, interiors, 0)
    return img_mask

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
            x_max = df_grid[df_grid['ImageId'] == img_path.split('/')[-1]]['Xmax']
            y_min = df_grid[df_grid['ImageId'] == img_path.split('/')[-1]]['Ymin']
            req_df = df_poly[df_poly['ImageId'] == img_path.split('/')[-1]]
            req_df = req_df[req_df['ClassType'] == lbl]


            polygons = []
            for poly in list(req_df['MultipolygonWKT']):
                polygons.append(wkt.loads(poly))

            x_scaler, y_scaler = get_scalers(img.shape[:2])
            x_scaler = x_scaler / float(x_max)
            y_scaler = y_scaler / float(y_min)

            scaled_polys = []
            for poly in polygons:
                scaled_polys.append(affinity.scale(poly, xfact=x_scaler, yfact=y_scaler, origin=(0, 0, 0)))

            masks = []
            for scaled_poly in scaled_polys:
                masks.append(create_mask(scaled_poly, img.shape[:2]))

            mask = masks[0]
            img_rgb = np.zeros((img.shape[0], img.shape[1], 3))

            img_rgb[:, :, 0] = img[:, :, 1] #blue
            img_rgb[:, :, 1] = img[:, :, 2] #green
            img_rgb[:, :, 2] = img[:, :, 4] #red
            img_rgb = stretch_8bit(img_rgb)

            img_rgb = cv2.resize(img_rgb, input_shape[:2])
            mask = cv2.resize(mask, input_shape[:2])

            X.append(img_rgb.astype('float') / 255.0)
            Y.append(mask)

        Y = np.expand_dims(Y, axis = 3)
        yield [np.array(X), np.array(Y)]
