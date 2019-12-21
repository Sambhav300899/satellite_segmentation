from shapely import wkt, affinity
from labels import labels

import cv2
import numpy as np
import pandas as pd
import tifffile as tiff
import matplotlib.pyplot as plt

def get_scalers(im_size):
    h, w = im_size
    w_ = w * (w / (w + 1))
    h_ = h * (h / (h + 1))
    return (w_, h_)

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

if __name__ == "__main__":
    img_path = 'dataset/three_band/6030_3_4'
    lbl = 5

    df_grid = pd.read_csv('dataset/grid_sizes.csv')
    df_poly = pd.read_csv('dataset/train_wkt_v4.csv')
    img = tiff.imread(img_path + '.tif').transpose([1, 2, 0])

    x_max = df_grid[df_grid['ImageId'] == img_path.split('/')[-1]]['Xmax']
    y_min = df_grid[df_grid['ImageId'] == img_path.split('/')[-1]]['Ymin']
    polygons = []

    req_df = df_poly[df_poly['ImageId'] == img_path.split('/')[-1]]
    req_df = req_df[req_df['ClassType'] == lbl]


    for poly in list(req_df['MultipolygonWKT']):
        polygons.append(wkt.loads(poly))

    x_scaler, y_scaler = get_scalers(img.shape[:2])
    x_scaler = x_scaler / x_max
    y_scaler = y_scaler / y_min

    scaled_polys = []

    for poly in polygons:
        scaled_polys.append(affinity.scale(poly, xfact=x_scaler, yfact=y_scaler, origin=(0, 0, 0)))

    masks = []

    for scaled_poly in scaled_polys:
        masks.append(create_mask(scaled_poly, img.shape[:2]))

    cv2.imwrite("{}.png".format(labels[lbl]), masks[0] * 255.0)
