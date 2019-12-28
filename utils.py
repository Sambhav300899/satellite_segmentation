import cv2
import numpy as np
from shapely import wkt, affinity

def stretch_nbit(bands, lower_percent = 5, higher_percent = 95):
    out = np.zeros_like(bands)

    for i in range(bands.shape[2]):
        a = 0
        b = 255.0
        c = np.percentile(bands[:, :, i], lower_percent)
        d = np.percentile(bands[:, :, i], higher_percent)
        t = a + (bands[: , :, i] - c) * (b - a) / (d - c)
        t[t < a] = a
        t[t > b] = b
        out[:, :, i] =t

    return out.astype(np.uint8)

def get_rgb_img(img):
    img_rgb = np.zeros((img.shape[0], img.shape[1], 3))

    img_rgb[:,:,2] = img[:,:,4] #red
    img_rgb[:,:,1] = img[:,:,2] #green
    img_rgb[:,:,0] = img[:,:,1] #blue
    img_rgb = stretch_nbit(img_rgb)

    return img_rgb

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

def get_scalers(im_size):
    h, w = im_size
    w_ = w * (w / (w + 1))
    h_ = h * (h / (h + 1))
    return (w_, h_)

def get_mask(df_grid, df_poly, img_path, img_shape, lbl):
    x_max = df_grid[df_grid['ImageId'] == img_path.split('/')[-1]]['Xmax']
    y_min = df_grid[df_grid['ImageId'] == img_path.split('/')[-1]]['Ymin']
    req_df = df_poly[df_poly['ImageId'] == img_path.split('/')[-1]]
    req_df = req_df[req_df['ClassType'] == lbl]

    polygons = []
    for poly in list(req_df['MultipolygonWKT']):
        polygons.append(wkt.loads(poly))

    x_scaler, y_scaler = get_scalers(img_shape[:2])
    x_scaler = x_scaler / float(x_max)
    y_scaler = y_scaler / float(y_min)

    scaled_polys = []
    for poly in polygons:
        scaled_polys.append(affinity.scale(poly, xfact = x_scaler, yfact = y_scaler, origin = (0, 0, 0)))

    masks = []
    for scaled_poly in scaled_polys:
        masks.append(create_mask(scaled_poly, img_shape[:2]))

    return masks[0]

def create_crops(img, input_shape):
    crops = []
    border_len = (0 - 3300) % input_shape[0] #https://stackoverflow.com/questions/10133194/reverse-modulus-operator/31735393#31735393
    img = cv2.copyMakeBorder(img, 0, border_len, 0, border_len, cv2.BORDER_REFLECT)

    for i in range(0, img.shape[1], input_shape[1]):
        for j in range(0, img.shape[0], input_shape[0]):
            crops.append(img[i : i + input_shape[1], j : j + input_shape[0]])

    return crops, img

def stitch(cropped_images, input_shape):
    border_len = (0 - 3300) % input_shape[0] #https://stackoverflow.com/questions/10133194/reverse-modulus-operator/31735393#31735393
    img_mask = np.zeros((3300 + border_len, 3300 + border_len, 1))
    k = 0
    for i in range(0, img_mask.shape[1], input_shape[1]):
        for j in range(0, img_mask.shape[0], input_shape[0]):
            img_mask[i : i + input_shape[1], j : j + input_shape[0]] = cropped_images[k]
            k = k + 1

    return img_mask
