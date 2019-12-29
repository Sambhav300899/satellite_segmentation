import numpy as np
np.random.seed(42)

import os
import cv2
import pandas as pd
import tifffile as tiff
import keras.backend as K
from generator import generator
import matplotlib.pyplot as plt
from loss import loss, dice_loss
from models import vgg_unet, unet
from keras.optimizers import Adam
from keras.utils import plot_model
from keras.metrics import MeanIoU
from keras.callbacks import TensorBoard, ModelCheckpoint
from utils import stretch_nbit, get_mask, create_crops, stitch

def get_epoch_len(lbl, polygon_path):
    df = pd.read_csv(polygon_path)
    df = df[df['ClassType'] == lbl]
    return len(df)

def iou(y_true, y_pred):
    intersection = y_true * y_pred

    notTrue = 1 - y_true
    union = y_true + (notTrue * y_pred)

    return K.sum(intersection)/K.sum(union)

class network:
    def __init__(self, model_path = None, input_shape = 512):

        self.input_shape = (input_shape, input_shape, 3)
        self.model = self.create_model()

        if model_path != None:
            self.model.load_weights(model_path)


    def create_model(self):
        net = vgg_unet.get_model(self.input_shape)
        net.summary()
        plot_model(net, 'model.png', show_shapes = True)

        return net

    def train(self, train_path, polygon_path, scaler_path, epochs, bs, lr, callback_dir, log_dir, lbl):

        train_gen = generator(lbl, img_path = train_path, polygon_path = polygon_path, scaler_path = scaler_path, bs = int(bs), input_shape = self.input_shape)

        self.model.compile(loss = loss, optimizer = Adam(lr = lr, decay = lr // epochs), metrics = [dice_loss, 'binary_crossentropy'])

        filepath = callback_dir + os.path.sep + "weights-improvement-{epoch:02d}-{loss:.2f}.hdf5"

        callbacks = [
        ModelCheckpoint(filepath, monitor = 'loss', verbose = 1, save_best_only = True, save_weights_only=False),
        TensorBoard(log_dir = log_dir)
        ]

        H = self.model.fit_generator(
        train_gen,
        steps_per_epoch = (get_epoch_len(lbl, polygon_path) * ((3300 / self.input_shape[0]) ** 2)) // bs,
        #validation_data = val_gen,
        #validation_steps = len(os.listdir(val_path)) // bs,
        epochs = epochs,
        shuffle = True,
        verbose = 1,
        callbacks = callbacks
        )

        self.model.save('classifier.hdf5')
        self.plot_data(H, epochs)

    def predict(self, img_path, scalar_path, polygon_path, lbl):
        img = tiff.imread(img_path).transpose([1, 2, 0])
        mask = None

        try:
            df_grid = pd.read_csv(scalar_path)
            df_poly = pd.read_csv(polygon_path)
            mask = get_mask(df_grid, df_poly, img_path.split('.')[0], img.shape, lbl)
            _, mask = create_crops(mask, self.input_shape)
            mask = cv2.resize(mask, (512, 512))
        except:
            print("no scaler and polygon path found, displaying predection without ground truth")

        img_rgb = stretch_nbit(img)
        img_rgb = cv2.resize(img_rgb, (3300, 3300))
        img_crops,img_rgb = create_crops(img_rgb, self.input_shape)
        mask_crops = []

        for img_crop in img_crops:
            print("processing")
            input = np.expand_dims(img_crop, axis = 0)
            out_img = self.model.predict(input.astype('float') / 255.0)[0]
            out_img = out_img > 0.5
            mask_crops.append(out_img.astype('int'))

        output = stitch(mask_crops, self.input_shape)

        output = cv2.resize(output, (512, 512))
        img_rgb = cv2.resize(img_rgb, (512, 512))

        cv2.imshow("img", img_rgb)
        cv2.imshow("preds", output * 255.0)

        if mask is not None:
            cv2.imshow("ground_truth", mask * 255.0)
            print (iou(mask, output))

        cv2.waitKey(0)
        cv2.imwrite("output.jpeg", output * 255.0)

    def evaluate(self, img_dir):
        pass

    def plot_data(self, H, n):
        plt.plot(np.arange(0, n), H.history['loss'], label = 'train loss', color = 'g')
        #plt.plot(np.arange(0, n), H.history['val_loss'], label = 'validation loss', color = 'b')
        plt.title('training loss')
        plt.xlabel('Epoch #')
        plt.ylabel('loss')
        plt.legend(loc = 'upper right')
        plt.savefig('graph_test.png')
