import numpy as np
np.random.seed(42)

import os
import cv2
import pandas as pd
from models import unet
import tifffile as tiff
import keras.backend as K
from generator import generator
import matplotlib.pyplot as plt
from keras.optimizers import Adam
from keras.utils import plot_model
from keras.models import load_model
from utils import get_rgb_img, get_mask
from keras.losses import binary_crossentropy
from keras.callbacks import TensorBoard, ModelCheckpoint

def get_epoch_len(lbl, polygon_path):
    df = pd.read_csv(polygon_path)
    df = df[df['ClassType'] == lbl]
    return len(df)

def jaccard_coef(y_true, y_pred, epsilon = 1e-6):
    axes = tuple(range(1, len(y_pred.shape)-1))

    num = K.sum(y_pred * y_true, axes)
    denum = K.sum(y_true + y_pred - y_pred * y_true)

    return K.mean(num / (denum + epsilon))

def dice_coef(y_true, y_pred, epsilon = 1e-6):
    axes = tuple(range(1, len(y_pred.shape)-1))

    num = 2. * K.sum(y_pred * y_true, axes)
    denum = K.sum(K.square(y_pred) + K.square(y_true), axes)

    return 1 - K.mean(num / (denum + epsilon))

def loss(y_true, y_pred):
    #return binary_crossentropy(y_true, y_pred) - K.log(jaccard_coef(y_true, y_pred))
    return dice_coef(y_true, y_pred) + binary_crossentropy(y_true, y_pred)

class network:
    def __init__(self, model_path = None, input_shape = 1024):

        self.input_shape = (input_shape, input_shape, 3)

        if model_path == None:
            self.model = self.create_model()
        else:
            self.model = load_model(model_path, custom_objects = {'dice_coef' : dice_coef, 'loss' : loss , 'jaccard_coef' : jaccard_coef})

    def create_model(self):
        net = unet.get_model(self.input_shape)
        net.summary()
        plot_model(net, 'model.png', show_shapes = True)

        return net

    def train(self, train_path, polygon_path, scaler_path, epochs, bs, lr, callback_dir, log_dir, lbl):

        train_gen = generator(lbl, img_path = train_path, polygon_path = polygon_path, scaler_path = scaler_path, bs = int(bs), input_shape = self.input_shape)

        self.model.compile(loss = dice_coef, optimizer = Adam(lr = lr, decay = lr // epochs), metrics = ['binary_crossentropy', dice_coef, jaccard_coef])
        #self.model.compile(loss = 'mae', optimizer = 'adadelta', metrics = ['mse'])
        #self.model.compile(loss = 'mae', optimizer = RAdam(), metrics = ['mse'])

        filepath = callback_dir + os.path.sep + "weights-improvement-{epoch:02d}-{loss:.2f}.hdf5"

        callbacks = [
        ModelCheckpoint(filepath, monitor = 'loss', verbose = 1, save_best_only = True, save_weights_only=False),
        TensorBoard(log_dir = log_dir)
        ]

        H = self.model.fit_generator(
        train_gen,
        steps_per_epoch = get_epoch_len(lbl, polygon_path) // bs,
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
            mask = get_mask(df_grid, df_poly, img_path.split('.')[0][:-2], img.shape, lbl)
            mask = cv2.resize(mask, self.input_shape[:2])
        except:
            print("no scaler and polygon path found, displaying predection without ground truth")

        img_rgb = get_rgb_img(img)
        img_rgb = cv2.resize(img_rgb, self.input_shape[:2])
        img_rgb = np.expand_dims(img_rgb, axis = 0)

        out_img = self.model.predict(img_rgb.astype('float') / 255.0)[0]
        out_img = out_img > 0.5

        cv2.imshow("img", img_rgb[0])
        cv2.imshow("preds", out_img.astype('int') * 255.0)

        if mask is not None:
            cv2.imshow("ground_truth", mask * 255.0)

        cv2.waitKey(0)
        cv2.imwrite("output.jpeg", out_img.astype('int') * 255.0)

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
