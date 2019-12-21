import numpy as np
np.random.seed(42)

import os
import cv2
import pandas as pd
from generator import generator
import matplotlib.pyplot as plt
from keras.optimizers import Adam
from keras.utils import plot_model
from keras.models import load_model, Model
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.layers import BatchNormalization, Dropout, Concatenate
from keras.layers import Conv2D, MaxPool2D, Input, Conv2DTranspose, Add

def get_epoch_len(lbl, polygon_path):
    df = pd.read_csv(polygon_path)
    df = df[df['ClassType'] == lbl]
    return len(df)

class network:
    def __init__(self, model_path = None, input_shape = 1024):

        self.input_shape = (input_shape, input_shape, 3)

        if model_path == None:
            self.model = self.create_model()
        else:
            self.model = load_model(model_path)

    def create_model(self):

        input = Input(shape = self.input_shape)
        filters = 16
        #encoder
        conv_1 = Conv2D(filters, (3, 3), padding = 'same', activation = 'relu')(input)
        conv_2 = Conv2D(filters, (3, 3), padding = 'same', activation = 'relu')(conv_1)
        pool_1 = MaxPool2D((2, 2))(conv_2)
        drop_1 = Dropout(0.5)(pool_1)

        conv_3 = Conv2D(filters * 2, (3, 3), padding = 'same', activation = 'relu')(drop_1)
        conv_4 = Conv2D(filters * 2, (3, 3), padding = 'same', activation = 'relu')(conv_3)
        pool_2 = MaxPool2D((2, 2))(conv_4)
        drop_2 = Dropout(0.5)(pool_2)

        conv_5 = Conv2D(filters * 4, (3, 3), padding = 'same', activation = 'relu')(drop_2)
        conv_6 = Conv2D(filters * 4, (3, 3), padding = 'same', activation = 'relu')(conv_5)
        pool_3 = MaxPool2D((2, 2))(conv_6)
        drop_3 = Dropout(0.5)(pool_3)

        conv_7 = Conv2D(filters * 8, (3, 3), padding = 'same', activation = 'relu')(drop_3)
        conv_8 = Conv2D(filters * 8, (3, 3), padding = 'same', activation = 'relu')(conv_7)
        pool_4 = MaxPool2D((2, 2))(conv_8)
        drop_4 = Dropout(0.5)(pool_4)

        #bridge
        conv_b1 = Conv2D(filters * 16, (3, 3), padding = 'same', activation = 'relu')(drop_4)
        conv_b2 = Conv2D(filters * 16, (3, 3), padding = 'same', activation = 'relu')(conv_b1)

        #decoder
        upsm_1 = Conv2DTranspose(filters * 8, (3, 3), strides = (2, 2), padding = 'same')(conv_b2)
        concat_3 = Concatenate()([upsm_1, conv_8])
        drop_5 = Dropout(0.5)(concat_3)
        conv_9 = Conv2D(filters * 8, (3, 3), padding = 'same', activation = 'relu')(drop_5)
        conv_10 = Conv2D(filters * 8, (3, 3), padding = 'same', activation = 'relu')(conv_9)

        upsm_2 = Conv2DTranspose(filters * 4, (3, 3), strides = (2, 2), padding = 'same')(conv_10)
        concat_4 = Concatenate()([upsm_2, conv_6])
        drop_6 = Dropout(0.5)(concat_4)
        conv_11 = Conv2D(filters * 4, (3, 3), padding = 'same', activation = 'relu')(drop_6)
        conv_12 = Conv2D(filters * 4, (3, 3), padding = 'same', activation = 'relu')(conv_11)

        upsm_3 = Conv2DTranspose(filters * 8, (3, 3), strides = (2, 2), padding = 'same')(conv_12)
        concat_5 = Concatenate()([upsm_3, conv_4])
        drop_7 = Dropout(0.5)(concat_5)
        conv_13 = Conv2D(filters * 2, (3, 3), padding = 'same', activation = 'relu')(drop_7)
        conv_14 = Conv2D(filters * 2, (3, 3), padding = 'same', activation = 'relu')(conv_13)

        upsm_4 = Conv2DTranspose(filters * 8, (3, 3), strides = (2, 2), padding = 'same')(conv_14)
        concat_6 = Concatenate()([upsm_4, conv_2])
        drop_8 = Dropout(0.5)(concat_6)
        conv_15 = Conv2D(filters * 1, (3, 3), padding = 'same', activation = 'relu')(drop_8)
        conv_16 = Conv2D(filters * 1, (3, 3), padding = 'same', activation = 'relu')(conv_15)
        output = Conv2D(1, (1, 1), padding = 'same', activation = 'sigmoid')(conv_16)

        net = Model(input, output)
        net.summary()
        plot_model(net, 'model.png', show_shapes = True)

        return net

    def train(self, train_path, polygon_path, scaler_path, epochs, bs, lr, callback_dir, log_dir, lbl):

        train_gen = generator(lbl, img_path = train_path, polygon_path = polygon_path, scaler_path = scaler_path, bs = int(bs), input_shape = self.input_shape)

        self.model.compile(loss = 'binary_crossentropy', optimizer = Adam(lr = lr, decay = lr // epochs), metrics = ['mae'])
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

    def predict(self, img):
        img = cv2.resize(img, self.input_shape[:2])
        img = np.expand_dims(img, axis = 0)

        out_img = self.model.predict(img)[0]
        out_img = out_img > 0.95

        cv2.imshow("img", img[0])
        cv2.imshow("preds", out_img.astype('int') * 255.0)
        cv2.waitKey(0)

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
