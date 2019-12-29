import numpy as np
import tensorflow as tf
import keras.backend as K
from keras.losses import binary_crossentropy

def dice_loss(y_true, y_pred):
  numerator = 2 * tf.reduce_sum(y_true * y_pred, axis=-1)
  denominator = tf.reduce_sum(y_true + y_pred, axis=-1)

  return 1 - (numerator + 1) / (denominator + 1)

def iou_coef(y_true, y_pred, smooth=1):
    y_pred = (y_pred > 0.5)
    y_pred = K.cast(y_pred, 'float')
    axes = tuple(range(1, len(y_pred.shape) - 1))

    inter = K.sum(K.abs(y_true * y_pred), axis = axes)
    union = K.sum(y_true, axes) + K.sum(y_pred, axes) - inter
    iou = K.mean((inter + smooth) / (union + smooth), axis = 0)

    return iou

def loss(y_true, y_pred):
    #return binary_crossentropy(y_true, y_pred) - K.log(jaccard_coef(y_true, y_pred))
    #return dice_coef(y_true, y_pred) + binary_crossentropy(y_true, y_pred)
    #return dice_coef(y_true, y_pred)
    return dice_loss(y_true, y_pred) + 0.5 * binary_crossentropy(y_true, y_pred)
