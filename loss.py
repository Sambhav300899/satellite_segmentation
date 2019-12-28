import keras.backend as K
from keras.losses import binary_crossentropy

def dice_coef(y_true, y_pred, epsilon = 1e-6):
    axes = tuple(range(1, len(y_pred.shape) - 1))

    num = 2. * K.sum(y_pred * y_true, axes)
    denum = K.sum(y_pred + y_true, axes)

    return 1 - K.mean(num / (denum + epsilon))

def loss(y_true, y_pred):
    #return binary_crossentropy(y_true, y_pred) - K.log(jaccard_coef(y_true, y_pred))
    #return dice_coef(y_true, y_pred) + binary_crossentropy(y_true, y_pred)
    return dice_coef(y_true, y_pred)
    #return binary_crossentropy(y_true, y_pred)
