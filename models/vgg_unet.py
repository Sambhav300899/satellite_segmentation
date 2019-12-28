from keras.models import Model
from keras.utils import plot_model
from keras.applications import VGG16
from keras.layers import BatchNormalization, Dropout, Concatenate
from keras.layers import Conv2D, MaxPool2D, Input, Conv2DTranspose

def get_model(input_shape):
    conv_base = VGG16(input_shape = input_shape, weights = 'imagenet', include_top = False)
    filters = 8
    conv_b1 = Conv2D(256, (3, 3), padding = 'same', activation = 'relu')(conv_base.output)
    conv_b2 = Conv2D(256, (3, 3), padding = 'same', activation = 'relu')(conv_b1)

    upsm_1 = Conv2DTranspose(filters * 8, (3, 3), strides = (2, 2), padding = 'same')(conv_b2)
    concat_3 = Concatenate()([upsm_1, conv_base.get_layer('block5_conv3').output])
    drop_5 = Dropout(0.5)(concat_3)
    conv_9 = Conv2D(filters * 8, (3, 3), padding = 'same', activation = 'relu')(drop_5)
    conv_10 = Conv2D(filters * 8, (3, 3), padding = 'same', activation = 'relu')(conv_9)
    BN6 = BatchNormalization()(conv_10)

    upsm_2 = Conv2DTranspose(filters * 4, (3, 3), strides = (2, 2), padding = 'same')(BN6)
    concat_4 = Concatenate()([upsm_2, conv_base.get_layer('block4_conv3').output])
    drop_6 = Dropout(0.5)(concat_4)
    conv_11 = Conv2D(filters * 4, (3, 3), padding = 'same', activation = 'relu')(drop_6)
    conv_12 = Conv2D(filters * 4, (3, 3), padding = 'same', activation = 'relu')(conv_11)
    BN7 = BatchNormalization()(conv_12)

    upsm_3 = Conv2DTranspose(filters * 8, (3, 3), strides = (2, 2), padding = 'same')(BN7)
    concat_5 = Concatenate()([upsm_3, conv_base.get_layer('block3_conv3').output])
    drop_7 = Dropout(0.5)(concat_5)
    conv_13 = Conv2D(filters * 2, (3, 3), padding = 'same', activation = 'relu')(drop_7)
    conv_14 = Conv2D(filters * 2, (3, 3), padding = 'same', activation = 'relu')(conv_13)
    BN8 = BatchNormalization()(conv_14)

    upsm_4 = Conv2DTranspose(filters * 8, (3, 3), strides = (2, 2), padding = 'same')(BN8)
    concat_6 = Concatenate()([upsm_4, conv_base.get_layer('block2_conv2').output])
    drop_8 = Dropout(0.5)(concat_6)
    conv_15 = Conv2D(filters * 1, (3, 3), padding = 'same', activation = 'relu')(drop_8)
    conv_16 = Conv2D(filters * 1, (3, 3), padding = 'same', activation = 'relu')(conv_15)
    BN9 = BatchNormalization()(conv_16)

    upsm_5 = Conv2DTranspose(filters * 8, (3, 3), strides = (2, 2), padding = 'same')(BN9)
    concat_7 = Concatenate()([upsm_5, conv_base.get_layer('block1_conv2').output])
    drop_9 = Dropout(0.5)(concat_7)
    conv_17 = Conv2D(filters * 1, (3, 3), padding = 'same', activation = 'relu')(drop_9)
    conv_18 = Conv2D(filters * 1, (3, 3), padding = 'same', activation = 'relu')(conv_17)
    BN10 = BatchNormalization()(conv_18)

    output = Conv2D(1, (1, 1), padding = 'same', activation = 'sigmoid')(BN10)

    net = Model(conv_base.input, output)

    for layer in net.layers:
        layer.trainable = False

        if layer.name == "block5_pool":
            break

    return net
