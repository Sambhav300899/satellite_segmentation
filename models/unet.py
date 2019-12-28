from keras.models import Model
from keras.layers import BatchNormalization, Dropout, Concatenate
from keras.layers import Conv2D, MaxPool2D, Input, Conv2DTranspose

def get_model(input_shape):
    input = Input(shape = input_shape)
    filters = 16
    #encoder
    conv_1 = Conv2D(filters, (3, 3), padding = 'same', activation = 'relu')(input)
    conv_2 = Conv2D(filters, (3, 3), padding = 'same', activation = 'relu')(conv_1)
    BN1 = BatchNormalization()(conv_2)
    pool_1 = MaxPool2D((2, 2))(BN1)
    drop_1 = Dropout(0.25)(pool_1)

    conv_3 = Conv2D(filters * 2, (3, 3), padding = 'same', activation = 'relu')(drop_1)
    conv_4 = Conv2D(filters * 2, (3, 3), padding = 'same', activation = 'relu')(conv_3)
    BN2 = BatchNormalization()(conv_4)
    pool_2 = MaxPool2D((2, 2))(BN2)
    drop_2 = Dropout(0.25)(pool_2)

    conv_5 = Conv2D(filters * 4, (3, 3), padding = 'same', activation = 'relu')(drop_2)
    conv_6 = Conv2D(filters * 4, (3, 3), padding = 'same', activation = 'relu')(conv_5)
    BN3 = BatchNormalization()(conv_6)
    pool_3 = MaxPool2D((2, 2))(BN3)
    drop_3 = Dropout(0.25)(pool_3)

    conv_7 = Conv2D(filters * 8, (3, 3), padding = 'same', activation = 'relu')(drop_3)
    conv_8 = Conv2D(filters * 8, (3, 3), padding = 'same', activation = 'relu')(conv_7)
    BN4 = BatchNormalization()(conv_8)
    pool_4 = MaxPool2D((2, 2))(BN4)
    drop_4 = Dropout(0.25)(pool_4)

    #bridge
    conv_b1 = Conv2D(filters * 16, (3, 3), padding = 'same', activation = 'relu')(drop_4)
    conv_b2 = Conv2D(filters * 16, (3, 3), padding = 'same', activation = 'relu')(conv_b1)
    BN5 = BatchNormalization()(conv_b2)

    #decoder
    upsm_1 = Conv2DTranspose(filters * 8, (3, 3), strides = (2, 2), padding = 'same')(BN5)
    concat_3 = Concatenate()([upsm_1, conv_8])
    drop_5 = Dropout(0.5)(concat_3)
    conv_9 = Conv2D(filters * 8, (3, 3), padding = 'same', activation = 'relu')(drop_5)
    conv_10 = Conv2D(filters * 8, (3, 3), padding = 'same', activation = 'relu')(conv_9)
    BN6 = BatchNormalization()(conv_10)

    upsm_2 = Conv2DTranspose(filters * 4, (3, 3), strides = (2, 2), padding = 'same')(BN6)
    concat_4 = Concatenate()([upsm_2, conv_6])
    drop_6 = Dropout(0.5)(concat_4)
    conv_11 = Conv2D(filters * 4, (3, 3), padding = 'same', activation = 'relu')(drop_6)
    conv_12 = Conv2D(filters * 4, (3, 3), padding = 'same', activation = 'relu')(conv_11)
    BN7 = BatchNormalization()(conv_12)

    upsm_3 = Conv2DTranspose(filters * 8, (3, 3), strides = (2, 2), padding = 'same')(BN7)
    concat_5 = Concatenate()([upsm_3, conv_4])
    drop_7 = Dropout(0.5)(concat_5)
    conv_13 = Conv2D(filters * 2, (3, 3), padding = 'same', activation = 'relu')(drop_7)
    conv_14 = Conv2D(filters * 2, (3, 3), padding = 'same', activation = 'relu')(conv_13)
    BN8 = BatchNormalization()(conv_14)

    upsm_4 = Conv2DTranspose(filters * 8, (3, 3), strides = (2, 2), padding = 'same')(BN8)
    concat_6 = Concatenate()([upsm_4, conv_2])
    drop_8 = Dropout(0.5)(concat_6)
    conv_15 = Conv2D(filters * 1, (3, 3), padding = 'same', activation = 'relu')(drop_8)
    conv_16 = Conv2D(filters * 1, (3, 3), padding = 'same', activation = 'relu')(conv_15)
    BN9 = BatchNormalization()(conv_16)

    output = Conv2D(1, (1, 1), padding = 'same', activation = 'sigmoid')(BN9)

    net = Model(input, output)
    return net
