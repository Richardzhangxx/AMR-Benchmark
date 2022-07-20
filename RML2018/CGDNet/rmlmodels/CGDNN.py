"""CLDNNLike model for RadioML.

# Reference:

- [CONVOLUTIONAL,LONG SHORT-TERM MEMORY, FULLY CONNECTED DEEP NEURAL NETWORKS ]

Adapted from code contributed by Mika.
"""
import os
import tensorflow as tf
WEIGHTS_PATH = ('resnet_like_weights_tf_dim_ordering_tf_kernels.h5')
import math
from keras.models import Model
from keras.layers import Input, Dense, Conv1D, MaxPool1D, ReLU, Dropout, Softmax, concatenate, Flatten, Reshape, \
    GaussianNoise,GaussianDropout
from keras.layers.convolutional import Conv2D
from keras.layers import CuDNNLSTM,Lambda,Multiply,Add,Subtract,MaxPool2D,CuDNNGRU,LeakyReLU,BatchNormalization
import tensorflow as tf


def CGDNN(weights=None,
           input_shape=[1, 2, 1024],
           classes=24,
           **kwargs):
    if weights is not None and not (os.path.exists(weights)):
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization), '
                         'or the path to the weights file to be loaded.')

    dr = 0.2  # dropout rate (%)
    input = Input(input_shape, name='input1')
    x1 = Conv2D(50, (1, 6), activation='relu', kernel_initializer='lecun_uniform', data_format="channels_first")(input)
    x1 = MaxPool2D(pool_size=(2, 2), strides=1, padding='same', data_format="channels_first")(x1)
    x1 = GaussianDropout(dr)(x1)
    x2 = Conv2D(50, (1, 6), activation='relu', kernel_initializer='glorot_uniform', data_format="channels_first")(x1)
    x2 = MaxPool2D(pool_size=(2, 2), strides=1, padding='same', data_format="channels_first")(x2)
    x2 = GaussianDropout(dr)(x2)
    x3 = Conv2D(50, (1, 6), activation='relu', kernel_initializer='glorot_uniform', data_format="channels_first")(x2)
    x3 = MaxPool2D(pool_size=(2, 2), strides=1, padding='same', data_format="channels_first")(x3)
    x3 = GaussianDropout(dr)(x3)
    x11 = concatenate([x1, x3], 3)
    x4 = Reshape(target_shape=((50, 4056)), name='reshape4')(x11)
    x4 = CuDNNGRU(units=50)(x4)
    x4 = GaussianDropout(dr)(x4)
    x = Dense(256, activation='relu', name='fc4',kernel_initializer='he_normal')(x4)
    x = GaussianDropout(dr)(x)
    x = Dense(classes, activation='softmax', name='softmax')(x)
    model = Model(inputs=input, outputs=x)

    # Load weights.
    if weights is not None:
        model.load_weights(weights)

    return model


import keras
from keras.utils.vis_utils import plot_model

if __name__ == '__main__':
    model = CGDNN(None, classes=10)

    adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=adam)
    plot_model(model, to_file='model.png', show_shapes=True)  # print model
    print('models layers:', model.layers)
    print('models config:', model.get_config())
    print('models summary:', model.summary())