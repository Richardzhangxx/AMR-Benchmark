import os
os.environ["KERAS_BACKEND"] = "tensorflow"
# os.environ["THEANO_FLAGS"]  = "device=gpu%d"%(0)
import numpy as np
import keras.models as models
from keras.layers.core import Reshape,Dense,Dropout,Activation,Flatten
from keras.layers.convolutional import Convolution2D,Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers.core import Activation
from keras.layers import Input, Dense, Conv1D, MaxPool1D, ReLU, Dropout, Softmax, concatenate, Flatten, Reshape, \
    GaussianNoise,Activation,GaussianDropout
from keras.models import Model


def CNN2Model(weights=None,
             input_shape=[2,1024],
             classes=26,
             **kwargs):
    if weights is not None and not (os.path.exists(weights)):
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization), '
                         'or the path to the weights file to be loaded.')
    dr = 0.5
    input = Input(input_shape + [1], name='input1')

    x = Conv2D(50, (1, 8), padding='same', activation="relu", name="conv1", kernel_initializer='glorot_uniform')(input)
    x = Dropout(dr)(x)
    x = Conv2D(50, (2, 8), padding="valid", activation="relu", name="conv2", kernel_initializer='glorot_uniform')(x)
    x = Dropout(dr)(x)
    x = Flatten()(x)
    x = Dense(256, activation='relu', kernel_initializer='he_normal', name="dense1")(x)
    x = Dropout(dr)(x)
    x = Dense(classes, activation='softmax', name='softmax')(x)
    model = Model(inputs=input, outputs=x)

    # Load weights.
    if weights is not None:
        model.load_weights(weights)

    return model

if __name__ == '__main__':
    print(CNN2Model().summary())
