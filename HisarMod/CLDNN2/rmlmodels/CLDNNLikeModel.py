"""CLDNNLike model for RadioML.

# Reference:

- [CONVOLUTIONAL,LONG SHORT-TERM MEMORY, FULLY CONNECTED DEEP NEURAL NETWORKS ]

Adapted from code contributed by Mika.
"""
import os
import numpy as np

from keras.models import Model
from keras.layers import Input,Dense,Conv1D,MaxPool1D,ReLU,Dropout,Softmax,concatenate,Conv2D
from keras.layers import LSTM,Permute,Reshape,ZeroPadding2D,Activation




def CLDNNLikeModel(weights=None,
             input_shape1=[2,1024],
             classes=26,
             **kwargs):
    if weights is not None and not (os.path.exists(weights)):
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization), '
                         'or the path to the weights file to be loaded.')

# %%
    dr = 0.5
    input_x = Input(input_shape1+[1],name='input')

    x = Conv2D(256, (1, 3), activation="relu", name="conv1", init='glorot_uniform')(input_x) # (b,c,h,w) (b,h,w,c)
    x = Dropout(dr)(x)
    x = Conv2D(256, (2, 3), activation="relu", name="conv2", init='glorot_uniform')(x)  # (b,c,h,w) (b,h,w,c)
    x = Dropout(dr)(x)
    x = Conv2D(80, (1, 3), activation="relu", name="conv3", init='glorot_uniform')(x)  # (b,c,h,w) (b,h,w,c)
    x = Dropout(dr)(x)
    x = Conv2D(80, (1, 3), activation="relu", name="conv4", init='glorot_uniform')(x)  # (b,c,h,w) (b,h,w,c)
    x = Dropout(dr)(x)
    x1 = Reshape((80, 1016))(x)
    lstm_out = LSTM(units=50)(x1)

    # layer_Flatten = Flatten()(lstm_out)
    x = Dense(128, activation='relu', name="dense1")(lstm_out)
    x = Dropout(dr)(x)
    output = Dense(26, activation='softmax',name="dense2")(x)

    model = Model(inputs=input_x, outputs=output)

    # Load weights.
    if weights is not None:
        model.load_weights(weights)

    return model

import keras
if __name__ == '__main__':
    model = CLDNNLikeModel(None,input_shape=(2,1024),classes=24)

    adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=adam)

    print('models layers:', model.layers)
    print('models config:', model.get_config())
    print('models summary:', model.summary())