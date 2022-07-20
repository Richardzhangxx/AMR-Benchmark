import os
import numpy as np

from keras.models import Model
from keras.layers import Input,Dense,Conv1D,MaxPool1D,ReLU,Dropout,Softmax,concatenate,Conv2D,CuDNNLSTM
from keras.layers import LSTM,Permute,Reshape,ZeroPadding2D,Activation




def CLDNNLikeModel(weights=None,
             input_shape1=[2,128],
             classes=11,
             **kwargs):
    if weights is not None and not (os.path.exists(weights)):
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization), '
                         'or the path to the weights file to be loaded.')

    dr = 0.5
    input_x = Input(shape=(1, 2, 128))

    input_x_padding = ZeroPadding2D((0, 2), data_format="channels_first")(input_x)

    layer11 = Conv2D(50, (1, 8), padding='valid', activation="relu", name="conv11", init='glorot_uniform',
                    data_format="channels_first")(input_x_padding)
    layer11 = Dropout(dr)(layer11)

    layer11_padding = ZeroPadding2D((0, 2), data_format="channels_first")(layer11)
    layer12 = Conv2D(50, (1, 8), padding="valid", activation="relu", name="conv12", init='glorot_uniform',
                    data_format="channels_first")(layer11_padding)
    layer12 = Dropout(dr)(layer12)

    layer12 = ZeroPadding2D((0, 2), data_format="channels_first")(layer12)
    layer13 = Conv2D(50, (1, 8), padding='valid', activation="relu", name="conv13", init='glorot_uniform',
                    data_format="channels_first")(layer12)
    layer13 = Dropout(dr)(layer13)

    concat = keras.layers.concatenate([layer11, layer13])
    concat_size = list(np.shape(concat))
    input_dim = int(concat_size[-1] * concat_size[-2])
    timesteps = int(concat_size[-3])
    concat = Reshape((timesteps, input_dim))(concat)
    lstm_out = CuDNNLSTM(units=50)(concat)

    layer_dense1 = Dense(256, activation='relu', init='he_normal', name="dense1")(lstm_out)
    layer_dropout = Dropout(dr)(layer_dense1)
    layer_dense2 = Dense(11, init='he_normal', name="dense2")(layer_dropout)
    layer_softmax = Activation('softmax')(layer_dense2)
    output = Reshape([11])(layer_softmax)

    model = Model(inputs=input_x, outputs=output)

    # Load weights.
    if weights is not None:
        model.load_weights(weights)

    return model

import keras
if __name__ == '__main__':
    model = CLDNNLikeModel(None,input_shape=(2,128),classes=11)

    adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=adam)

    print('models layers:', model.layers)
    print('models config:', model.get_config())
    print('models summary:', model.summary())