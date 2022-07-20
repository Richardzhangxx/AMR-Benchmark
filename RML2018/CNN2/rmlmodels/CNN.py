import os

from keras.models import Model
from keras.layers import Input,Dense,ReLU,Dropout,Softmax,Conv2D,MaxPool2D
from keras.layers import Bidirectional,Flatten,CuDNNGRU
from keras.utils.vis_utils import plot_model

def CNN(weights=None,
             input_shape=[2,1024],
             classes=11,
             **kwargs):
    if weights is not None and not (os.path.exists(weights)):
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization), '
                         'or the path to the weights file to be loaded.')
    
    input = Input(input_shape+[1],name='input')
    x = input

    x=Conv2D(256,(2,8),padding='same', activation="relu", name="conv1", kernel_initializer='glorot_uniform')(input)
    x=MaxPool2D(pool_size=(1,2))(x)
    x=Dropout(0.5)(x)
    x=Conv2D(128,(2,8),padding='same', activation="relu", name="conv2", kernel_initializer='glorot_uniform')(x)
    x=MaxPool2D(pool_size=(1,2))(x)
    x=Dropout(0.5)(x)
    x=Conv2D(64,(2,8),padding='same', activation="relu", name="conv3", kernel_initializer='glorot_uniform')(x)
    x=MaxPool2D(pool_size=(1,2))(x)
    x=Dropout(0.5)(x)
    x=Conv2D(64,(2,8),padding='same', activation="relu", name="conv4", kernel_initializer='glorot_uniform')(x)
    x=MaxPool2D(pool_size=(1,2))(x)
    x=Dropout(0.5)(x)
    x=Flatten()(x)
    x = Dense(128,activation='relu',name='dense1')(x)
    x = Dense(24,activation='softmax',name='dense2')(x)

    model = Model(inputs = input,outputs = x)

    # Load weights.
    if weights is not None:
        model.load_weights(weights)

    return model

import keras
if __name__ == '__main__':
    model =  CNN(None,input_shape=[2,1024],classes=24)

    adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=adam)

    print('models layers:', model.layers)
    print('models config:', model.get_config())
    print('models summary:', model.summary())