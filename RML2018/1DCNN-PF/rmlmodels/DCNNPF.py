import os

WEIGHTS_PATH = ('resnet_like_weights_tf_dim_ordering_tf_kernels.h5')

from keras.models import Model
from keras.layers import Input,Dense,Conv1D,MaxPool1D,ReLU,Dropout,Softmax,concatenate,Flatten,Reshape
from keras.layers.convolutional import Conv2D
from keras.layers import CuDNNLSTM


def DCNNPF(weights=None,
             input_shape=[1024],
             classes=24,
             **kwargs):
    if weights is not None and not (os.path.exists(weights)):
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization), '
                         'or the path to the weights file to be loaded.')

    dr = 0.5  # dropout rate (%)
    tap = 8
    
    input1 =Input(input_shape,name='input1')
    Reshape1=Reshape(input_shape+[1])(input1)
    input2=Input(input_shape,name='input2')
    Reshape2=Reshape(input_shape+[1])(input2)
    #Cnvolutional Block
    # L = 4
    # for i in range(L):
    #     x = ConvBNReluUnit(x,kernel_size = tap,index=i)

    # SeparateChannel Combined Convolutional Neural Networks
    
    x2=Conv1D(64,3, activation="relu" )(Reshape1)
    x2=Dropout(0.2)(x2)
    x2=Conv1D(64,3, activation="relu" )(x2)
    x2=Dropout(0.2)(x2)
    x2=Conv1D(64,3, activation="relu" )(x2)
    x2=Dropout(0.2)(x2)
    x2=Conv1D(64,3, activation="relu" )(x2)
    x2=Dropout(0.2)(x2)
    
    x3=Conv1D(64,3, activation="relu" )(Reshape2)
    x3=Dropout(0.2)(x3)
    x3=Conv1D(64,3, activation="relu" )(x3)
    x3=Dropout(0.2)(x3)
    x3=Conv1D(64,3, activation="relu" )(x3)
    x3=Dropout(0.2)(x3)
    x3=Conv1D(64,3, activation="relu" )(x3)
    x3=Dropout(0.2)(x3)

    x=concatenate([x2,x3])
    x=Conv1D(64,3, activation="relu" )(x)
    x=Dropout(0.2)(x)   
    x=MaxPool1D(pool_size=2)(x)
    x=Conv1D(64,3, activation="relu")(x)
    x=Dropout(0.2)(x)   
    x=MaxPool1D(pool_size=2)(x)
    x=Conv1D(64,3, activation="relu")(x)
    x=Dropout(0.2)(x)   
    x=MaxPool1D(pool_size=2)(x)
    x=Conv1D(64,3, activation="relu")(x)
    x=Dropout(0.2)(x)   
    x=MaxPool1D(pool_size=2)(x)    
    x=Conv1D(64,3, activation="relu")(x)
    x=Dropout(0.2)(x)   
    x=MaxPool1D(pool_size=2)(x)
    x=Flatten()(x)
    x = Dense(128,activation='selu',name='fc1')(x)
    x=Dropout(dr)(x)
    x = Dense(128,activation='selu',name='fc2')(x)
    x=Dropout(dr)(x)
    x = Dense(classes,activation='softmax',name='softmax')(x)

    model = Model(inputs = [input1,input2],outputs = x)

    # Load weights.
    if weights is not None:
        model.load_weights(weights)

    return model

import keras
from keras.utils.vis_utils import plot_model
if __name__ == '__main__':
    model = DCNNPF(None,classes=11)

    adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=adam)
    #plot_model(model, to_file='model.png',show_shapes=True) # print model
    print('models layers:', model.layers)
    print('models config:', model.get_config())
    print('models summary:', model.summary())