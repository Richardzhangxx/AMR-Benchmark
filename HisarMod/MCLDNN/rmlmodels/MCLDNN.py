"""CLDNNLike model for RadioML.

# Reference:

- [CONVOLUTIONAL,LONG SHORT-TERM MEMORY, FULLY CONNECTED DEEP NEURAL NETWORKS ]

Adapted from code contributed by Mika.
"""
import os

WEIGHTS_PATH = ('resnet_like_weights_tf_dim_ordering_tf_kernels.h5')

from keras.models import Model
from keras.layers import Input,Dense,Conv1D,MaxPool1D,ReLU,Dropout,Softmax,concatenate,Flatten,Reshape
from keras.layers.convolutional import Conv2D
from keras.layers import CuDNNLSTM


def MCLDNN(weights=None,
             input_shape1=[2,1024],
             input_shape2=[1024,1],
             classes=26,
             **kwargs):
    if weights is not None and not (os.path.exists(weights)):
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization), '
                         'or the path to the weights file to be loaded.')

    dr = 0.5  # dropout rate (%)
    input1 =Input(input_shape1+[1],name='input1')
    input2 =Input(input_shape2,name='input2')
    input3=Input(input_shape2,name='input3')

    # SeparateChannel Combined Convolutional Neural Networks
    x1=Conv2D(50,(2,8),padding='same', activation="relu", name="conv1_1", kernel_initializer='glorot_uniform')(input1)
    x2=Conv1D(50,8,padding='causal', activation="relu", name="conv1_2", kernel_initializer='glorot_uniform')(input2)
    x2_reshape=Reshape([-1,1024,50])(x2)
    x3=Conv1D(50,8,padding='causal', activation="relu", name="conv1_3", kernel_initializer='glorot_uniform')(input3)
    x3_reshape=Reshape([-1,1024,50],name='reshap2')(x3)
    x=concatenate([x2_reshape,x3_reshape],axis=1)
    x=Conv2D(50,(1,8),padding='same', activation="relu", name="conv2", kernel_initializer='glorot_uniform')(x)
    x=concatenate([x1,x])
    x=Conv2D(100,(2,5),padding='valid', activation="relu", name="conv4", kernel_initializer='glorot_uniform')(x)
    #LSTM Unit
    # batch_size,64,2
    x= Reshape(target_shape=((1020,100)),name='reshape')(x)
    x = CuDNNLSTM(units=128,return_sequences = True)(x)
    x = CuDNNLSTM(units=128)(x)

    #DNN
    x = Dense(128,activation='selu',name='fc1')(x)
    x=Dropout(dr)(x)
    x = Dense(128,activation='selu',name='fc2')(x)
    x=Dropout(dr)(x)
    x = Dense(classes,activation='softmax',name='softmax')(x)

    model = Model(inputs = [input1,input2,input3],outputs = x)

    # Load weights.
    if weights is not None:
        model.load_weights(weights)
    
    return model


import keras
from keras.utils.vis_utils import plot_model
if __name__ == '__main__':
    model = MCLDNN(None,classes=10)

    adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=adam)
    plot_model(model, to_file='model.png',show_shapes=True) # print model
    print('models layers:', model.layers)
    print('models config:', model.get_config())
    print('models summary:', model.summary())