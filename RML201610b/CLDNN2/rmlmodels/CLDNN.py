"""CLDNNLike model for RadioML.

# Reference:

- [CONVOLUTIONAL,LONG SHORT-TERM MEMORY, FULLY CONNECTED DEEP NEURAL NETWORKS ]

Adapted from code contributed by Mika.
"""
import os


from keras.models import Model
from keras.layers import Input,Dense,Conv1D,MaxPool1D,ReLU,Dropout,Softmax,concatenate,Flatten,Reshape,MaxPool2D,LSTM,Activation, CuDNNLSTM
from keras.layers.convolutional import Conv2D
from keras.layers import CuDNNLSTM


def CLDNN(weights=None,
             input_shape1=[2,128],
             classes=10,
             **kwargs):
    if weights is not None and not (os.path.exists(weights)):
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization), '
                         'or the path to the weights file to be loaded.')

    dr = 0.6  # dropout rate (%) 卷积层部分  https://keras-cn.readthedocs.io/en/latest/layers/convolutional_layer/#conv2d

    # 这里使用keras的函数式编程 http://keras-cn.readthedocs.io/en/latest/getting_started/functional_API/
    # Reshape [N,2,128] to [N,1,2,128] on input
    input_x = Input(input_shape1 + [1], name='input')

    # 卷积核维度(输出尺度),卷积核的宽度和长度,“valid”代表只进行有效的卷积，即对边界数据不处理,
    # 层权重weights的初始化函数
    # channels_first corresponds to inputs with shape (batch, channels, height, width).

    x = Conv2D(256, (1, 3), activation="relu", name="conv1", init='glorot_uniform')(input_x)  # (b,c,h,w) (b,h,w,c)
    x = Dropout(dr)(x)
    x = Conv2D(256, (2, 3), activation="relu", name="conv2", init='glorot_uniform')(x)  # (b,c,h,w) (b,h,w,c)
    x = Dropout(dr)(x)
    x = Conv2D(80, (1, 3), activation="relu", name="conv3", init='glorot_uniform')(x)  # (b,c,h,w) (b,h,w,c)
    x = Dropout(dr)(x)
    x = Conv2D(80, (1, 3), activation="relu", name="conv4", init='glorot_uniform')(x)  # (b,c,h,w) (b,h,w,c)
    x = Dropout(dr)(x)
    # 形如（samples，timesteps，input_dim）的3D张量
    x1 = Reshape((80, 120))(x)
    lstm_out = CuDNNLSTM(units=50)(x1)
    # 当 输出为250的时候正确里更高
    # lstm_out = LSTM(250, input_dim=input_dim, input_length=timesteps)(concat)

    # layer_Flatten = Flatten()(lstm_out)
    x = Dense(128, activation='relu', name="dense1")(lstm_out)
    x = Dropout(dr)(x)
    x = Dense(10, name="dense2")(x)
    output = Activation('softmax')(x)

    model = Model(inputs=input_x, outputs=output)

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
    print('models layers:', model.layers)
    print('models config:', model.get_config())
    print('models summary:', model.summary())