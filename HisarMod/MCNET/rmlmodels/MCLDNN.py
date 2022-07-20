import os

WEIGHTS_PATH = ('resnet_like_weights_tf_dim_ordering_tf_kernels.h5')

from keras.models import Model
from keras.layers import Input,Dense,Conv1D,MaxPool1D,ReLU,Dropout,Softmax,concatenate,Flatten,Reshape,LeakyReLU,Subtract,CuDNNGRU
from keras.layers.convolutional import Conv2D
from keras.layers import CuDNNLSTM,AveragePooling2D,MaxPool2D,Add


def MCNET(weights=None,
           input_shape=[2,1024],
           input_shape2=[1024],
           classes=26,
           **kwargs):
    if weights is not None and not (os.path.exists(weights)):
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization), '
                         'or the path to the weights file to be loaded.')

    input = Input(input_shape + [1], name='input1')
    x1 = Conv2D(64, (3, 7), strides=(1, 2), activation="relu", padding='same', name="conv1_1",
                    kernel_initializer='glorot_uniform')(
            input)
    x1 = MaxPool2D((1, 3), strides=(1, 2), padding='same', name="pool1_1", data_format='channels_last')(x1)

        # preblock
    x2 = Conv2D(32, (3, 1), strides=(1, 1), activation="relu", padding='same', name="conv2_1",
                    kernel_initializer='glorot_uniform')(x1)
    x2 = AveragePooling2D((1, 3), strides=(1, 2), padding='same', name="pool2_1", data_format='channels_last')(x2)
    x22 = Conv2D(32, (1, 3), strides=(1, 2), activation="relu", padding='same', name="conv2_2",
                     kernel_initializer='glorot_uniform')(x1)
    x222 = concatenate([x2, x22], axis=-1)

        # skip
    xx1 = Conv2D(128, (1, 1), strides=(1, 2), activation="relu", padding='same', name="conv111",
                    kernel_initializer='glorot_uniform')(x222)
    xx1 = MaxPool2D((1, 3), strides=(1, 2), padding='same', data_format='channels_last', name="pool2_2")(xx1)

        # Mblockp1
    x3 = MaxPool2D((1, 3), strides=(1, 2), padding='same', data_format='channels_last', name="pool3_1")(x222)
    x3 = Conv2D(32, (1, 1), strides=(1, 1), padding='same', activation="relu", name="conv3_1",
                    kernel_initializer='glorot_uniform')(x3)
    x31 = Conv2D(48, (3, 1), strides=(1, 1), padding='same', activation="relu", name="conv3_2",
                     kernel_initializer='glorot_uniform')(x3)
    x31 = MaxPool2D((1, 3), strides=(1, 2), padding='same', name="pool3_2")(x31)
    x32 = Conv2D(48, (1, 3), strides=(1, 2), padding='same', activation="relu", name="conv3_3",
                     kernel_initializer='glorot_uniform')(x3)
    x33 = Conv2D(32, (1, 1), strides=(1, 2), padding='same', activation="relu", name="conv3_4",
                     kernel_initializer='glorot_uniform')(x3)
    x31 = concatenate([x31, x32], axis=-1)
    x333 = concatenate([x33, x31], axis=-1)

        # add1
    add1 = Add()([x333, xx1])
        # Mblock2
    x4 = Conv2D(32, (1, 1), strides=(1, 1), padding='same', activation="relu", name="conv4_1",
                    kernel_initializer='glorot_uniform')(add1)
    x41 = Conv2D(48, (3, 1), strides=(1, 1), padding='same', activation="relu", name="conv4_2",
                     kernel_initializer='glorot_uniform')(x4)
    x42 = Conv2D(48, (1, 3), strides=(1, 1), padding='same', activation="relu", name="conv4_3",
                     kernel_initializer='glorot_uniform')(x4)
    x43 = Conv2D(32, (1, 1), strides=(1, 1), padding='same', activation="relu", name="conv4_4",
                     kernel_initializer='glorot_uniform')(x4)
    x41 = concatenate([x41, x42], axis=-1)
    x444 = concatenate([x43, x41], axis=-1)

        # add2
    add2 = Add()([x444, add1])
        # Mblockp3
    x5 = Conv2D(32, (1, 1), strides=(1, 1), padding='same', activation="relu", name="conv5_1",
                    kernel_initializer='glorot_uniform')(add2)
    x51 = Conv2D(48, (3, 1), strides=(1, 1), padding='same', activation="relu", name="conv5_2",
                     kernel_initializer='glorot_uniform')(x5)
    x51 = MaxPool2D((1, 3), strides=(1, 2), padding='same', name="pool5_2")(x51)
    x52 = Conv2D(48, (1, 3), strides=(1, 2), padding='same', activation="relu", name="conv5_3",
                     kernel_initializer='glorot_uniform')(x5)
    x53 = Conv2D(32, (1, 1), strides=(1, 2), padding='same', activation="relu", name="conv5_4",
                     kernel_initializer='glorot_uniform')(x5)
    x51 = concatenate([x51, x52], axis=-1)
    x555 = concatenate([x53, x51], axis=-1)

        # add3
    ad3 = MaxPool2D((2, 2), strides=(1, 2), padding='same', name="pool5_3")(add2)
    add3 = Add()([x555, ad3])

        # Mblockp4
    x6 = Conv2D(32, (1, 1), strides=(1, 1), padding='same', activation="relu", name="conv6_1",
                    kernel_initializer='glorot_uniform')(add3)
    x61 = Conv2D(48, (3, 1), strides=(1, 1), padding='same', activation="relu", name="conv6_2",
                     kernel_initializer='glorot_uniform')(x6)
    x62 = Conv2D(48, (1, 3), strides=(1, 1), padding='same', activation="relu", name="conv6_3",
                     kernel_initializer='glorot_uniform')(x6)
    x63 = Conv2D(32, (1, 1), strides=(1, 1), padding='same', activation="relu", name="conv6_4",
                     kernel_initializer='glorot_uniform')(x6)
    x61 = concatenate([x61, x62], axis=-1)
    x666 = concatenate([x63, x61], axis=-1)

        # add4
    add4 = Add()([x666, add3])
        # Mblockp5
    x7 = Conv2D(32, (1, 1), strides=(1, 1), padding='same', activation="relu", name="conv7_1",
                    kernel_initializer='glorot_uniform')(add4)
    x71 = Conv2D(48, (3, 1), strides=(1, 1), padding='same', activation="relu", name="conv7_2",
                     kernel_initializer='glorot_uniform')(x7)
    x71 = MaxPool2D((1, 3), strides=(1, 2), padding='same', name="pool7_2")(x71)
    x72 = Conv2D(48, (1, 3), strides=(1, 2), padding='same', activation="relu", name="conv7_3",
                     kernel_initializer='glorot_uniform')(x7)
    x73 = Conv2D(32, (1, 1), strides=(1, 2), padding='same', activation="relu", name="conv7_4",
                     kernel_initializer='glorot_uniform')(x7)
    x71 = concatenate([x71, x72], axis=-1)
    x777 = concatenate([x73, x71], axis=-1)

        # add5
    ad5 = MaxPool2D((2, 2), strides=(1, 2), padding='same', name="pool7_3")(add4)
    add5 = Add()([x777, ad5])
        # Mblockp6
    x8 = Conv2D(32, (1, 1), strides=(1, 1), padding='same', activation="relu", name="conv8_1",
                    kernel_initializer='glorot_uniform')(add5)
    x81 = Conv2D(96, (3, 1), strides=(1, 1), padding='same', activation="relu", name="conv8_2",
                     kernel_initializer='glorot_uniform')(x8)
    x82 = Conv2D(96, (1, 3), strides=(1, 1), padding='same', activation="relu", name="conv8_3",
                     kernel_initializer='glorot_uniform')(x8)
    x83 = Conv2D(64, (1, 1), strides=(1, 1), padding='same', activation="relu", name="conv8_4",
                     kernel_initializer='glorot_uniform')(x8)
    x81 = concatenate([x81, x82], axis=-1)
    x888 = concatenate([x83, x81], axis=-1)

    x_con = concatenate([x888, add5], axis=-1)
    xout = AveragePooling2D((2, 8))(x_con)
    xout = Dropout(0.5)(xout)
    xout = Flatten()(xout)
    x = Dense(classes, activation='softmax', name='softmax')(xout)

    model = Model(inputs=input, outputs=x)

        # Load weights.
    if weights is not None:
            model.load_weights(weights)

    return model
import keras
from keras.utils.vis_utils import plot_model
if __name__ == '__main__':
    model = MCNET(None,classes=24)

    adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=adam)
    plot_model(model, to_file='model.png',show_shapes=True) # print model
    print('models layers:', model.layers)
    print('models config:', model.get_config())
    print('models summary:', model.summary())