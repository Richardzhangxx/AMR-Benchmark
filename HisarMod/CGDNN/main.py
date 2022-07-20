"Adapted from the code (https://github.com/leena201818/radiom) contributed by leena201818"
import os,random
os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
import matplotlib
#matplotlib.use('Tkagg')
import matplotlib.pyplot as plt 
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
import pickle,sys,h5py
import keras
import keras.backend as K
from keras.callbacks import LearningRateScheduler
from keras.regularizers import *
from keras.optimizers import adam
from keras.models import model_from_json
import mltools
import rmlmodels.CGDNN as mcl
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
from keras.models import Model
from keras.utils.np_utils import to_categorical
import pandas as pd
classes = ['BPSK',
               'QPSK',
               '8PSK',
               '16PSK',
               '32PSK',
               '64PSK',
               '4QAM',
               '8QAM',
               '16QAM',
               '32QAM',
               '64QAM',
               '128QAM',
               '256QAM',
               '2FSK',
               '4FSK',
               '8FSK',
               '16FSK',
               '4PAM',
               '8PAM',
               '16PAM',
               'AM-DSB',
               'AM-DSB-SC',
               'AM-USB',
               'AM-LSB',
                'FM',
                'PM']
##traindata
data1 = h5py.File('/home/neural/ZhangFuXin/AMR/tranining/HisarMod2019.1/Train/train.mat','r')
train=data1['data_save'][:]
train=train.swapaxes(0,2)

data2 = h5py.File('/home/neural/ZhangFuXin/AMR/tranining/HisarMod2019.1/Test/test.mat','r')
test=data2['data_save'][:]
test=test.swapaxes(0,2)

##label
train_labels = pd.read_csv('/home/neural/ZhangFuXin/AMR/tranining/HisarMod2019.1/Train/train_labels1.csv',header=None)
train_labels=np.array(train_labels)
train_labels = to_categorical(train_labels, num_classes=None)

test_labels = pd.read_csv('/home/neural/ZhangFuXin/AMR/tranining/HisarMod2019.1/Test/test_labels1.csv',header=None)
test_labels =np.array(test_labels)
test_labels = to_categorical(test_labels, num_classes=None)

##snr
train_snr=pd.read_csv('/home/neural/ZhangFuXin/AMR/tranining/HisarMod2019.1/Train/train_snr.csv',header=None)
train_snr=np.array(train_snr)

test_snr=pd.read_csv('/home/neural/ZhangFuXin/AMR/tranining/HisarMod2019.1/Test/test_snr.csv',header=None)
test_snr=np.array(test_snr)

# [N,1024,2]
n_examples = train.shape[0]
n_train = int(n_examples * 0.8)
n_val = int(n_examples * 0.2)
train_idx = list(np.random.choice(range(0, n_examples), size=n_train, replace=False))
val_idx = list(set(range(0, n_examples)) - set(train_idx))
np.random.shuffle(train_idx)
np.random.shuffle(val_idx)
X_train = train[train_idx]
Y_train = train_labels[train_idx]
X_val = train[val_idx]
Y_val = train_labels[val_idx]
X_test = test
Y_test = test_labels
Z_test = test_snr
X_train=np.expand_dims(X_train,axis=1)
X_test=np.expand_dims(X_test,axis=1)
X_val=np.expand_dims(X_val,axis=1)

# Set up some params
nb_epoch = 10000     # number of epochs to train on
batch_size = 400  # training batch size

# Build framework (model)
model=mcl.CGDNN()
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

model.summary()

# Train the framework (model)
filepath = r'weights/weights.h5'

history = model.fit(X_train,
    Y_train,
    batch_size=batch_size,
    epochs=nb_epoch,
    verbose=2,
    validation_data=(X_val,Y_val),
    # validation_data=([X_test,X1_test,X2_test],Y_test),
    callbacks = [
                keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='auto'),
                keras.callbacks.ReduceLROnPlateau(monitor='val_loss',factor=0.5,verbose=1,patince=5,min_lr=0.0000001),
                keras.callbacks.EarlyStopping(monitor='val_loss', patience=50, verbose=1, mode='auto'),
                # keras.callbacks.TensorBoard(histogram_freq=1,write_graph=True,write_images=True,batch_size=batch_size)
                ]
                    )
mltools.show_history(history)

# Show simple version of performance
score = model.evaluate(X_test, Y_test, verbose=1, batch_size=batch_size)
print(score)


def predict(model):
    model.load_weights(filepath)
    # Plot confusion matrix
    test_Y_hat = model.predict(X_test, batch_size=batch_size)
    cm, right, wrong = mltools.calculate_confusion_matrix(Y_test, test_Y_hat, classes)
    acc = round(1.0 * right / (right + wrong), 4)
    print('Overall Accuracy:%.2f%s / (%d + %d)' % (100 * acc, '%', right, wrong))
    mltools.plot_confusion_matrix(cm, labels=['BPSK',
                                              'QPSK',
                                              '8PSK',
                                              '16PSK',
                                              '32PSK',
                                              '64PSK',
                                              '4QAM',
                                              '8QAM',
                                              '16QAM',
                                              '32QAM',
                                              '64QAM',
                                              '128QAM',
                                              '256QAM',
                                              '2FSK',
                                              '4FSK',
                                              '8FSK',
                                              '16FSK',
                                              '4PAM',
                                              '8PAM',
                                              '16PAM',
                                              'AM-DSB',
                                              'AM-DSB-SC',
                                              'AM-USB',
                                              'AM-LSB',
                                              'FM',
                                              'PM'], save_filename='figure/lstm3_total_confusion.png')
    mltools.calculate_acc_cm_each_snr(Y_test, test_Y_hat, Z_test, classes, min_snr=-18)
predict(model)

