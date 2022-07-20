# Import all the things we need ---
#   by setting env variables before Keras import you can set up which backend and which GPU it uses
import os,random
os.environ["KERAS_BACKEND"] = "tensorflow"
# os.environ["THEANO_FLAGS"]  = "device=gpu%d"%(0)
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import numpy as np
import matplotlib
#matplotlib.use('Tkagg')
import matplotlib.pyplot as plt 
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
#from matplotlib import pyplot as plt
import pickle, random, sys,h5py
import keras
import keras.backend as K
from keras.callbacks import LearningRateScheduler,TensorBoard
from keras.regularizers import *
from keras.optimizers import adam
from keras.models import model_from_json
#from keras.utils.vis_utils import plot_model
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
import mltools
import rmlmodels.CuDNNLSTMModel as culstm

from keras.utils.np_utils import to_categorical
import pandas as pd

def to_amp_phase(X_train, X_val, X_test, nsamples):
    X_train_cmplx = X_train[:, :,0] + 1j * X_train[:, :,1]
    X_val_cmplx = X_val[:, :,0] + 1j * X_val[:, :,1]
    X_test_cmplx = X_test[:, :,0] + 1j * X_test[:, :,1]

    X_train_amp = np.abs(X_train_cmplx)
    X_train_ang = np.arctan2(X_train[:, :,1], X_train[:, :,0]) / np.pi

    X_train_amp = np.reshape(X_train_amp, (-1, 1, nsamples))
    X_train_ang = np.reshape(X_train_ang, (-1, 1, nsamples))

    X_train = np.concatenate((X_train_amp, X_train_ang), axis=1)
    X_train = np.transpose(np.array(X_train), (0, 2, 1))

    X_val_amp = np.abs(X_val_cmplx)
    X_val_ang = np.arctan2(X_val[:, :,1], X_val[:, :,0]) / np.pi

    X_val_amp = np.reshape(X_val_amp, (-1, 1, nsamples))
    X_val_ang = np.reshape(X_val_ang, (-1, 1, nsamples))

    X_val = np.concatenate((X_val_amp, X_val_ang), axis=1)
    X_val = np.transpose(np.array(X_val), (0, 2, 1))

    X_test_amp = np.abs(X_test_cmplx)
    X_test_ang = np.arctan2(X_test[:, :,1], X_test[:, :,0]) / np.pi

    X_test_amp = np.reshape(X_test_amp, (-1, 1, nsamples))
    X_test_ang = np.reshape(X_test_ang, (-1, 1, nsamples))

    X_test = np.concatenate((X_test_amp, X_test_ang), axis=1)
    X_test = np.transpose(np.array(X_test), (0, 2, 1))
    return (X_train, X_val, X_test)

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
train=train.swapaxes(1,2)

data2 = h5py.File('/home/neural/ZhangFuXin/AMR/tranining/HisarMod2019.1/Test/test.mat','r')
test=data2['data_save'][:]
test=test.swapaxes(0,2)
test=test.swapaxes(1,2)

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
X_train,X_val,X_test = to_amp_phase(X_train,X_val,X_test,1024)

# Set up some params
nb_epoch =  10000    # number of epochs to train on
batch_size = 400  # training batch size
print(batch_size)

model = culstm.LSTMModel()
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
model.summary()


filepath = 'weights/weights.h5'
history = model.fit(X_train,
    Y_train,
    batch_size=batch_size,
    epochs=nb_epoch,
    verbose=2,
    validation_data=(X_val,Y_val),
    callbacks = [
                keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='auto'),
                keras.callbacks.ReduceLROnPlateau(monitor='val_loss',factor=0.5,verbose=1,patince=5,min_lr=0.000001),
                keras.callbacks.EarlyStopping(monitor='val_loss', patience=30, verbose=1, mode='auto')
                #keras.callbacks.TensorBoard(log_dir='./logs/',histogram_freq=1,write_graph=False,write_grads=1,write_images=False,update_freq='epoch')
                ]
                    )

mltools.show_history(history)

#Show simple version of performance
score = model.evaluate(X_test, Y_test, verbose=1, batch_size=batch_size)
print(score)



def predict(model):
    # (mods,snrs,lbl),(X_train,Y_train),(X_test,Y_test),(train_idx,test_idx) = \
    #     rmldataset2016.load_data()
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