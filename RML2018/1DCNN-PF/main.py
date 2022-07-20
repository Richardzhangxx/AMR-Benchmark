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
import rmlmodels.DCNNPF as mcl
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
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

classes = ['OOK',
               '4ASK',
               '8ASK',
               'BPSK',
               'QPSK',
               '8PSK',
               '16PSK',
               '32PSK',
               '16APSK',
               '32APSK',
               '64APSK',
               '128APSK',
               '16QAM',
               '32QAM',
               '64QAM',
               '128QAM',
               '256QAM',
               'AM-SSB-WC',
               'AM-SSB-SC',
               'AM-DSB-WC',
               'AM-DSB-SC',
               'FM',
               'GMSK',
               'OQPSK']
# def train(from_filename = '/media/norm_XYZ_1024_128k.hdf5',weight_file='weights/norm_res-like-128k.wts.h5',init_weight_file=None):
from_filename ='/DATASET.hdf5'
f = h5py.File(from_filename,'r')  # 打开h5文件
X = f['X'][:,:,:]  # ndarray(2555904*1024*2)
Y = f['Y'][:,:]  # ndarray(2M*24)
Z = f['Z'][:]  # ndarray(2M*1)
# [N,1024,2]
in_shp = X[0].shape
n_examples = X.shape[0]
n_train = int(n_examples * 0.6)
n_val = int(n_examples * 0.2)
train_idx = list(np.random.choice(range(0, n_examples), size=n_train, replace=False))
val_idx =list(np.random.choice(list(set(range(0,n_examples))-set(train_idx)), size=n_val, replace=False))
test_idx = list(set(range(0, n_examples)) - set(train_idx)-set(val_idx))
np.random.shuffle(train_idx)
np.random.shuffle(val_idx)
np.random.shuffle(test_idx)
X_train = X[train_idx]
Y_train = Y[train_idx]
X_val = X[val_idx]
Y_val = Y[val_idx]
X_test = X[test_idx]
Y_test = Y[test_idx]
Z_test = Z[test_idx]
X_train,X_val,X_test = to_amp_phase(X_train,X_val,X_test,1024)

X1_train=X_train[:,:,0]
X1_test=X_test[:,:,0]
X1_val=X_val[:,:,0]
X2_train=X_train[:,:,1]
X2_test=X_test[:,:,1]
X2_val=X_val[:,:,1]


# Set up some params
nb_epoch = 10000     # number of epochs to train on
batch_size = 400  # training batch size

# Build framework (model)
model=mcl.DCNNPF()
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
model.summary()

# Train the framework (model)
filepath = 'weights/weights.h5'
history = model.fit([X1_train,X2_train],
    Y_train,
    batch_size=batch_size,
    epochs=nb_epoch,
    verbose=2,
    validation_data=([X1_val,X2_val],Y_val),
    callbacks = [
                keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='auto'),
                keras.callbacks.ReduceLROnPlateau(monitor='val_loss',factor=0.5,verbose=1,patince=5,min_lr=0.0000001),
                keras.callbacks.EarlyStopping(monitor='val_loss', patience=50, verbose=1, mode='auto')
                ]
                    )

# We re-load the best weights once training is finished
# model.save_weights(filepath)
mltools.show_history(history)

# Show simple version of performance
score = model.evaluate([X1_test,X2_test], Y_test, verbose=1, batch_size=batch_size)
print(score)
def predict(model):
    # (mods,snrs,lbl),(X_train,Y_train),(X_test,Y_test),(train_idx,test_idx) = \
    #     rmldataset2016.load_data()
    model.load_weights(filepath)
    # Plot confusion matrix
    test_Y_hat = model.predict([X1_test,X2_test], batch_size=batch_size)
    cm, right, wrong = mltools.calculate_confusion_matrix(Y_test, test_Y_hat, classes)
    acc = round(1.0 * right / (right + wrong), 4)
    print('Overall Accuracy:%.2f%s / (%d + %d)' % (100 * acc, '%', right, wrong))
    # mltools.plot_confusion_matrix(cm, labels=['32PSK',
    #        '16APSK',
    #        '32QAM',
    #        'FM',
    #        'GMSK',
    #        '32APSK',
    #        'OQPSK',
    #        '8ASK',
    #        'BPSK',
    #        '8PSK',
    #        'AM-SSB-SC',
    #        '4ASK',
    #        '16PSK',
    #        '64APSK',
    #        '128QAM',
    #        '128APSK',
    #        'AM-DSB-SC',
    #        'AM-SSB-WC',
    #        '64QAM',
    #        'QPSK',
    #        '256QAM',
    #        'AM-DSB-WC',
    #        'OOK',
    #        '16QAM'],save_filename='figure/lstm3_total_confusion.png')
    mltools.plot_confusion_matrix(cm, labels=['OOK',
               '4ASK',
               '8ASK',
               'BPSK',
               'QPSK',
               '8PSK',
               '16PSK',
               '32PSK',
               '16APSK',
               '32APSK',
               '64APSK',
               '128APSK',
               '16QAM',
               '32QAM',
               '64QAM',
               '128QAM',
               '256QAM',
               'AM-SSB-WC',
               'AM-SSB-SC',
               'AM-DSB-WC',
               'AM-DSB-SC',
               'FM',
               'GMSK',
               'OQPSK'], save_filename='figure/lstm3_total_confusion.png')
    mltools.calculate_acc_cm_each_snr(Y_test, test_Y_hat, Z_test, classes, min_snr=0)
predict(model)

