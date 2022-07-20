import numpy as np
import h5py
import sklearn.preprocessing
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

#   切分X的采样长度，并归一化到单位能量，保存成文件
def sliceX_normalized(from_filename = '/media/GOLD_XYZ_OSC.0001_1024.hdf5',slice_len = 1024,to_filename='/media/XYZ_1024_2m_norm.hdf5'):
    # file = '/media/GOLD_XYZ_OSC.0001_1024.hdf5'
    from_file = h5py.File(from_filename, 'r')  # 打开h5文件
    X = from_file['X']
    Y = from_file['Y']
    Z = from_file['Z']
    to_file = h5py.File(to_filename,'w')
    X_slice = to_file.create_dataset(shape=(X.shape[0],slice_len,X.shape[2]),# 数据集的维度
                                     maxshape = (None,slice_len,X.shape[2]), #数据集的允许最大维度
                                     dtype=np.float32,compression='gzip',name='X',#数据类型、是否压缩，以及数据集的名字
                                     chunks=(1,slice_len,X.shape[2])) #分块存储，每一分块的大小
    Y_slice = to_file.create_dataset(shape=Y.shape,  # 数据集的维度
                                     maxshape=(None,Y.shape[1]),  # 数据集的允许最大维度
                                     dtype=np.uint8, compression='gzip', name='Y',
                                     chunks=(1,Y.shape[1]))
    Z_slice = to_file.create_dataset(shape=Z.shape,  # 数据集的维度
                                     dtype=np.int, compression='gzip', name='Z',
                                     chunks=None)
    batch_size = 4096
    for i in range(int(X.shape[0]/batch_size)):
    # for i in range(5):
        batch_X = X[i*batch_size:(i+1)*batch_size,0:slice_len,:]    #batch_sample{ndarray[batch_size,slice_len,2],ndarray[4096,512,2]}
        for j in range(batch_X.shape[0]):                           #each sample{ndarray[slice_len,2],ndarray[512,2]}
            x = batch_X[j]
            # x = sklearn.preprocessing.scale(x,axis=0)
            n_s0,n_s1 = x.shape[0],x.shape[1]
            x = np.reshape(x,(n_s0*n_s1,))
            x = sklearn.preprocessing.minmax_scale(x,feature_range=(-10,10))
            x = np.reshape(x,(n_s0,n_s1))
            batch_X[j] = x
        X_slice[i*batch_size:(i+1)*batch_size,:,:] = batch_X
        Y_slice[i * batch_size:(i + 1) * batch_size, :] = Y[i * batch_size:(i + 1) * batch_size, :]
        Z_slice[i * batch_size:(i + 1) * batch_size]    = Z[i * batch_size:(i + 1) * batch_size]
        print('write batch {}-{}'.format(i*batch_size,(i+1)*batch_size))
    from_file.close()
    to_file.close()

#   采样数据，分块随机打乱顺序，保存成文件
def subsample_data_2018_tofile(from_filename="/media/XYZ_1024_2m_norm.hdf5", sample_rate=1/12,to_filename="/media/norm_XYZ_1024_512k.hdf5"):
    f = h5py.File(from_filename, 'r')  # 打开h5文件
    X = f['X']  # ndarray(2555904*512*2)
    Y = f['Y']  # ndarray(2M*24)
    Z = f['Z']  # ndarray(2M*1)
    to_file = h5py.File(to_filename,'w')
    n_subsample = int(X.shape[0] * sample_rate)
    X_slice = to_file.create_dataset(shape=(n_subsample,X.shape[1],X.shape[2]),
                                     maxshape = (None,X.shape[1],X.shape[2]),
                                     dtype=np.float32,compression='gzip',name='X',
                                     chunks=(1,X.shape[1],X.shape[2]))
    Y_slice = to_file.create_dataset(shape=(n_subsample,Y.shape[1]),
                                     maxshape=(None,Y.shape[1]),
                                     dtype=np.uint8, compression='gzip', name='Y',
                                     chunks=(1,Y.shape[1]))
    Z_slice = to_file.create_dataset(shape=(n_subsample,1),
                                     dtype=np.int, compression='gzip', name='Z',
                                     chunks=None)
    snr_count = 26
    batch_size = int(X.shape[0]/snr_count)
    n_slice = int(batch_size * sample_rate)
    # random sample from each snr samples
    for i in range(snr_count):
        print('subsample the snr {}'.format(i))
        batch_X = X[i * batch_size:(i + 1) * batch_size]
        batch_Y = Y[i * batch_size:(i + 1) * batch_size]
        batch_Z = Z[i * batch_size:(i + 1) * batch_size]
        np.random.seed(2016)
        rand_idx = np.random.choice(np.arange(0,batch_size),size=n_slice,replace=False)
        X_slice[i * n_slice:(i + 1) * n_slice, :, :]    = batch_X[rand_idx]
        Y_slice[i * n_slice:(i + 1) * n_slice, :]       = batch_Y[rand_idx]
        Z_slice[i * n_slice:(i + 1) * n_slice]          = batch_Z[rand_idx]
    to_file.close()
    f.close()
    print('subsample complete.total samples:{}'.format(n_subsample))

#   提取hdf5数据集
def load_data_2018(from_filename="/media/XYZ.0001_0512_NORM.hdf5"):
    f = h5py.File(from_filename, 'r')
    X = f['X'][:]  # ndarray(2555904*1024*2)
    Y = f['Y'][:] # ndarray(2M*24)
    Z = f['Z'][:]  # ndarray(2M*1)
    f.close()
    return X,Y,Z
def data_analyse(filename = None):
    import pandas as pd
    X,_,_ = load_data_2018(from_filename=filename)
    pX = pd.Panel(X)
    pX_max = pX.max()
    print('max(I,Q) of the origin data is {}\r'.format(pX_max.max(axis=1)))
    pX_min = pX.min()
    print('min(I) of the origin data is {}\r'.format(pX_min.min(axis=1)))
    print('=========================================================')
    Xn, _, _ = load_data_2018(from_filename=filename)
    pXn = pd.Panel(Xn)
    pXn_max = pXn.max()
    print('max(I,Q) of the normalized data is {}'.format(pXn_max.max(axis=1)))
    pXn_min = pXn.min()
    print('min(I) of the normalized data is {}'.format(pXn_min.min(axis=1)))

def data_structure(filename = '/media/XYZ_1024_512k.hdf5'):
    # HDF5的读取：
    f = h5py.File(filename, 'r')  # 打开h5文件
    print('hdf5 keys include 3 DataSet:')
    for k in f.keys():
        print(k)
    # /X(2M,1024,2) samples X,DataSet
    # /Y(2M,24) Label Y
    # /Z(2M) SNR
    print('each dataset shape:')
    print('X:{},Y:{},Z:{}'.format(f['X'].shape, f['Y'].shape, f['Z'].shape))
    print('each dataset type:')
    print(type(f['X']), type(f['Y']), type(f['Z']))
    print('each dataset element shape:')
    print('X[i]:{},Y[i]:{},Z[i]:{}'.format(f['X'][0].shape, f['Y'][0].shape, f['Z'][0].shape))
    print('Y element like :')
    print(f['Y'][0])
    snrs = sorted(list(set(f['Z'][:, 0])))
    print('Z element include :')
    print(snrs)
    Z = f['Z']
    lenZ = int(4096 * 24 / 32)
    plt.figure()
    plt.title('Z snr distribute')
    plt.plot(np.arange(lenZ), Z[0:lenZ])
    plt.show()
    Y = f['Y']
    sumY = np.sum(Y, axis=0)
    print(sumY)
    # signal1(db1...db23),signal2...signal23
    # 106496 samples per mod signal
    # -20db--50db per mod signal
    Y_not_onehot = np.zeros((Y.shape[0]))
    Y_not_onehot = np.argmax(Y, axis=1)
    print(Y_not_onehot)
    lenY = Y_not_onehot.shape[0]
    plt.figure()
    plt.title('Y mod distribute')
    plt.plot(np.arange(lenY), Y_not_onehot[0:lenY])
    plt.show()
    # print(f['X'][0])
    oneX = f['X'][0]
    print(oneX)
    oneX = np.power(oneX, 2)
    print(oneX)
    oneX = np.sqrt(oneX[:, 0] + oneX[:, 1])
    print(oneX)
    print(np.sum(oneX))
    f.close()
if __name__ == '__main__':
    subsample_data_2018_tofile(from_filename=r"\GOLD_XYZ_OSC.0001_1024.hdf5", sample_rate=1/12,to_filename=r"\XYZ_1024_1_12.hdf5")
    # sliceX_normalized(from_filename='/media/GOLD_XYZ_OSC.0001_1024.hdf5', slice_len=1024,to_filename='/media/XYZ_1024_2m_norm.hdf5')
    # data_analyse(filename='/media/minmax_norm_XYZ_1024_64k.hdf5')
    # data_structure()
    pass