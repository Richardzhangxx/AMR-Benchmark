"Adapted from the code (https://github.com/leena201818/radiom) contributed by leena201818"
import pickle
import numpy as np
from numpy import linalg as la

maxlen=128
def l2_normalize(x, axis=-1):
    y = np.max(np.sum(x ** 2, axis, keepdims=True), axis, keepdims=True)
    return x / np.sqrt(y)

def norm_pad_zeros(X_train,nsamples):
    print ("Pad:", X_train.shape)
    for i in range(X_train.shape[0]):
        X_train[i,:,0] = X_train[i,:,0]/la.norm(X_train[i,:,0],2)
    return X_train


def to_amp_phase(X_train,X_val,X_test,nsamples):
    X_train_cmplx = X_train[:,0,:] + 1j* X_train[:,1,:]
    X_val_cmplx = X_val[:,0,:] + 1j* X_val[:,1,:]
    X_test_cmplx = X_test[:,0,:] + 1j* X_test[:,1,:]

    X_train_amp = np.abs(X_train_cmplx)
    X_train_ang = np.arctan2(X_train[:,1,:],X_train[:,0,:])/np.pi


    X_train_amp = np.reshape(X_train_amp,(-1,1,nsamples))
    X_train_ang = np.reshape(X_train_ang,(-1,1,nsamples))

    X_train = np.concatenate((X_train_amp,X_train_ang), axis=1)
    X_train = np.transpose(np.array(X_train),(0,2,1))

    X_val_amp = np.abs(X_val_cmplx)
    X_val_ang = np.arctan2(X_val[:,1,:],X_val[:,0,:])/np.pi


    X_val_amp = np.reshape(X_val_amp,(-1,1,nsamples))
    X_val_ang = np.reshape(X_val_ang,(-1,1,nsamples))

    X_val = np.concatenate((X_val_amp,X_val_ang), axis=1)
    X_val = np.transpose(np.array(X_val),(0,2,1))

    X_test_amp = np.abs(X_test_cmplx)
    X_test_ang = np.arctan2(X_test[:,1,:],X_test[:,0,:])/np.pi


    X_test_amp = np.reshape(X_test_amp,(-1,1,nsamples))
    X_test_ang = np.reshape(X_test_ang,(-1,1,nsamples))

    X_test = np.concatenate((X_test_amp,X_test_ang), axis=1)
    X_test = np.transpose(np.array(X_test),(0,2,1))
    return (X_train,X_val,X_test)
#
# def load_data(filename=r'E:\Richard_zhangxx\My_Research\AMR\Thesis_code\Thesis_code\data\RML2016.10a_dict.pkl'):
def load_data(filename=r'/home/neural/ZhangFuXin/AMR/tranining/RML2016.10a_dict.pkl'):
    Xd =pickle.load(open(filename,'rb'),encoding='iso-8859-1')#Xd(1cd 20W,2,128) 10calss*20SNR*6000samples
    mods,snrs = [sorted(list(set([k[j] for k in Xd.keys()]))) for j in [0,1] ] #mods['8PSK', 'AM-DSB', 'BPSK', 'CPFSK', 'GFSK', 'PAM4', 'QAM16', 'QAM64', 'QPSK', 'WBFM']
    X = []
    lbl = []
    train_idx=[]
    val_idx=[]
    np.random.seed(2016)
    a=0

    for mod in mods:
        for snr in snrs:
            X.append(Xd[(mod,snr)])     #ndarray(6000,2,128)
            for i in range(Xd[(mod,snr)].shape[0]):
                lbl.append((mod,snr))
            train_idx+=list(np.random.choice(range(a*1000,(a+1)*1000), size=600, replace=False))
            val_idx+=list(np.random.choice(list(set(range(a*1000,(a+1)*1000))-set(train_idx)), size=200, replace=False))
            a+=1
    X = np.vstack(X)
    # print(len(lbl))
    # transfer I/Q to amplitude and phase
    # X_amplitude=np.sqrt(np.square(X[:,0,:])+np.square(X[:,1,:]))
    # X_phase=np.arctan(X[:,1,:]/X[:,0,:])
    # # L2 normalized for X_amplitude
    # X_amplitude=l2_normalize(X_amplitude)
    # # Normalized [-Π/2，Π/2] to [-1,1]
    # for i in range(X_phase.shape[0]):
    #     k=2/(X_phase[i,:].max()-X_phase[i,:].min())
    #     X_phase[i,:]=-1+k*(X_phase[i,:]-X_phase[i,:].min())
    # print(X_phase.min())
    # X =np.stack((X_amplitude,X_phase),axis=1) #x_vp(220000,2,128)
    # Scramble the order between samples
    # and get the serial number of training, validation, and test sets
    n_examples=X.shape[0]
    test_idx = list(set(range(0,n_examples))-set(train_idx)-set(val_idx))
    np.random.shuffle(train_idx)
    np.random.shuffle(val_idx)
    np.random.shuffle(test_idx)
    X_train = X[train_idx]
    X_val=X[val_idx]
    X_test =  X[test_idx]
    # print(len(train_idx))
    # print(len(val_idx))
    # print(len(test_idx))

    # transfor the label form to one-hot
    def to_onehot(yy):
        # yy1 = np.zeros([len(yy), max(yy)+1])
        yy1 = np.zeros([len(yy), len(mods)])
        yy1[np.arange(len(yy)), yy] = 1
        return yy1
    Y_train = to_onehot(list(map(lambda x: mods.index(lbl[x][0]), train_idx)))
    Y_val=to_onehot(list(map(lambda x: mods.index(lbl[x][0]), val_idx)))
    Y_test = to_onehot(list(map(lambda x: mods.index(lbl[x][0]),test_idx)))

    # X_train,X_val,X_test = to_amp_phase(X_train,X_val,X_test,128)

    #
    X_train = X_train.swapaxes(2, 1)
    X_val = X_val.swapaxes(2, 1)
    X_test = X_test.swapaxes(2, 1)
    # X_train = X_train[:,:maxlen,:]
    # X_val = X_val[:,:maxlen,:]
    # X_test = X_test[:,:maxlen,:]
    #
    # X_train = norm_pad_zeros(X_train,maxlen)
    # X_val = norm_pad_zeros(X_val,maxlen)
    # X_test = norm_pad_zeros(X_test,maxlen)


    print(X_train.shape)
    print(X_val.shape)
    print(X_test.shape)
    print(Y_train.shape)
    print(Y_val.shape)
    print(Y_test.shape)
    # X_train=X_train.swapaxes(2,1)
    # X_val=X_val.swapaxes(2,1)
    # X_test=X_test.swapaxes(2,1)
    return (mods,snrs,lbl),(X_train,Y_train),(X_val,Y_val),(X_test,Y_test),(train_idx,val_idx,test_idx)

if __name__ == '__main__':
    (mods, snrs, lbl), (X_train, Y_train),(X_val,Y_val), (X_test, Y_test), (train_idx,val_idx,test_idx) = load_data()
