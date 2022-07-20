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

    
# def load_data(filename=r'E:\Richard_zhangxx\My_Research\AMR\Thesis_code\Thesis_code\data\RML2016.10b.dat'):
def load_data(filename=r'/home/neural/ZhangFuXin/AMR/tranining/RML2016.10b.dat'):
    Xd =pickle.load(open(filename,'rb'),encoding='iso-8859-1')#Xd(120W,2,128) 10calss*20SNR*6000samples
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
            train_idx+=list(np.random.choice(range(a*6000,(a+1)*6000), size=3600, replace=False))
            val_idx+=list(np.random.choice(list(set(range(a*6000,(a+1)*6000))-set(train_idx)), size=1200, replace=False))
            a+=1
    X = np.vstack(X)
    n_examples=X.shape[0]
    test_idx = list(set(range(0,n_examples))-set(train_idx)-set(val_idx))
    np.random.shuffle(train_idx)
    np.random.shuffle(val_idx)
    np.random.shuffle(test_idx)
    X_train = X[train_idx]
    X_val=X[val_idx]
    X_test =  X[test_idx]
    # transfor the label form to one-hot
    def to_onehot(yy):
        # yy1 = np.zeros([len(yy), max(yy)+1])
        yy1 = np.zeros([len(yy), len(mods)])
        yy1[np.arange(len(yy)), yy] = 1
        return yy1
    Y_train = to_onehot(list(map(lambda x: mods.index(lbl[x][0]), train_idx)))
    Y_val=to_onehot(list(map(lambda x: mods.index(lbl[x][0]), val_idx)))
    Y_test = to_onehot(list(map(lambda x: mods.index(lbl[x][0]),test_idx)))

    print(X_train.shape)
    print(X_val.shape)
    print(X_test.shape)
    print(Y_train.shape)
    print(Y_val.shape)
    print(Y_test.shape)
    return (mods,snrs,lbl),(X_train,Y_train),(X_val,Y_val),(X_test,Y_test),(train_idx,val_idx,test_idx)

if __name__ == '__main__':
    (mods, snrs, lbl), (X_train, Y_train),(X_val,Y_val), (X_test, Y_test), (train_idx,val_idx,test_idx) = load_data()
