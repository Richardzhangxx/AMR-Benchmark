import pickle
import numpy as np



# def load_data(filename=r'/home/xujialang/ZhangFuXin/AMR/tranining/RML2016.10a_dict.pkl'):
def load_data(filename=r'F:\Richard_zhangxx\My_Research\AMR\Thesis_code\Thesis_code1080ti-experiment\data\RML2016.10a_dict.pkl'):
#    Xd1 = pickle.load(open(filename1,'rb'),encoding='iso-8859-1')#Xd1.keys() mod中没有AM-SSB Xd1(120W,2,128)
    Xd =pickle.load(open(filename,'rb'),encoding='iso-8859-1')#Xd2(22W,2,128)
    mods,snrs = [sorted(list(set([k[j] for k in Xd.keys()]))) for j in [0,1] ] 
    X = []
    # X2=[]
    lbl = []
    # lbl2=[]
    train_idx=[]
    val_idx=[]
    np.random.seed(2016)
    a=0

    for mod in mods:
        for snr in snrs:
            X.append(Xd[(mod,snr)])     #ndarray(1000,2,128)
            for i in range(Xd[(mod,snr)].shape[0]):
                lbl.append((mod,snr))
            train_idx+=list(np.random.choice(range(a*1000,(a+1)*1000), size=600, replace=False))
            val_idx+=list(np.random.choice(list(set(range(a*1000,(a+1)*1000))-set(train_idx)), size=200, replace=False))
            a+=1
    X = np.vstack(X)                    #(220000,2,128)  mods * snr * 1000,total 220000 samples            #(162060,2,128)
    print(len(lbl))
    n_examples=X.shape[0]
    # n_test=X2.shape[0] 
    test_idx = list(set(range(0,n_examples))-set(train_idx)-set(val_idx))
    np.random.shuffle(train_idx)
    np.random.shuffle(val_idx)
    np.random.shuffle(test_idx)
    # test_idx=np.random.choice(range(0,n_test),size=n_test,replace=False)
    X_train = X[train_idx]
    X_val=X[val_idx]
    X_test =  X[test_idx]
    print(len(train_idx))
    print(len(val_idx))
    print(len(test_idx))

    print(X_train.shape)
    print(X_val.shape)
    print(X_test.shape)
    def to_onehot(yy):
        # yy1 = np.zeros([len(yy), max(yy)+1])
        yy1 = np.zeros([len(yy), len(mods)])
        yy1[np.arange(len(yy)), yy] = 1
        return yy1

    # yy = list(map(lambda x: mods.index(lbl[x][0]), train_idx))

    Y_train = to_onehot(list(map(lambda x: mods.index(lbl[x][0]), train_idx)))
    Y_val=to_onehot(list(map(lambda x: mods.index(lbl[x][0]), val_idx)))
    Y_test = to_onehot(list(map(lambda x: mods.index(lbl[x][0]),test_idx)))
    
    print(Y_train.shape)
    print(Y_val.shape)
    print(Y_test.shape)
    return (mods,snrs,lbl),(X_train,Y_train),(X_val,Y_val),(X_test,Y_test),(train_idx,val_idx,test_idx)

if __name__ == '__main__':
    (mods, snrs, lbl), (X_train, Y_train),(X_val,Y_val), (X_test, Y_test), (train_idx,val_idx,test_idx) = load_data()
